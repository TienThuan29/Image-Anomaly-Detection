import torch
import math
from torch import nn
from inspect import isfunction
from torch.functional import F
from diffusion.util import patch_forward_guide

""" Positional Encoding for timestep"""
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


""" Swish activation function """
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(
            self, dim, dim_out, *, nl_emb_dim=None, norm_groups=1,
            dropout_p=0, resname='ResnetBlockDY3h', with_attn=False # resname = 'ResnetBlockDY3h'
    ):
        super().__init__()
        self.with_attn = with_attn

        # Khởi tạo lớp ResnetBlockDY3h thông qua tên lớp - eval(resname)
        self.res_block = eval(resname)(dim, dim_out, nl_emb_dim, norm_groups=norm_groups, dropout=dropout_p)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb, kwargs={}):
        x = self.res_block(x, time_emb, **kwargs) # **kwargs : { 'guide' : x_init }
        if (self.with_attn):
            x = self.attn(x)
        return x


""" Conditional Intergration Module in pipleline """
class ResnetBlockDY3h(nn.Module):
    def __init__(
            self, dim, dim_out, nl_emb_dim=None,
            dropout=0, use_affine_level=False, norm_groups=1,
            nset=8 # N trong paper chính là nset trong code.
    ):
        super().__init__()
        self.noise_func = nn.Sequential(nn.Linear(nl_emb_dim, nset), Swish(), nn.Linear(nset, nset), )
        self.nset = nset
        self.dim_out = dim_out

        # đầu vào là input sau khi đi qua lớp Conv2d đầu tiên
        self.norm1 = nn.GroupNorm(norm_groups, dim)
        self.conv1 = nn.Conv2d(dim, dim_out, 3, padding=1)

        self.norm2 = nn.GroupNorm(norm_groups, dim_out)
        self.conv2 = nn.Sequential(
            # Conv2d(3 → 2*nset, k=1): nén/đổi kênh từ ảnh guide (3 kênh RGB) sang 2·nset kênh
            nn.Conv2d(3, self.nset * 2, 1),
            SimpleGate(),
            nn.Conv2d(self.nset, self.nset, kernel_size=3, padding=1),
        )

        self.spdyconv = nn.Conv2d(dim_out, dim_out * self.nset, kernel_size=3, padding=1, groups=self.nset)
        self.swish = Swish()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    """
        Forward flow của một layer trong model unet diffusion 
    """
    def forward(self, x, time_emb, guide):
        b, c, H, w = x.shape
        # print(time_emb.shape, self.noise_func)
        """ Time embed -> Linear -> activation -> Linear """
        attw = self.noise_func(time_emb).view(b, -1)
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.swish(h)
        h = self.norm2(h)
        # h = self.conv2(h, attw)

        """ AKGM - Adaptive Kernal Guidance Module """
        # An equivalent implementation of AKGM by using group conv
        # input: guide = x_init
        ratio = w / guide.shape[-1]
        guide = F.interpolate(guide, scale_factor=ratio, mode='bilinear', align_corners=False)

        # conv2 : bao gồm Conv2d -> Simplegate -> Conv2d
        # att_sp = feature_map_G * S
        # M      = att_sp
        att_sp = self.conv2(guide) * attw.view(b, self.nset, 1, 1)

        hset = self.spdyconv(h).view(b, self.dim_out, self.nset, H, w)
        h = torch.sum(hset * att_sp.unsqueeze(1), dim=2, keepdim=False)
        h = self.swish(h)
        # short cut của residual block
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


""" Main diffusion network - learn residual """
class DY3h(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            inner_channel: int,
            norm_groups: int,
            channel_mults: list,
            attn_res: list,
            res_blocks: int,
            dropout_p: float,
            image_size: int,
            with_noise_level_emb=True,
            resname='ResnetBlockDY3h'
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            # Time embedding in diffusion
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4), Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        """ Down Unet """
        downs = [nn.Conv2d(in_channel, inner_channel ,kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            # biến cờ (boolean) quyết định có bật self-attention ở tầng hiện tại hay không.
            # attn_res : list các resolution để bật self-attention
            use_attn = (now_res in attn_res)

            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(pre_channel, channel_mult, nl_emb_dim=noise_level_channel,
                                       norm_groups=norm_groups, dropout=dropout_p, with_attn=use_attn,
                                       resname=resname)
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult

            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs) # Full down layers unet

        """ Mid unet """
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, nl_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout_p, with_attn=True, resname=resname),
            ResnetBlocWithAttn(pre_channel, pre_channel, nl_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout_p, with_attn=False, resname=resname)
        ])


        """ Up Unet """
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(), channel_mult, nl_emb_dim=noise_level_channel,
                        norm_groups=norm_groups, dropout=dropout_p, with_attn=use_attn, resname=resname
                    )
                )
                pre_channel = channel_mult

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.prec = pre_channel
        # self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        # for dynamic networks
        dim = self.prec
        dim_out = default(out_channel, in_channel)
        # Final layer - KHÔNG có activation
        self.final_conv = nn.Sequential(
            nn.GroupNorm(1, dim), Swish(),
            nn.Dropout(dropout_p) if dropout_p != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )


    """ 'native_forward' là một flow chạy hoàn chỉnh của unet, Down -> Mid ->  Up"""
    def naive_forward(self, x, time, guide):
        # Tạo embedding thơi gian
        # time (integer) -> Positional Encoding -> MLP
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        feats = []  # value của skip-connection
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, {'guide': guide})
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, {'guide': guide})
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t, {'guide': guide})
            else:
                x = layer(x)

        return self.final_conv(x)  # No activation => Raw regression value


    def forward(self, x, time, guide):
        """
            x : torch.cat([x_in['SR'], x_noisy], dim=1)
            time : continuous_sqrt_alpha_cumprod
            guide : ({'guide': x_init}) là x_init là ảnh dự đoán ban đầu qua cục unet_1
        """
        _, _, h, w = x.shape

        """" 
            image_size : 128 
            h * w = 16,384 
        """

        if h * w > 1024 * 1024:  # guide image should be split to patches
            return patch_forward_guide(
                x, # [B, 6, H, W] : ảnh concat cùi với noise residual
                self.naiveforward,
                params={'time': time, 'guide': guide},
                skip=1024,
                padding=64
            )
        else:
            # return self.naiveforward(x, time, guide)
            fac = 32
            padh, padw = (h // fac + 1) * fac - h, (w // fac + 1) * fac - w
            x = F.pad(x, (0, padw, 0, padh), mode='reflect')
            guide = F.pad(guide, (0, padw, 0, padh), mode='reflect')
            return self.naiveforward(x, time, guide)[..., :-padh, :-padw]


""" UNetSeeInDark: Mạng dự đoán ban đầu (predictor) - tạo ra guidance """
""" kiến trúc Unet thuần có skip connection """
""" Input: ảnh cùi """
class UNetSeeInDark(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetSeeInDark, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        _, _, h, w = x.shape
        fac = 32
        padh, padw = (h // fac + 1) * fac - h, (w // fac + 1) * fac - w
        x = F.pad(x, (0, padw, 0, padh), mode='reflect')
        # print('unet ', x.shape, padh, padw)
        return self.naive_forward(x)[..., :-padh, :-padw]

    def naive_forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        # out = nn.functional.pixel_shuffle(conv10, 2)
        out = conv10
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x) # LeakyReLU(slope=0.2)
        return outt




