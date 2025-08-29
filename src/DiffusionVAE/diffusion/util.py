import torch
import torch.nn.functional as F
import numpy as np


def patch_forward_guide(noisy, kpn_net, params={}, skip=512, padding=32):
    ''' Args:
        noisy: b c h w
        kpn_net:
    '''
    assert noisy.is_cuda
    pd = min(noisy.shape[-1], noisy.shape[-2])
    pd = skip - pd + padding if pd < skip else padding

    noisy = F.pad(noisy, (pd, pd, pd, pd), mode='reflect')
    guide_pad = F.pad(params['guide'], (pd, pd, pd, pd), mode='reflect')

    denoised = torch.zeros_like(noisy)[:, :3]  # assume output channel == 3

    _, _, H, W = noisy.shape
    shift = skip - padding * 2
    for i in np.arange(0, H, shift):
        for j in np.arange(0, W, shift):
            h_start, h_end = i, i + skip
            w_start, w_end = j, j + skip
            # print('\nidx', h_start, h_end, w_start, w_end)
            if h_end > H:
                h_end = H
                h_start = H - skip
            if w_end > W:
                w_end = W
                w_start = W - skip
            # print('\nidx2', h_start, h_end, w_start, w_end, H, W)
            patch = noisy[..., h_start: h_end, w_start: w_end]
            guide_patch = guide_pad[..., h_start: h_end, w_start: w_end]
            params['guide'] = guide_patch
            with torch.no_grad():
                input_var = patch
                out_var = kpn_net(input_var, **params)

            out = out_var
            denoised[..., h_start + padding: h_end - padding, w_start + padding: w_end - padding] = \
                out[..., padding:-padding, padding:-padding]
    return denoised[..., pd:-pd, pd:-pd]

