import torch
import os
import logging
import copy
import time
from torch import nn
from torch.functional import F
from collections import OrderedDict
from diffusion.model import networks
from diffusion.model.base_model import BaseModel
from config import load_config
from utils.optimizers import get_optimizer

logger = logging.getLogger('base')
config = load_config()

# training
_category = config.training.category
_phase = config.diffusion_model.phase
_finetune_norm = False
_optimizer_name = config.diffusion_model.optimizer_name
_lr = float(config.diffusion_model.lr)
_resume_checkpoint_path = config.diffusion_model.resume_checkpoint

# train result dir
_train_result_dir = config.diffusion_model.train_result_base_dir + _category + '/'
_pretrained_save_dir = config.diffusion_model.pretrained_save_base_dir + _category + '/'

# ema scheduler
_ema_scheduler = config.diffusion_model.ema_scheduler
_use_ema_scheduler = config.diffusion_model.ema_scheduler.use
_ema_decay = config.diffusion_model.ema_scheduler.ema_decay
_step_start_ema = config.diffusion_model.ema_scheduler.step_start_ema
_update_ema_every = config.diffusion_model.ema_scheduler.update_ema_every

# beta schedule for training
_train_beta_schedule = config.diffusion_model.beta_schedule.train
# beta schedule for testing/val
_test_beta_schedule = config.diffusion_model.beta_schedule.val

""" Ema gpu """
class EMA():
    # beta = hệ số giảm (decay)
    # Giá trị càng gần 1 → EMA thay đổi càng chậm (mượt hơn).
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DDPM(BaseModel):
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
            channels: int,
            loss_type: str
    ):
        super(DDPM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # đính nghĩa unet model từ network.py
        temp_model = networks.define_G(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            channel_mults=channel_mults,
            attn_res=attn_res,
            res_blocks=res_blocks,
            dropout_p=dropout_p,
            image_size=image_size,
            channels=channels,
            loss_type=loss_type
        )
        self.netG = self.set_device(temp_model)
        self.netG = self.netG.to(self.device)
        self.schedule_phase = None
        self.data = None

        if _use_ema_scheduler:
            #   ema_scheduler:
            #     use: true
            #     step_start_ema: 5000
            #     update_ema_every: 1
            #     ema_decay: 0.9999
            self.ema_scheduler = _ema_scheduler
            # Dùng deepcopy để tạo một mô hình mới với weights y hệt ban đầu, độc lập với netG.
            self.netG_EMA = copy.deepcopy(temp_model)
            # !!! Ema giữ một bản sao của model, làm tăng mức sử dụng vram khi được chuyển vào gpu
            self.netG_EMA = self.netG_EMA.to(self.device)
            # Khắc phục: chuyển vào cpu
            self.EMA = EMA(beta=_ema_decay)
        else:
            self.ema_scheduler = None

        self.set_loss()
        # print("_train_beta_schedule", _train_beta_schedule)
        self.set_new_noise_schedule(
            _train_beta_schedule['schedule'],
            _train_beta_schedule['n_timestep'],
            _train_beta_schedule['linear_start'],
            _train_beta_schedule['linear_end'],
            schedule_phase='train'
        )

        if _phase == 'train':
            self.netG.train()
            # find the parameters to optimize
            if _finetune_norm:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = get_optimizer(optimizer_name=_optimizer_name,params=optim_params,lr=_lr)
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()
        self.iter = 0

        self.clip_norm = None # self.opt.get('clip_norm', None)
        logger.info('* clip norm %s' % self.clip_norm)


    def feed_data(self, data):
        self.data = self.set_device(data)

    def _init_ema_from(self, model: nn.Module) -> nn.Module:
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.to('cpu')  # EMA trên CPU
        return ema

    """ Train  """
    def optimize_parameters(self):
        self.optG.zero_grad()

        # self.netG là diffusion model
        # Gọi forward() của diffusion model, tức là gọi p_losses()
        # end-to-end training -> return loss value
        l_pix = self.netG(self.data)  # l_pix : loss value

        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.clip_norm)
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        # print('self.ema_scheduler', self.ema_scheduler)
        if self.ema_scheduler is not None:
            if self.iter > self.ema_scheduler.step_start_ema and self.iter % self.ema_scheduler.update_ema_every == 0:
                self.EMA.update_model_average(self.netG_EMA, self.netG)
        self.iter += 1

    """ Test """
    def test(self, continous=False):
        self.netG.eval()
        pd = 64
        self.data['SR'] = F.pad(self.data['SR'], (pd, pd, pd, pd), mode='reflect')
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous
                )
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous
                )
        self.netG.train()
        self.SR = self.SR[..., pd:-pd, pd:-pd]
        self.data['SR'] = self.data['SR'][..., pd:-pd, pd:-pd]


    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()


    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)


    def set_new_noise_schedule(
            self, schedule: str, n_timestep: int,
            linear_start: float, linear_end: float,
            schedule_phase='train', force=False
    ):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase or force:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule, n_timestep, linear_start, linear_end, self.device
                )
            else:
                self.netG.set_new_noise_schedule(
                    schedule, n_timestep, linear_start, linear_end, self.device
                )


    def set_noise_schedule_for_training(self):
        self.set_new_noise_schedule(
            _train_beta_schedule['schedule'],
            _train_beta_schedule['n_timestep'],
            _train_beta_schedule['linear_start'],
            _train_beta_schedule['linear_end'],
            schedule_phase='train'
        )


    def set_noise_schedule_for_val(self,):
        self.set_new_noise_schedule(
            _test_beta_schedule['schedule'],
            _test_beta_schedule['n_timestep'],
            _test_beta_schedule['linear_start'],
            _test_beta_schedule['linear_end'],
            schedule_phase='val'
        )


    def save_network(self, epoch, iter_step, checkpoint_type="latest", loss_history=None, eval_history=None, best_loss=None, best_epoch=None):
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(_pretrained_save_dir):
            os.makedirs(_pretrained_save_dir)
            
        # Determine filename based on checkpoint type
        if checkpoint_type == "best":
            filename = f'diffusion_best.pth'
        elif checkpoint_type == "latest":
            filename = f'diffusion_latest.pth'
        elif checkpoint_type == "early_stop":
            filename = f'diffusion_early_stop_epoch_{epoch}.pth'
        else:
            filename = f'diffusion_{checkpoint_type}.pth'
            
        checkpoint_path = os.path.join(_pretrained_save_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'iter_step': iter_step,
            'model_state_dict': self.netG.state_dict(),
            'optimizer_state_dict': self.optG.state_dict(),
            'selfiter': self.iter,
            'checkpoint_type': checkpoint_type,
            'timestamp': time.time()
        }
        
        # Add training history if provided
        if loss_history is not None:
            checkpoint['loss_history'] = loss_history
        if eval_history is not None:
            checkpoint['eval_history'] = eval_history
        if best_loss is not None:
            checkpoint['best_loss'] = best_loss
        if best_epoch is not None:
            checkpoint['best_epoch'] = best_epoch
        
        # Add EMA model if available
        if _use_ema_scheduler:
            checkpoint['ema_model_state_dict'] = self.netG_EMA.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Saved diffusion model checkpoint: {checkpoint_path}')
        
        return checkpoint_path


    def load_network(self):
        if _resume_checkpoint_path and os.path.exists(_resume_checkpoint_path):
            logger.info(f'Loading diffusion model from: {_resume_checkpoint_path}')
            
            checkpoint = torch.load(_resume_checkpoint_path, map_location=self.device)
            
            # Load main model
            self.netG.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer if in training phase
            if _phase == 'train' and 'optimizer_state_dict' in checkpoint:
                self.optG.load_state_dict(checkpoint['optimizer_state_dict'])
                self.iter = checkpoint.get('selfiter', 0)
                logger.info(f'Loaded optimizer state, current iter: {self.iter}')
            
            # Load EMA model if available and using EMA
            if _use_ema_scheduler and 'ema_model_state_dict' in checkpoint:
                self.netG_EMA.load_state_dict(checkpoint['ema_model_state_dict'])
                logger.info('Loaded EMA model state')
            
            # Set iteration counter
            if 'iter_step' in checkpoint:
                self.iter = checkpoint['iter_step']
            
            logger.info(f'Successfully loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
        else:
            logger.info('No checkpoint to load or checkpoint path does not exist')


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))







