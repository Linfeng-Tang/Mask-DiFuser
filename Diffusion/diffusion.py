from typing import Dict
from tensorboardX import SummaryWriter
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import Myloss
from kornia.losses import ssim_loss
import numpy as np
import lpips
import time
from thop import profile, clever_format


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)) 


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, Pre_train=None, args=None):
        super().__init__()

        self.model = model  
        self.T = T
        self.change_epoch = args.change_epoch
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.L_color = None
        self.num = 0
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')


    def forward(self, gt_images, data_concate, epoch):
        t = torch.randint(self.T, size=(gt_images.shape[0],), device=gt_images.device)
        noise = torch.randn_like(gt_images)
        y_t = (
            extract(self.sqrt_alphas_bar, t, gt_images.shape) * gt_images +
            extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise
        )

        input = torch.cat([data_concate, y_t], dim=1).float()
        noise_pred = self.model(input, t)

        # loss
        loss = 0
        mse_loss_weight = 10
        mse_loss = F.mse_loss(noise_pred, noise, reduction='none') * 10

        y_0_pred = 1 / extract(self.sqrt_alphas_bar, t, gt_images.shape) * (
                    y_t - extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise_pred).float()
        y_0_pred = torch.clip(y_0_pred, -1, 1)

        col_loss = 0
        col_loss_weight = 10
        if epoch <= self.change_epoch:
            col_loss_weight = 0
        col_loss = Myloss.color_loss(y_0_pred, gt_images) * 10

        img_mse_loss = 0
        img_mse_loss_weight = 0.5
        if epoch <= self.change_epoch:
            img_mse_loss_weight = 0
        img_mse_loss = F.mse_loss(y_0_pred, gt_images, reduction='none') * 10

        exposure_loss = 0
        exposure_loss_weight = 0
        if epoch <= self.change_epoch:
            exposure_loss_weight = 0
        exposure_loss = Myloss.light_loss(y_0_pred, gt_images) * 10

        simLoss = 0
        ssimLoss_weight = 0.5
        if epoch <= self.change_epoch:
            ssimLoss_weight = 0
        ssimLoss = ssim_loss(y_0_pred, gt_images, window_size=11) * 10
        
        vgg_loss = 0
        vgg_loss_wight = 1
        if epoch <= self.change_epoch:
            vgg_loss_wight = 0
        vgg_loss = self.loss_fn_vgg(gt_images, y_0_pred) * 10

        loss = (mse_loss * mse_loss_weight) + (col_loss * col_loss_weight) + (img_mse_loss * img_mse_loss_weight) + (ssimLoss * ssimLoss_weight) + (vgg_loss * vgg_loss_wight) 
        return [loss, mse_loss, col_loss, img_mse_loss, exposure_loss, ssimLoss, vgg_loss]


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,):
        super().__init__()

        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.sqrt_alphas_bar=alphas_bar
        self.sqrt_one_minus_alphas_bar=torch.sqrt(1. - alphas_bar)
        self.alphas_bar=alphas_bar
        self.one_minus_alphas_bar=(1. - alphas_bar)
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.grad_coeff = None
        self.every_mins = 0


    def predict_xt_prev_mean_from_eps(self,  t, eps,y_t):
        assert y_t.shape == eps.shape
        return (
            extract(self.coeff1, t, y_t.shape) * y_t -
            extract(self.coeff2, t, y_t.shape) * eps
        )

    def p_mean_variance(self, input, t,y_t,brightness_level):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, input.shape)
        eps = self.model(input, t,brightness_level)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(t, eps,y_t)
        return xt_prev_mean, var


    def forward(self, data_concate, ddim=True, ddim_step=None, ddim_eta=1., seed=None, type=None):
        torch.manual_seed(seed)
        total_time = 0.0
        # ddpm
        if not ddim:
            device = data_concate.device
            noise = torch.randn_like(data_concate[:, :3, :, :]).to(device)
            y_t = noise
            for time_step in reversed(range(self.T)):
                t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step  
                input = torch.cat([data_concate, y_t], dim=1).float()
                mean, var = self.p_mean_variance(input, t, y_t)
                if time_step > 0:
                    noise = torch.randn_like(y_t)
                else:
                    noise = 0
                y_t = mean + torch.sqrt(var) * noise
            y_0 = y_t
            return y_0
        # ddim
        else:
            device = data_concate.device
            noise = torch.randn_like(data_concate[:, :3, :, :]).to(device)
            y_t = noise
            
            start_time = time.time() 
            fa = self.model.encoder(data_concate[:, :3, :, :])
            fb = self.model.encoder(data_concate[:, 3:6, :, :])            
            # flops_encoder, params_encoder = profile(self.model.encoder, inputs=(data_concate[:, :3, :, :], ))
            f_f = self.model.content_encoder(data_concate[:, :3, :, :], data_concate[:, 3:6, :, :])
            # flops_content, params_content = profile(self.model.content_encoder, inputs=(data_concate[:, :3, :, :], data_concate[:, 3:6, :, :], ))
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            
            step = 1000 / ddim_step
            step = int(step)
            seq = range(0, 1000, step)
            seq_next = [-1] + list(seq[:-1])
            
            for num, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(y_t.shape[0]) * i).to(device).long()
                next_t = (torch.ones(y_t.shape[0]) * j).to(device).long()
                at = extract(self.alphas_bar.to(device), (t + 1).long(), y_t.shape)
                at_next = extract(self.alphas_bar.to(device), (next_t + 1).long(), y_t.shape)
                input = torch.cat([data_concate, y_t], dim=1).float()
                start_time = time.time() 
                eps = self.model(input, t, fa, fb, f_f)
                # if num == 0:
                #     flops_UNet, params_UNet = profile(self.model, inputs=(input, t, fa, fb, f_f, ))
                #     flops = flops_encoder * 2 + flops_content + flops_UNet
                #     params = params_encoder + params_content + params_UNet                    
                #     print("Params: {:.4f} M | Flops : {:.4f} G".format(params / 1e6, flops / 1e9))
                
                y0_pred = (y_t - eps * (1 - at).sqrt()) / at.sqrt()
                if type == 'MFF':
                    y0_pred = torch.clip(y0_pred, -1.5, 1)    
                elif type == 'Med':
                    y0_pred = torch.clip(y0_pred, -1., 3)
                else:
                    y0_pred = torch.clip(y0_pred, -1, 1)   
                c1 = ddim_eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                y_t = at_next.sqrt() * y0_pred + c1 * torch.randn_like(data_concate[:, :3, :, :]) + c2 * eps
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
            y_0 = y_t

            return y_0, total_time
