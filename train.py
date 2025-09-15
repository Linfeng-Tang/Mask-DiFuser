import os
import datetime
from typing import Dict, List
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from Diffusion.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.model import UNet
from Scheduler import GradualWarmupScheduler
from loss import Myloss
import numpy as np
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM    
import time
import argparse
from get_data import MaskData
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    torchvision.utils.save_image(img, file_directory)

def input_T(X):
    return 2 * X - 1.0

def output_T(X):
    min_val = torch.min(X)
    max_val = torch.max(X)
    X = (X - min_val) / (max_val - min_val)
    return X

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_color_map(im):
    return im / (rgb2gray(im)[..., np.newaxis] + 1e-6) * 100

def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calculate_ssim(img1, img2):
    score, _ = SSIM(img1, img2, full=True)
    return score

def train(args: Dict):
    device = args.device
    
    DATASET = MaskData(args)
    train_loader, _ = DATASET.get_loader()

    net_model = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn, num_res_blocks=args.num_res_blocks, dropout=args.dropout)

    if args.pretrained_path is not None:
        ckpt = torch.load(os.path.join(args.pretrained_path), map_location=device)
        net_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    
    net_model.to(device)
    if dist.is_available() and dist.is_initialized():
        net_model = DistributedDataParallel(net_model, find_unused_parameters=True)
    else:
        net_model = DataParallel(net_model)
    
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier, warm_epoch=1000, after_scheduler=cosineScheduler)
    
    trainer = GaussianDiffusionTrainer(net_model, args.beta_1, args.beta_T, args.T, args=args).to(device)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_savedir = os.path.join(args.output_path, 'logs', current_time)
    os.makedirs(log_savedir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_savedir)

    ckpt_savedir = os.path.join(args.output_path, 'ckpt')
    os.makedirs(ckpt_savedir, exist_ok=True)

    save_txt = os.path.join(args.output_path, 'res.txt')
    
    num = 0
    for e in range(1, args.epoch + 1):
        if is_main_process():
            tqdmDataLoader = tqdm(train_loader, dynamic_ncols=True)
        else:
            tqdmDataLoader = train_loader
        for images, labels in tqdmDataLoader:
            condA = input_T(images[0].to(device))
            condB = input_T(images[1].to(device))
            gt_imgs = input_T(images[2].to(device))
            data_concate = torch.cat([condA, condB], dim=1)
            optimizer.zero_grad()
            [loss, mse_loss, col_loss, img_mse_loss, exp_loss, ssim_loss, vgg_loss] = trainer(gt_imgs, data_concate, e)
            loss = loss.mean()
            mse_loss = mse_loss.mean()
            img_mse_loss = img_mse_loss.mean()
            ssim_loss = ssim_loss.mean()
            vgg_loss = vgg_loss.mean()
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optimizer.step()
            if is_main_process():
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "mse_loss":mse_loss.item(),
                    "img_mse_loss":img_mse_loss.item(),
                    "exp_loss":exp_loss.item(),
                    "col_loss":col_loss.item(),
                    'ssim_loss':ssim_loss.item(),
                    'vgg_loss':vgg_loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    "iter":num + 1
                })
                loss_num = loss.item()
                mse_num = mse_loss.item()
                img_mse_num = img_mse_loss.item()
                exp_num = exp_loss.item()
                col_num = col_loss.item()
                ssim_num = ssim_loss.item()
                vgg_num = vgg_loss.item()
                writer.add_scalars(
                    'loss', 
                    {
                        "loss_total":loss_num,
                        "mse_loss":mse_num,
                        "img_mse_loss":img_mse_num,
                        "exp_loss":exp_num,
                        'ssim_loss':ssim_num,
                        "col_loss":col_num,
                        "vgg_loss":vgg_num,
                    }, num
                )
                num += 1

        warmUpScheduler.step()
        torch.cuda.synchronize()

        #save ckpt and evaluate on test dataset
        if e % args.fluency == 0:
            if is_main_process():
                torch.save(net_model.module.state_dict(), os.path.join(ckpt_savedir, f"ckpt_{e}.pt"))
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if e % args.fluency == 0:
            avg_psnr, avg_ssim = Test(args, e)
            write_data = 'epoch: {}  psnr: {:.4f} ssim: {:.4f}\n'.format(e, avg_psnr, avg_ssim)
            f = open(save_txt, 'a+')
            f.write(write_data)
            f.close()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()


def Test(args: Dict, epoch):
    # load model and evaluate
    device = args.device
    DATASET = MaskData(args)
    _, val_dataloder = DATASET.get_loader()

    model = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn, num_res_blocks=args.num_res_blocks, dropout=0.)
    ckpt_path = os.path.join(args.output_path, "ckpt", f"ckpt_{epoch}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    if is_main_process():
        print("model load weight done.") 
    save_dir = os.path.join(args.output_path, "result", f"epoch{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    save_txt_name = os.path.join(save_dir, 'res.txt')
    f = open(save_txt_name, 'w+')
    f.close()

    psnr_list = []
    ssim_list = []
    
    model.eval()
    sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T).to(device)
    with torch.no_grad():
        if is_main_process():
            tqdmDataLoader = tqdm(val_dataloder, dynamic_ncols=True)
        else:
            tqdmDataLoader = val_dataloder
        for images, labels in tqdmDataLoader:
            condA = input_T(images[0].to(device))
            condB = input_T(images[1].to(device))
            gt_images = input_T(images[2].to(device))
            data_concate = torch.cat([condA, condB], dim=1)

            time_start = time.time()
            
            sampledImgs, _ = sampler(data_concate, ddim=True, ddim_step=args.ddim_step, ddim_eta=1.0, seed=args.seed)
            time_end = time.time()

            sampledImgs = output_T(sampledImgs)                   
            for j in range(sampledImgs.shape[0]):
                res_Imgs = sampledImgs.detach().cpu().numpy()[j].transpose(1, 2, 0)[:,:,::-1] 
                gt_imgs = images[2].detach().cpu().numpy()[j].transpose(1, 2, 0)[:,:,::-1]

                # compute psnr
                psnr = PSNR(res_Imgs, gt_imgs)
                res_gray = rgb2gray(res_Imgs)
                gt_gray = rgb2gray(gt_imgs)
                # compute ssim
                ssim_score = SSIM(res_gray, gt_gray, multichannel=True, data_range=1)
                res_Imgs = (res_Imgs * 255)
                
                psnr_list.append(psnr)
                ssim_list.append(ssim_score)
                save_path = os.path.join(save_dir, labels[j])
                cv2.imwrite(save_path, res_Imgs)
                if is_main_process():
                    tqdmDataLoader.set_description('{} | cost time: {:.3f}s | psnr: {:.3f} | ssim: {:.3f}'.format(labels[j], time_end - time_start, psnr, ssim_score))
        
        if dist.is_available() and dist.is_initialized():
            all_psnr_list = [None] * dist.get_world_size()
            all_ssim_list = [None] * dist.get_world_size()
            dist.all_gather_object(all_psnr_list, psnr_list)
            dist.all_gather_object(all_ssim_list, ssim_list)
            psnr_list = [psnr for sublist in all_psnr_list for psnr in sublist]
            ssim_list = [ssim for sublist in all_ssim_list for ssim in sublist]
            
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)

            if is_main_process():
                print(f'Average PSNR: {avg_psnr:.3f}, Average SSIM: {avg_ssim:.3f}')
                with open(save_txt_name, 'w') as f:
                    f.write('\nPSNR List:')
                    f.write(str(psnr_list))
                    f.write('\nSSIM List:')
                    f.write(str(ssim_list))
                    f.write('\nAverage PSNR:')
                    f.write(str(avg_psnr))
                    f.write('\nAverage SSIM:')
                    f.write(str(avg_ssim))
        else:
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            print(f'Average PSNR: {avg_psnr:.3f}, Average SSIM: {avg_ssim:.3f}')

            with open(save_txt_name, 'w') as f:
                f.write('\nPSNR List:')
                f.write(str(psnr_list))
                f.write('\nSSIM List:')
                f.write(str(ssim_list))
                f.write('\nAverage PSNR:')
                f.write(str(avg_psnr))
                f.write('\nAverage SSIM:')
                f.write(str(avg_ssim))
        torch.cuda.empty_cache()
        return avg_psnr, avg_ssim


if __name__== "__main__" :

    parser = argparse.ArgumentParser()
    modelArgs = {
        "epoch": 3500,
        "batch_size": 8,
        "val_batch": 1,
        "num_workers":8, 
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.1, 
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1.,
        "ddim":True,
        "ddim_step":10,
        "fluency": 500,
        "change_epoch": 200,
    }
    def parse_tuple(s):
        try:
            values = s.strip("()").split(",")
            return tuple(map(int, values))
        except:
            raise argparse.ArgumentTypeError("Tuple must be in the format (x y)")

    parser.add_argument('--pretrained_path', type=str, default=None)  
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output results")
    parser.add_argument("--patch_row_col", type=parse_tuple, default=(8, 8))
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--gpu_ids", default="0,1,2", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    for key, value in modelArgs.items():
        setattr(args, key, value)
        
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    print(f"local_rank: {local_rank}")
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if local_rank == -1:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    else:
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda", local_rank)
    args.device = device
    
    print("Start train!")
    train(args)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()