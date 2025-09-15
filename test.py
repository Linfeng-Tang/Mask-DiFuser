import os
from typing import Dict, List
import cv2
import torch
from tqdm import tqdm
from Diffusion.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.model import UNet
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import time
import argparse
from get_data import FusionData
import torch.nn.functional as F
import statistics

def write_txt(path, a):
    with open(path, "a") as file:
        file.write(f"{a}\n")

def input_T(X):
    return 2 * X - 1.0

def output_T(X):
    min_val = torch.min(X)
    max_val = torch.max(X)
    X = (X - min_val) / (max_val - min_val)
    return X

def Fusion(args: Dict):
    # load model and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = FusionData(args)
    fusion_loader = DATASET.get_fusion_loader()

    model = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn, num_res_blocks=args.num_res_blocks, dropout=0.)
    ckpt_path = args.pretrained_path
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("model load weight done.")
    # save_dir = os.path.join(args.output_path, args.dataset_name)
    save_dir = os.path.join(args.output_path)
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(os.path.join(save_dir, "time"), exist_ok=True)
    model.eval()
    sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T).to(device)

    with torch.no_grad():
        with tqdm(fusion_loader, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                all_time = 0
                time_list = []
                for images, labels in tqdmDataLoader:
                    # relevant data
                    img_ir = input_T(images[0].to(device))
                    img_vi = input_T(images[1].to(device))
                    _, _, h, w = img_ir.shape
                    multiple = 2 ** len(args.channel_mult)
                    crop_height = int(multiple * np.ceil(h / multiple))
                    crop_width = int(multiple * np.ceil(w / multiple))

                    condA = F.pad(img_ir, (0, crop_width - w, 0, crop_height - h), "reflect")
                    condB = F.pad(img_vi, (0, crop_width - w, 0, crop_height - h), "reflect")
                    name = labels[0]

                    data_concate = torch.cat([condA, condB], dim=1)

                    time_start = time.time()
                    sampledImgs, cost_time = sampler(data_concate, ddim=True, ddim_step=args.ddim_step, ddim_eta=1.0, seed=args.seed, type=args.task_type)
                    all_time += cost_time
                    time_list.append(cost_time)
                    # write_txt(os.path.join(save_dir, "time", f"step{args.ddim_step}.txt"), f"image id: {name}\tcost time: {cost_time}")
                    sampledImgs = sampledImgs[:, :, :h, :w]
                    time_end = time.time()

                    sampledImgs = output_T(sampledImgs)                   

                    for j in range(sampledImgs.shape[0]):
                        res_Imgs = sampledImgs.detach().cpu().numpy()[j].transpose(1, 2, 0)[:,:,::-1] 
                        res_Imgs = (res_Imgs * 255)
                        save_path = os.path.join(save_dir, labels[j])
                        print(f"Sample over! Save to {save_path}")
                        cv2.imwrite(save_path, res_Imgs)
                        tqdmDataLoader.set_description('{} | {:.3f} s'.format(labels[j], time_end - time_start))
                avg_time = statistics.mean(time_list)
                print(" Average running time: {:.3f} s".format(avg_time))
                # write_txt(os.path.join(save_dir, "time", f"step{args.ddim_step}.txt"), f"Average cost time: {average_time}")
                


if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    modelArgs = {
        "epoch": 3500,
        "batch_size":2,
        "val_batch": 1,
        "num_workers":8, 
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.0,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1.,
        "fluency": 500,
        "change_epoch": 100,
        "ddim":True,
        "ddim_step":5,
    }
    
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--task_type', type=str, required=True, help='options: VIF, MEF, MFF, Med, Pol, Nir')
    parser.add_argument('--dirA', type=str, required=True, help='Path to dataset A')
    parser.add_argument('--dirB', type=str, required=True, help='Path to dataset B')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output results')

    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--gpu_ids", default="0", type=str)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    for key, value in modelArgs.items():
        setattr(args, key, value)
    Fusion(args)
