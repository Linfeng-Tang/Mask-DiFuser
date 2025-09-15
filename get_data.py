import os
import torch
import torchvision
from PIL import Image, ImageFilter
import re
import random
from natsort import natsorted
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import cv2
import io


def random_choice(image):
    def random_odd_number(low=5, high=35):
        number = random.randint(low, high)
        return number if number % 2 != 0 else number + 1

    def add_blur(image, ksize=5):
        ksize = random_odd_number(3, 7)
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def add_noise(image, noise_type='gaussian'):
        if noise_type == 'gaussian':
            mean = 0
            std = random_odd_number(low=5, high=35)
            gauss = np.random.normal(mean, std, image.shape).astype('float32')
            noisy_image = image + gauss
            noisy_image = np.clip(noisy_image, 0, 255)
            return noisy_image

    def add_rain_snow(image, effect='rain'):
        if effect == 'rain':
            noise = np.zeros_like(image)
            num_drops = 1000
            for i in range(num_drops):
                x = np.random.randint(0, image.shape[1])
                y = np.random.randint(0, image.shape[0])
                noise[y:y+5, x:x+2] = 255
            noise = cv2.blur(noise, (5, 5))
            image = cv2.addWeighted(image, 0.8, noise, 0.2, 0)
            return image
        elif effect == 'snow':
            noise = np.zeros_like(image)
            num_flakes = 1000
            for i in range(num_flakes):
                x = np.random.randint(0, image.shape[1])
                y = np.random.randint(0, image.shape[0])
                noise[y:y+2, x:x+2] = 255
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
            image = cv2.addWeighted(image, 0.7, noise, 0.3, 0)
            return image

    def degrade_frequency_domain(image, np_random=5):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        phase_spectrum = np.angle(fshift)
        magnitude_spectrum = cv2.GaussianBlur(magnitude_spectrum, (0, 0), 0.5)
        for _ in range(np_random):
            idx1, idx2 = np.random.randint(0, phase_spectrum.shape[0], 2), np.random.randint(0, phase_spectrum.shape[1], 2)
            phase_spectrum[idx1[0], idx1[1]], phase_spectrum[idx2[0], idx2[1]] = phase_spectrum[idx2[0], idx2[1]], phase_spectrum[idx1[0], idx1[1]]

        fshift_new = magnitude_spectrum * np.exp(1j * phase_spectrum)
        f_ishift = np.fft.ifftshift(fshift_new)
        image_back = np.fft.ifft2(f_ishift)
        image_back = np.abs(image_back)

        return np.clip(image_back, 0, 255).astype('uint8')

    def gamma_transform(image, gamma_range=(0.3, 3.0)):
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    operations = [
        lambda img: add_blur(img),
        lambda img: add_noise(img, noise_type='gaussian'),
        lambda img: add_rain_snow(img, effect='snow'),
        lambda img: degrade_frequency_domain(img, np_random=5),
        lambda img: gamma_transform(img, gamma_range=(0.3, 3.0)),
    ]
    operation = random.choice(operations)
    return operation(image)


def random_mask(pil_image, num_row_col=(8, 8), set_seed=False, seed=3407, ratio=1.0, blur_ratio=0.1):
    def crop_resize(image):
        width, height = image.size
        crop_size = 256
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    # pil_image = crop_resize(pil_image)
    image = np.array(pil_image)
    rows, cols, channels = image.shape

    patch_size_row = rows // num_row_col[0]
    patch_size_col = cols // num_row_col[1]

    patches_indices = [(i, j) for i in range(num_row_col[0]) for j in range(num_row_col[1])]

    selected_indices = random.sample(patches_indices, k=int(len(patches_indices) * ratio))
    first_mask_indices = random.sample(selected_indices, k=int(len(selected_indices) * 0.5))
    first_pixel_ids = random.sample(first_mask_indices, k=int(len(first_mask_indices) * 0.9))
    first_deg_ids = list(set(first_mask_indices) - set(first_pixel_ids))  

    second_mask_indices = list(set(selected_indices) - set(first_mask_indices))    
    second_pixel_ids = random.sample(second_mask_indices, k=int(len(second_mask_indices) * 0.9))
    second_deg_ids = list(set(second_mask_indices) - set(second_pixel_ids))   

    deg_ids1 = random.sample(second_mask_indices, k=int(len(second_pixel_ids) * blur_ratio))
    deg_ids2 = random.sample(first_mask_indices, k=int(len(first_pixel_ids) * blur_ratio))
    
    image_1 = image.copy()
    image_2 = image.copy()

    def apply_blur(patch, radius):
        pil_patch = Image.fromarray(patch)
        blurred_patch = pil_patch.filter(ImageFilter.GaussianBlur(radius))
        return np.array(blurred_patch)

    deg_choice_img = random_choice(image)
    def process_blocks(image, pixel_ids, p_ids, deg_ids, pixel):
        for idx in pixel_ids:
            i, j = idx
            image[
                i * patch_size_row : (i + 1) * patch_size_row,
                j * patch_size_col : (j + 1) * patch_size_col,
                :,
            ] = pixel 

        for idx in p_ids:
            i, j = idx
            image[
                i * patch_size_row : (i + 1) * patch_size_row,
                j * patch_size_col : (j + 1) * patch_size_col,
                :,
            ] = deg_choice_img[
                i * patch_size_row : (i + 1) * patch_size_row,
                j * patch_size_col : (j + 1) * patch_size_col,
                :,
            ]
            
        for idx in deg_ids:
            i, j = idx
            image[
                i * patch_size_row : (i + 1) * patch_size_row,
                j * patch_size_col : (j + 1) * patch_size_col,
                :,
            ] = deg_choice_img[
                i * patch_size_row : (i + 1) * patch_size_row,
                j * patch_size_col : (j + 1) * patch_size_col,
                :,
            ]
        return image
    pixel_level = np.random.randint(0, 256)
    image_1 = process_blocks(image_1, first_pixel_ids, first_deg_ids, deg_ids1, pixel=pixel_level)
    image_2 = process_blocks(image_2, second_pixel_ids, second_deg_ids, deg_ids2, pixel=pixel_level)

    pil_image_1 = Image.fromarray(image_1.astype(np.uint8))
    pil_image_2 = Image.fromarray(image_2.astype(np.uint8))
    return pil_image_1, pil_image_2


class MaskData:
    def __init__(self, args):
        self.args = args
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loader(self):
        train_dataset = MaskTrainDataset(
            dir=os.path.join(self.args.dataset_path, "train"),
            transforms=self.transforms,
            args=self.args,
        )
        val_dataset = MaskValDataset(
            dir=os.path.join(self.args.dataset_path, "val"),
            transforms=self.transforms,
            args=self.args,
        )
        if dist.is_available() and dist.is_initialized():
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=True,
            )
            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                batch_size=self.args.val_batch,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=self.args.val_batch,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        return train_loader, val_loader


class MaskTrainDataset(Dataset):
    def __init__(self, dir, transforms, args):
        super().__init__()
        self.patch_row_col = args.patch_row_col
        self.crop_size = 128
        dir_gt = dir
        gt_names = []
        file_list = natsorted(os.listdir(dir_gt))

        for item in file_list:
            if item.endswith((".jpg", ".png", ".bmp")):
                gt_names.append(os.path.join(dir_gt, item))
        self.gt_names = gt_names
        self.transforms = transforms

    def __len__(self):
        return len(self.gt_names)

    def crop_resize(self, image):

        width, height = image.size
        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)

        right = left + self.crop_size
        bottom = top + self.crop_size
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    def __getitem__(self, idx):
        gt_name = self.gt_names[idx]
        img_id = re.split("/", gt_name)[-1]
        gt_img = Image.open(gt_name)
        gt_img = self.crop_resize(gt_img)
        deg1_img, deg2_img = random_mask(gt_img, num_row_col=self.patch_row_col)
        # deg1_img, deg2_img = random_mask_ablation(gt_img, num_row_col=self.patch_row_col)

        to_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
            ]
        )
        stack_imgs = torch.stack(
            [
                self.transforms(deg1_img),
                self.transforms(deg2_img),
                self.transforms(gt_img),
            ],
            dim=0,
        )
        stack_imgs = to_transform(stack_imgs)
        res = torch.unbind(stack_imgs, dim=0)
        for i in range(len(res)):
            if res[i].size(0) != 3:
                if res[i].size(0) == 4:
                    channels_to_keep = [0, 1, 2]
                    res[i] = torch.index_select(
                        res[i], dim=0, index=torch.tensor(channels_to_keep)
                    )
                else:
                    res[i] = res[i].expand(3, -1, -1)
        return (res, img_id)

class MaskValDataset(Dataset):

    def __init__(self, dir, transforms, args):
        super().__init__()
        self.patch_row_col = args.patch_row_col
        self.crop_size = 512
        dir_gt = dir

        gt_names = []
        file_list = natsorted(os.listdir(dir_gt))

        for item in file_list:
            if item.endswith((".jpg", ".png", ".bmp")):
                gt_names.append(os.path.join(dir_gt, item))
        self.gt_names = gt_names
        self.transforms = transforms

    def __len__(self):
        return len(self.gt_names)

    def crop_resize(self, image):

        width, height = image.size
        left = (width - self.crop_size) // 2
        top = (height - self.crop_size) // 2
        right = left + self.crop_size
        bottom = top + self.crop_size
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    def __getitem__(self, idx):
        gt_name = self.gt_names[idx]
        img_id = re.split("/", gt_name)[-1]
        gt_img = Image.open(gt_name)
        gt_img = self.crop_resize(gt_img)
        deg1_img, deg2_img = random_mask(gt_img, num_row_col=(16, 16), set_seed=True)
        # deg1_img, deg2_img = random_mask_ablation(gt_img, num_row_col=(16, 16), set_seed=True)
        res = [
            self.transforms(deg1_img),
            self.transforms(deg2_img),
            self.transforms(gt_img),
        ]
        for i in range(len(res)):
            if res[i].size(0) != 3:
                if res[i].size(0) == 4:
                    channels_to_keep = [0, 1, 2]
                    res[i] = torch.index_select(
                        res[i], dim=0, index=torch.tensor(channels_to_keep)
                    )
                else:
                    res[i] = res[i].expand(3, -1, -1)
        return (res, img_id)


class FusionData:
    def __init__(self, args):
        self.args = args
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_fusion_loader(self):
        fusion_dataset = FusionDataset(dirA=self.args.dirA, dirB=self.args.dirB, transforms=self.transforms)
        if dist.is_available() and dist.is_initialized():
            fusion_sampler = DistributedSampler(
                fusion_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=True,
            )
            fusion_loader = DataLoader(
                fusion_dataset,
                sampler=fusion_sampler,
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            fusion_loader = DataLoader(
                fusion_dataset,
                shuffle=False,
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        return fusion_loader


class FusionDataset(Dataset):
    def __init__(self, dirA, dirB, transforms):
        super().__init__()
        
        self.t_dir = dirA
        self.rgb_dir = dirB
    
        t_names, rgb_names = [], []
        file_list = natsorted(os.listdir(self.rgb_dir))
        for item in file_list:
            if item.endswith((".jpg", ".png", ".bmp")):
                t_names.append(os.path.join(self.t_dir, item))
                rgb_names.append(os.path.join(self.rgb_dir, item))

        self.t_names = t_names
        self.rgb_names = rgb_names
        self.transforms = transforms

    def __len__(self):
        return len(self.t_names)

    def __getitem__(self, idx):
        t_name = self.t_names[idx]
        rgb_name = self.rgb_names[idx]
        img_id = re.split("/", t_name)[-1]
        t_img = Image.open(t_name)
        rgb_img = Image.open(rgb_name)
        res = [self.transforms(t_img), self.transforms(rgb_img)]

        for i in range(len(res)):
            if res[i].size(0) != 3:
                if res[i].size(0) == 4:
                    channels_to_keep = [0, 1, 2]
                    res[i] = torch.index_select(
                        res[i], dim=0, index=torch.tensor(channels_to_keep)
                    )
                else:
                    res[i] = res[i].expand(3, -1, -1)
        return (res, img_id)
