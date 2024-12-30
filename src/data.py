import os
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import utils
import kornia
from kornia.augmentation.container import AugmentationSequential


img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224,224), (0.8,1), p=0.3),
    kornia.augmentation.Resize((224, 224)),
    kornia.augmentation.RandomBrightness(brightness=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomContrast(contrast=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
    kornia.augmentation.RandomSaturation((0.8,1.2), p=0.2),
    kornia.augmentation.RandomHue((-0.1,0.1), p=0.2),
    kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
    kornia.augmentation.RandomGrayscale(p=0.2),
    data_keys=["input"],
)

class NSDDataset(Dataset):
    def __init__(self, root_dir, extensions=None, num_keyvoxel=512, num_neighbor=36, length=None, perform_group=True, use_fMRI_norm=False):
        self.root_dir = root_dir
        self.perform_group = perform_group
        self.extensions = extensions if extensions else []
        self.num_keyvoxel = num_keyvoxel
        self.num_neighbor = num_neighbor
        self.samples = self._load_samples()
        self.samples_keys = sorted(self.samples.keys())
        self.length = length
        self.use_fMRI_norm = use_fMRI_norm
        if length is not None:
            if length > len(self.samples_keys):
                pass # enlarge the dataset
            elif length > 0:
                self.samples_keys = self.samples_keys[:length]
            elif length < 0:
                self.samples_keys = self.samples_keys[length:]
            elif length == 0:
                raise ValueError("length must be a non-zero value!")
        else:
            self.length = len(self.samples_keys)

    def _load_samples(self):
        files = os.listdir(self.root_dir)
        samples = {}
        for file in files:
            file_path = os.path.join(self.root_dir, file)
            sample_id, ext = file.split(".",maxsplit=1)
            if ext in self.extensions:
                if sample_id in samples.keys():
                    samples[sample_id][ext] = file_path
                else:
                    samples[sample_id]={"subj": file_path}
                    samples[sample_id][ext] = file_path
            # print(samples)
        return samples
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image
    
    def _load_npy(self, npy_path):
        array = np.load(npy_path)
        array = torch.from_numpy(array)
        return array
    
    def vox_process(self, x):
        x = group_voxels(x, self.num_keyvoxel, self.num_neighbor)
        return x

    def norm_fMRI(self, x):
        mean, std = 0.0028, 0.9925              # refer to count_data.py, obtained from training data
        x = (x - mean) / std
    
    def subj_process(self, key):
        id = int(key.split("/")[-2].split("subj")[-1])
        return id
    
    def aug_process(self, brain3d):
        return brain3d

    def __len__(self):
        # return len(self.samples_keys)
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.samples_keys)
        sample_key = self.samples_keys[idx]
        sample = self.samples[sample_key]
        items = []
        for ext in self.extensions:
            if ext == "jpg":
                items.append(self._load_image(sample[ext]))

            elif ext == "nsdgeneral.npy":
                voxel = self._load_npy(sample[ext])
                if self.use_fMRI_norm:
                    voxel = self.norm_fMRI(voxel)
                if self.perform_group:
                    items.append(self.vox_process(voxel).float())
                else:
                    items.append(voxel)
                
            elif ext == "coco73k.npy":
                items.append(self._load_npy(sample[ext]))

            elif ext == "subj":
                items.append(self.subj_process(sample[ext]))

            elif ext == "wholebrain_3d.npy":
                brain3d = self._load_npy(sample[ext])
                items.append(self.aug_process(brain3d, ))

        return items

def group_voxels(voxels, num_keyvoxel, num_neighbor):
    num_scan, length = voxels.shape     # num_scan = 3, length varies

    # Step 1: index from num_neighbor // 2 (4) to length - num_neighbor // 2
    key_indices = torch.linspace(num_neighbor // 2, length - num_neighbor // 2, num_keyvoxel).long()
    key_indices = torch.clamp(key_indices, num_neighbor // 2, length - num_neighbor // 2)
    key_fMRI_signal = voxels[:, key_indices]
    
    # Step 2: extract index of neighbors
    half_neighbor = num_neighbor // 2
    neighbor_indices = torch.arange(-half_neighbor, half_neighbor)

    # extract neighors of each keyvoxel
    all_indices = key_indices.unsqueeze(1) + neighbor_indices.unsqueeze(0)
    all_indices = all_indices.clamp(0, length - 1)              # num_keyvoxel * num_neighbor

    # Step 3: extract gruped signals
    grouped_fMRI_signal = voxels[:, all_indices]                # (num_scan, num_keyvoxel, num_neighbor)
    return grouped_fMRI_signal


def get_dataloader(
        root_dir,
        batch_size,
        num_workers=1,
        seed=42,
        is_shuffle=True,
        extensions=['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"],
        num_keyvoxel=512,
        num_neighbor=36,
        length=None,
        perform_group=True,
        use_fMRI_norm=False,
    ):
    utils.seed_everything(seed)
    dataset = NSDDataset(root_dir=root_dir, extensions=extensions, num_keyvoxel=num_keyvoxel, num_neighbor=num_neighbor, length=length, perform_group=perform_group)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=is_shuffle)

    return dataloader

def get_dls(subject, data_path, batch_size, val_batch_size, num_workers, num_keyvoxel, num_neighbor, length, seed, use_fMRI_norm=False):
    train_path = "{}/webdataset_avg_split/train/subj0{}".format(data_path, subject)
    val_path = "{}/webdataset_avg_split/val/subj0{}".format(data_path, subject)
    extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"]

    train_dl = get_dataloader(
        train_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=extensions,
        num_keyvoxel=num_keyvoxel,
        num_neighbor=num_neighbor,
        is_shuffle=True,
        length=length,
        use_fMRI_norm=use_fMRI_norm,
    )

    val_dl = get_dataloader(
        val_path,
        batch_size=val_batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=extensions,
        num_keyvoxel=num_keyvoxel,
        num_neighbor=num_neighbor,
        is_shuffle=False,
        use_fMRI_norm=use_fMRI_norm,
    )

    num_train=len(train_dl.dataset)
    num_val=len(val_dl.dataset)
    print(train_path,"\n",val_path)
    print("number of train data:", num_train)
    print("batch_size", batch_size)
    print("number of val data:", num_val)
    print("val_batch_size", val_batch_size)

    return train_dl, val_dl
