import torch


def fMRI_random_noise(fMRI, noise_level=0.01, noise_range=0.05):
    # apply random noise to fMRI signals
    batch_size, fMRI_length, fMRI_dim = fMRI.shape
    device = fMRI.device

    noise = torch.randn(batch_size, fMRI_length, fMRI_dim).to(device)
    noise = noise * noise_level
    noise = torch.clamp(noise, -1 * noise_range, noise_range)

    fMRI = fMRI + noise

    return fMRI


def fMRI_random_crop(fMRI, crop_ratio=0.1):
    # crop the fMRI signals, pad with 0
    batch_size, fMRI_length, fMRI_dim = fMRI.shape
    device = fMRI.device

    num_voxel = fMRI_length * fMRI_dim
    num_mask = int(num_voxel * crop_ratio)

    mask = torch.ones_like(fMRI, dtype=torch.bool).to(device)

    for bb in range(batch_size):
        mask_indices = torch.randperm(num_voxel)[:num_mask]
        mask[bb].view(-1)[mask_indices] = False

    fMRI = fMRI.masked_fill(~mask, 0)

    return fMRI


if __name__ == '__main__':
    data = torch.rand([2, 10, 4]).cuda()
    data1 = fMRI_random_noise(data)
    data2 = fMRI_random_crop(data)
    print(data)
    print('---')
    print(data1)
    print('---')
    print(data2)