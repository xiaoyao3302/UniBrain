# UniBrain

This is the official PyTorch implementation of our paper: **[UniBrain: A Unified Model for Cross-Subject Brain Decoding](https://arxiv.org/pdf/2412.19487)**



**Abstract** 
> Brain decoding aims to reconstruct original stimuli from fMRI signals, providing insights into interpreting mental content. Current approaches rely heavily on subject-specific models due to the complex brain processing mechanisms and the variations in fMRI signals across individuals. Therefore, these methods greatly limit the generalization of models and fail to capture cross-subject commonalities. To address this, we present UniBrain, a unified brain decoding model that requires no subject-specific parameters. Our approach includes a group-based extractor to handle variable fMRI signal lengths, a mutual assistance embedder to capture cross-subject commonalities, and a bilevel feature alignment scheme for extracting subject-invariant features. We validate our UniBrain on the brain decoding benchmark, achieving comparable performance to current state-of-the-art subject-specific models with extremely fewer parameters. We also propose a generalization benchmark to encourage the community to emphasize cross-subject commonalities for more general brain decoding. 



# Getting Started

## 1. Installation

```bash
conda create -n UniBrain python=3.10
conda activate UniBrain
pip install -r requirements.txt
```

## 2. Data preparation
Please download the **[NSD dataset](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/webdataset_avg_split)** and the **[COCO captions](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)**.
After the preparation of the data, the folder should be like
```
├── NSD
    ├── nsddata
        └── experiments
            └── nsd
                └── nsd_stim_info_merged.csv
    ├── nsddata_stimuli
        └── stimuli
            └── nsd
                └── annotations
                    ├── captions_train2017.json
                    ├── ...
                    └── person_keypoints_val2017.json
    └── webdataset_avg_split
        ├── test
            ├── subj01
            ├── ...
            └── subj07
        ├── train
            └── ...
        ├── val
            └── ...
        ├── ...
        └── webdataset_avg_split_metadata_subj07.json
```

## 3. Model preparation
Please download the pretrained **[CLIP model](https://github.com/openai/CLIP)**.

Please download the pretrained **[Versatile Diffusion model](https://huggingface.co/shi-labs/versatile-diffusion)**.

Please download the pretrained **[VAE model](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/mindeye_models)**. Note that the pretrained VAE model is only used when reconstructing the stimuli with subject-specific parameters, using img-to-img like MindEye.

Please specify the path for each pretrained model.

## 4. Usage

### I. Training

#### a. UniBrain
```
bash ./scripts/train_uni.sh
```

#### b. UniBrain with subject-specific parameters
```
bash ./scripts/train_sep.sh
```

#### c. UniBrain under the OOD settings
```
bash ./scripts/train_OOD.sh
```

### II. Inference

#### a. UniBrain without subject-specific parameters, not using img-to-img
```
bash ./scripts/inference_no_img2img.sh
```

#### b. UniBrain with subject-specific parameters, not using img-to-img
```
bash ./scripts/inference_no_img2img_sep.sh
```

#### c. UniBrain with subject-specific parameters, using img-to-img
```
bash ./scripts/inference_img2img_sep.sh
```

### III. Checkpoints

All of the **checkpoints**, including the checkpoints for **each ablation** can be found **[here](https://drive.google.com/drive/folders/1AvI84zY8PzBHbDl48TNX_z8QLhBHJ5r8?usp=sharing)**. Please refer to note.md in the cloud folder for more details about choosing your required checkpoints.


### IV. Notes

If you want to visualize the reconstructed stimuli from different subjects, remember to change the subject index.

To run with different settings, please modify the settings in the bash file.

Note that all of our experiments are tested on 4 40G A100 GPUs.




## Citation

If you find this project useful, please consider citing our paper.
```
@article{wang2024unibrain,
  title={UniBrain: A Unified Model for Cross-Subject Brain Decoding},
  author={Wang, Zicheng and Zhao, Zhen and Zhou, Luping and Nachev, Parashkev},
  journal={arXiv preprint arXiv:2412.19487},
  year={2024}
}
```


## Acknowledgement

We thank [NSD](https://github.com/cvnlab/nsddatapaper/), [MindEye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD), [MindBridge](https://github.com/littlepure2333/MindBridge), [UMBRAE](https://github.com/weihaox/UMBRAE) and other relevant works for their amazing open-sourced projects!
