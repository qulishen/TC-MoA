# Introduction 🌟
This is the unofficial code for the paper **Task-Customized Mixture of Adapters for General Image Fusion**, which resolves the issue of black images during inference. [Issue](https://github.com/YangSun22/TC-MoA/issues/2)

[中文版本 (Chinese Version)](README-cn.md)

# Preparation 🛠️

## Pretrained Checkpoints:
From MAE ([GitHub - facebookresearch/mae: PyTorch implementation of MAE](https://github.com/facebookresearch/mae))

```bash
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth
```

## Dataset:
- **BaiduYun**: [Link](https://pan.baidu.com/s/1R2R58PjJuMaS2P4uwlTBqg?pwd=hyqv) Code: `hyqv`
- **Google Drive**: [Link](https://drive.google.com/drive/folders/1yFHwmebySDmgLwImQRT-XdEVZ6HjO1Vc?usp=drive_link)

## TC-MoA Checkpoints:
- **BaiduYun**: [Link](https://pan.baidu.com/s/19u8OgMQbQqfvNyaDkmRlNQ?pwd=iqzf) Code: `iqzf`
- **Google Drive**: [Link](https://drive.google.com/file/d/1S23P6Sw-UQMaPY16XxOnegojjEexm3ER/view?usp=drive_link)

# Training 🚀

```bash
CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 NCCL_P2P_LEVEL=NVL nohup python -m torch.distributed.launch \
    --nproc_per_node 3 --master_port 22222 \
    main_train.py --config_path ./config/base.yaml \
     > test.log 2>&1 & 
```

# Testing 🧪

```bash
CUDA_VISIBLE_DEVICES=0 python main_predict.py --config_path ./config/predict.yaml
```

The folder structure for the test dataset should follow these formats:

```python
for dataset_name in self.EvalDataSet.keys():
    ddir = self.EvalDataSet[dataset_name]

    if dataset_name in ["LLVIP", "LLVIP_Test"]:       
        rgb_dir = os.path.join(ddir, "visible", "test")     # RGB
        t_dir = os.path.join(ddir, "infrared", "test")      # Infrared
    elif dataset_name in ["MandP", "M3FD"]:
        rgb_dir = os.path.join(ddir, "vi")     # RGB
        t_dir = os.path.join(ddir, "ir")      # Infrared
    elif dataset_name in ["MEFB", "MEF", "MFF"]:
        rgb_dir = os.path.join(ddir, "input")             # Overexposed
        t_dir = os.path.join(ddir, "input")               # Underexposed
    elif dataset_name == "Lytro":
        rgb_dir = os.path.join(ddir, "BB")     # Far focus
        t_dir = os.path.join(ddir, "AA")       # Near focus
    elif dataset_name == "TNO":
        rgb_dir = os.path.join(ddir, "vi")     # RGB
        t_dir = os.path.join(ddir, "ir")      # Infrared
    elif dataset_name == "SCIE_test":
        rgb_dir = os.path.join(ddir, "oe")     # RGB
        t_dir = os.path.join(ddir, "ue")
    else:
        print("Dataset name error!", dataset_name)
```

# Citation 📜

If you use this code, please cite:

```
@InProceedings{Zhu_2024_CVPR,
    author    = {Zhu, Pengfei and Sun, Yang and Cao, Bing and Hu, Qinghua},
    title     = {Task-Customized Mixture of Adapters for General Image Fusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {7099-7108}
}
```

---
