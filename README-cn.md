# 简介 🌟
这是论文 **Task-Customized Mixture of Adapters for General Image Fusion** 的非官方代码，解决了推理图片为黑色的问题。[问题链接](https://github.com/YangSun22/TC-MoA/issues/2)

[English Version (英文版本)](README.md)

# 准备 🛠️

## 预训练模型：
来自 MAE ([GitHub - facebookresearch/mae: PyTorch implementation of MAE](https://github.com/facebookresearch/mae))

```bash
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth
```

## 数据集：
- **百度云**: [链接](https://pan.baidu.com/s/1R2R58PjJuMaS2P4uwlTBqg?pwd=hyqv) 提取码: `hyqv`
- **Google Drive**: [链接](https://drive.google.com/drive/folders/1yFHwmebySDmgLwImQRT-XdEVZ6HjO1Vc?usp=drive_link)

## TC-MoA 模型：
- **百度云**: [链接](https://pan.baidu.com/s/19u8OgMQbQqfvNyaDkmRlNQ?pwd=iqzf) 提取码: `iqzf`
- **Google Drive**: [链接](https://drive.google.com/file/d/1S23P6Sw-UQMaPY16XxOnegojjEexm3ER/view?usp=drive_link)

# 训练 🚀

```bash
CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 NCCL_P2P_LEVEL=NVL nohup python -m torch.distributed.launch \
    --nproc_per_node 3 --master_port 22222 \
    main_train.py --config_path ./config/base.yaml \
     > test.log 2>&1 & 
```

# 测试 🧪

```bash
CUDA_VISIBLE_DEVICES=0 python main_predict.py --config_path ./config/predict.yaml
```

测试数据集的文件夹路径格式如下：

```python
for dataset_name in self.EvalDataSet.keys():
    ddir = self.EvalDataSet[dataset_name]

    if dataset_name in ["LLVIP", "LLVIP_Test"]:       
        rgb_dir = os.path.join(ddir, "visible", "test")     # RGB
        t_dir = os.path.join(ddir, "infrared", "test")      # 红外
    elif dataset_name in ["MandP", "M3FD"]:
        rgb_dir = os.path.join(ddir, "vi")     # RGB
        t_dir = os.path.join(ddir, "ir")      # 红外
    elif dataset_name in ["MEFB", "MEF", "MFF"]:
        rgb_dir = os.path.join(ddir, "input")             # 过曝
        t_dir = os.path.join(ddir, "input")               # 低曝
    elif dataset_name == "Lytro":
        rgb_dir = os.path.join(ddir, "BB")     # 远焦
        t_dir = os.path.join(ddir, "AA")       # 近焦
    elif dataset_name == "TNO":
        rgb_dir = os.path.join(ddir, "vi")     # RGB
        t_dir = os.path.join(ddir, "ir")      # 红外
    elif dataset_name == "SCIE_test":
        rgb_dir = os.path.join(ddir, "oe")     # RGB
        t_dir = os.path.join(ddir, "ue")
    else:
        print("数据集名称错误！", dataset_name)
```

# 引用 📜

如果您使用了此代码，请引用：

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
