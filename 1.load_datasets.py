import csv
import os
import numpy as np
import torch
import tifffile as tiff
from torchvision import transforms
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self, csv_path):
        self.csv_file = csv_path
        with open(self.csv_file, 'r') as file:
            self.data = list(csv.reader(file))

        # ✅ 使用绝对路径，请修改为你自己的图片路径
        self.image_root = r"C:\\Users\\LBH\\Desktop\\PNI实验 Part II\\SYMH test\\datasets"

        # 通道统计
        self.stats = {"gray": 0, "rgb": 0, "rgba": 0, "unknown": 0}

    def preprocess_image(self, image_path):
        # ✅ 拼接绝对路径
        full_path = os.path.join(self.image_root, image_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"❌ 图像文件不存在：{full_path}")

        image = tiff.imread(full_path)
        image = image.astype(np.float32) / 65535.0

        # ✅ 统一不同通道格式
        if image.ndim == 2:
            # 灰度图
            image = np.stack([image]*3, axis=0)
            self.stats["gray"] += 1

        elif image.ndim == 3:
            if image.shape[2] == 3:
                # RGB 图像
                image = np.transpose(image, (2, 0, 1))
                self.stats["rgb"] += 1
            elif image.shape[2] == 4:
                # RGBA 图像，取前三通道
                image = image[:, :, :3]
                image = np.transpose(image, (2, 0, 1))
                self.stats["rgba"] += 1
            else:
                self.stats["unknown"] += 1
                raise ValueError(f"Unsupported image shape: {image.shape} for {image_path}")
        else:
            self.stats["unknown"] += 1
            raise ValueError(f"Unexpected image shape {image.shape} for {image_path}")

        # 转换为 tensor
        image = torch.from_numpy(image)

        # ✅ 与原始结构保持一致的 transform
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),  
            transforms.RandomRotation(degrees=15),  
            transforms.ColorJitter(brightness=0.2),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 
        ])
        image = transform(image)

        return image
    
    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = self.preprocess_image(image_path)
        return image, int(label), image_path

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # ✅ 修改为你的 CSV 的绝对路径
    csv_file_path = r"C:\\Users\\LBH\\Desktop\\PNI实验 Part II\\SYMH test\\dataset.csv"
    dataset = DatasetLoader(csv_file_path)

    print(f"📦 共加载 {len(dataset)} 张图像 ✅")

    # 遍历以统计真实通道数分布
    for i in range(len(dataset)):
        _ = dataset[i]

    print("\n📊 图像通道类型统计结果：")
    for k, v in dataset.stats.items():
        print(f"{k}: {v}")

    # 打印第一个样本验证
    image, label, name = dataset[0]
    print(f"\n🖼️ 示例：{name} | 标签={label} | 形状={image.shape}")
