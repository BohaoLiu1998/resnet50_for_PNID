
import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import tifffile
import cv2
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================== 基本设置 ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前设备: {device}")

model_path = r"C:\Users\LBH\Desktop\PNI实验 Part II\SYMH test\checkpoint\best_model.pth"
input_folder = r"C:\Users\LBH\Desktop\PNI实验 Part II\SYMH test\datasets"
output_folder = r"C:\Users\LBH\Desktop\PNI实验 Part II\SYMH test\GradCAM_results_ROI_crop"
os.makedirs(output_folder, exist_ok=True)
cam_threshold = 0.2


# ============================== 加载模型 ==============================
print("🔹 正在加载 ResNet50 模型...")
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)
model = model.to(device)
state_dict = torch.load(model_path, map_location=device)
for k in ["fc.weight", "fc.bias"]:
    if k in state_dict:
        del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ 模型加载完成。")

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
print("✅ Grad‑CAM 初始化成功。")


# ============================== 图像预处理 ==============================
def preprocess_tif(image_path):
    """读取并预处理 .tiff 图像，支持灰度/RGB/RGBA"""
    img = tifffile.imread(image_path)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=2)
    elif img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1]:
        img = img.transpose(1, 2, 0)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    h, w = img.shape[:2]
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= img.max()

    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    return img, input_tensor, h, w


# ============================== Grad‑CAM 批量生成（模式A） ==============================
image_list = sorted(glob.glob(os.path.join(input_folder, "*.tiff")))
if not image_list:
    print("❌ 没有找到任何 .tiff 图像！")
else:
    print(f"📂 检测到 {len(image_list)} 张图像，将逐一生成 Grad‑CAM 热图...")

saved_count = 0
for img_path in tqdm(image_list, desc="🎯 Generating ROI‑only Grad‑CAM", ncols=100):
    name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
    try:
        combined, input_tensor, h, w = preprocess_tif(img_path)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))
        grayscale_cam_resized[grayscale_cam_resized < cam_threshold] = 0

        rgb_img = np.clip(combined, 0, 1)
        overlay = np.uint8(255 * rgb_img)
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), cv2.COLORMAP_JET)
        cam_overlay = cv2.addWeighted(overlay, 0.5, heatmap, 0.5, 0)
        cam_rgb = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)

        # 自动检测 ROI 非黑区域
        gray_for_mask = cv2.cvtColor(np.uint8(255 * rgb_img), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray_for_mask, 5, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)

        if coords is not None:
            x, y, w_roi, h_roi = cv2.boundingRect(coords)
            cam_crop = cam_rgb[y:y+h_roi, x:x+w_roi]
        else:
            cam_crop = cam_rgb

        save_path = os.path.join(output_folder, f"{name_no_ext}_CAM_ROI.tiff")
        tifffile.imwrite(save_path, cam_crop.astype(np.uint8), photometric='rgb')
        saved_count += 1

    except Exception as e:
        print(f"⚠️ {name_no_ext} 处理失败: {e}")

print(f"\n✅ 所有图像处理完成，共保存 {saved_count}/{len(image_list)} 张 ROI 热图。")
print(f"📁 输出目录: {output_folder}")
