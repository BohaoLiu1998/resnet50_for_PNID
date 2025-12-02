import os
import sys
import torch
import torchvision
from torchvision.models import ResNet50_Weights
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

# ✅ 1. 设置路径
os.chdir(r"C:\\Users\\LBH\\Desktop\\PNI实验 Part II\\SYMH 1")
sys.path.append(os.getcwd())
print("📂 当前工作目录:", os.getcwd())

# ✅ 2. 导入你自己的数据加载模块
from load_datasets2 import DatasetLoader

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# ====== 训练函数 ======
def train(model, device, train_dataloader, optimizer, criterion, epoch, num_epochs):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    all_labels = []
    all_outputs = []
    for iter, (inputs, labels, filenames) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs > 0.5
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.append(labels.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    train_accuracy = 100 * correct / total
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    auc_score = roc_auc_score(all_labels, all_outputs)

    print(f"Epoch [{epoch}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Loss: {total_loss/len(train_dataloader):.4f}, AUC: {auc_score:.4f}")
    return train_accuracy, auc_score, all_labels, all_outputs


# ====== 验证函数 ======
def test(model, device, test_dataloader, epoch, num_epochs):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_filenames = []
    with torch.no_grad():
        for iter, (inputs, labels, filenames) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)

            predicted = outputs > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_filenames.extend(filenames)

    test_accuracy = correct / total * 100
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    auc_score = roc_auc_score(all_labels, all_predictions)

    print(f"Epoch [{epoch}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%, AUC: {auc_score:.4f}")
    return test_accuracy, auc_score, all_labels, all_predictions, all_filenames


# ====== 主程序 ======
num_epochs = 200
lr = 1e-4
lambda_l2 = 1e-6
batch_size = 64
num_classes = 1

# ✅ 自动选择设备
try:
    use_mps = torch.backends.mps.is_available()
except AttributeError:
    use_mps = False

if torch.cuda.is_available():
    device = "cuda"
elif use_mps:
    device = "mps"
else:
    device = "cpu"
print(f"✅ Using device: {device}")

# ✅ 加载 CSV 数据集
csv_file = r"C:\\Users\\LBH\\Desktop\\PNI实验 Part II\\SYMH 1\\dataset.csv"
dataset = DatasetLoader(csv_file)

# ✅ 将验证集比例改成 40%
val_size = int(0.4 * len(dataset))
train_size = len(dataset) - val_size

# ✅ 固定随机种子确保划分一致性
torch.manual_seed(42)

TrainDataset, ValDataset = random_split(dataset, [train_size, val_size])

# ✅ 保存划分索引，供推理脚本复现相同训练/验证集
split_dir = os.path.join(os.getcwd(), "checkpoint")
os.makedirs(split_dir, exist_ok=True)

np.save(os.path.join(split_dir, "train_indices.npy"), TrainDataset.indices)
np.save(os.path.join(split_dir, "val_indices.npy"), ValDataset.indices)
print(f"💾 已保存划分索引文件 ({len(TrainDataset.indices)} train, {len(ValDataset.indices)} val)")

# ✅ 构建 DataLoader
TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)

# ✅ 初始化 ResNet50 模型
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, num_classes),
    nn.Sigmoid()
)
model.to(device)

# ✅ 定义损失函数、优化器、LR 调度器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# ✅ 提前停止配置
early_stopping_rounds = 50
early_stopping_counter = 0
best_auc = 0.0

print(f"📦 Start training... total epochs = {num_epochs}")
for epoch in range(1, num_epochs + 1):
    train_accuracy, train_auc, train_labels, train_outputs = train(model, device, TrainDataLoader, optimizer, criterion, epoch, num_epochs)
    test_accuracy, auc_score, labels, predictions, filenames = test(model, device, ValDataLoader, epoch, num_epochs)
    scheduler.step()

    # ✅ 保存最优模型
    if auc_score > best_auc:
        best_auc = auc_score
        early_stopping_counter = 0
        best_model_path = os.path.join(os.getcwd(), "checkpoint", "best_model.pth")
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 Saved new best model with AUC: {best_auc:.4f}")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_rounds:
        print(f"⏹️ Early stopping at epoch {epoch}")
        break

print(f"🏁 Training finished! Best AUC: {best_auc:.4f}")