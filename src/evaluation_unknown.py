import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import MusicDatasets
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from architect.efficientnet_v2_s_test import efficientnet_v2_s
from architect.adjust1ch import update_model_channels


# データセットパスとモデルパス
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/"
sets = '2048s'
seed = 55
kind = "_gray1chs_pl_decre90"
model_path = "../model/Best_EfficientnetV2_" + sets + '_' + str(seed) + kind + ".pth"

# ハイパーパラメータ
batch_size = 64

# 前処理
transform = transforms.Compose(
    [
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        # grayscale 画像の場合---
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # grayscale3ch 画像の場合---
        # transforms.Grayscale(num_output_channels=3),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # calor 画像の場合---
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=False, random_seed=seed)
test_loader = DataLoader(dataset = test_datasets, batch_size=batch_size, shuffle=False)

# モデルの構築
model = efficientnet_v2_s(weights=None)
# model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更

#--- 1ch -----------------------------------------------------------------------------------------
model = update_model_channels(model)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更
#--------------------------------------------------------------------------------------------------

# 自作モデルの場合
# model = efficientnet_v2_s(num_classes=4)

# print('model : ', model)

model.load_state_dict(torch.load(model_path))

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[LOG] Using device: {device}")
print('[LOG] Complete parameter adaptation from ' + model_path + ' .')

# モデルをGPUに移動
model.to(device)

# # パラメータ数の表示
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total number of trainable parameters: {count_parameters(model):,}")

# モデルを評価モードに
model.eval()

# 予測と真のラベルを格納するリスト
all_preds = []
all_labels = []

# バッチごとに予測
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracyの計算
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 混同行列の作成
conf_matrix = confusion_matrix(all_labels, all_preds)

# 混同行列を割合に変換
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100

print("--- Confusion Matrix ---")
print(conf_matrix)

# 混同行列の可視化（割合表示）
class_names = ['Q1', 'Q2', 'Q3', 'Q4']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Percentage)')
plt.show()

# 混同行列の保存
plt.savefig('../result/confusion_matrix_' + sets + '_' + str(seed) + kind + '.png')