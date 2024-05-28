import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import MusicDatasets
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import warnings

# 警告を非表示（torch backends のバージョン警告？）
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# データセットパスとモデルパス
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/"
sets = '1024s'
seed = 11
model_path = "../model/Best_EfficientnetV2_" + sets + '_' + str(seed) + ".pth"
os.makedirs('../result/heatmap', exist_ok=True)

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[LOG] Using device: {device}")

# ハイパーパラメータ
batch_size = 64

# 前処理
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=False, random_seed=seed)
test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)

# モデルの構築
model = efficientnet_v2_s(num_classes=4)
model.load_state_dict(torch.load(model_path))
print('[LOG] Complete parameter adaptation from ' + model_path + ' .')

# デバイスの指定
model.to(device)
model.eval()

# モデルのノード名を取得
train_nodes, eval_nodes = get_graph_node_names(model)
# print(f"Train nodes: {train_nodes}")
# print.f"Eval nodes: {eval_nodes}")

# Grad-CAM のための関数
def grad_cam(model, images, target_layer, class_idx=None):
    feature_extractor = create_feature_extractor(model, return_nodes={target_layer: 'features', 'classifier.1': 'classifier'})
    outputs = feature_extractor(images)
    features = outputs['features']
    features.retain_grad()  # 勾配を保持するように設定
    outputs = outputs['classifier']
    if class_idx is None:
        class_idx = outputs.argmax(dim=1)
    
    one_hot_output = torch.zeros_like(outputs)
    one_hot_output.scatter_(1, class_idx.view(-1, 1), 1.0)
    
    model.zero_grad()
    outputs.backward(gradient=one_hot_output, retain_graph=True)
    
    grads = features.grad
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_grads[i]
    
    heatmap = torch.mean(features, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    heatmap = cv2.resize(heatmap, (images.shape[2], images.shape[3]))  # ここでリサイズ
    
    return heatmap

# 各クラスごとに正しい予測を行った画像を10枚ずつ取得
correct_images = {i: [] for i in range(4)}
correct_labels = {i: [] for i in range(4)}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        for i in range(images.size(0)):
            label = labels[i].item()
            if preds[i] == labels[i] and len(correct_images[label]) < 10:
                correct_images[label].append(images[i].unsqueeze(0))
                correct_labels[label].append(labels[i].item())
        
        if all(len(v) >= 10 for v in correct_images.values()):
            break

# Grad-CAM の可視化と保存
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class_names = ['Q1', 'Q2', 'Q3', 'Q4']  # クラス名を定義

for class_idx, img_tensors in correct_images.items():
    for idx, img_tensor in enumerate(img_tensors):
        img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 0.229 + 0.485, 0, 1)  # Undo normalization

        heatmap = grad_cam(model, img_tensor, 'features')  # 修正された特徴マップ層
        cam_img = show_cam_on_image(img, heatmap)

        fig, ax = plt.subplots()
        cax = ax.imshow(cam_img)
        fig.colorbar(cax, ax=ax)
        
        ax.set_title(f'Grad-CAM for Class {class_names[class_idx]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        plt.savefig(f'../result/heatmap/grad_cam_class_{class_names[class_idx]}_{idx}.png')
        plt.show()

print("[LOG] Successfully created heat maps.")
