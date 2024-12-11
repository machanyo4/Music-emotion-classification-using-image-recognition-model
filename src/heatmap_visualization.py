import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import MusicDatasets
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import warnings
# from architect.adjust1ch import update_model_channels
from architect.input_1ch import modify_input_layer_to_grayscale

# 警告を非表示（torch backends のバージョン警告？）
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# データセットパスとモデルパス
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/grayscale/" #grayscale/
sets = '1024s'
seed = 33
kind = "gray_raw_input3ch_decre90"
model_path = "../model/Best_EfficientnetV2_" + sets + '_' + str(seed) + kind + ".pth"

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[LOG] Using device: {device}")

# ハイパーパラメータ
batch_size = 64

# データセットの読み込みと前処理
transform = transforms.Compose(
    [
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        # grayscale1ch 画像の場合----
        # transforms.Grayscale(num_output_channels=1),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
        # grayscale3ch 画像の場合---
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # calor 画像の場合---
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=False, random_seed=seed)
test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)

# モデルの構築
model = efficientnet_v2_s(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更

#--- input_1ch ------------------------------
# model = modify_input_layer_to_grayscale(model)
# model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更
#-------------------------------------

#--- gray_1chs -----------------------------------------------------------------------------------------
# model = update_model_channels(model)
# model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更
#--------------------------------------------------------------------------------------------------

# 自作モデルの場合
# model = efficientnet_v2_s(num_classes=4)

# print('model : ', model)

model.load_state_dict(torch.load(model_path))

# デバイスの指定
model.to(device)
model.eval()

# モデルのノード名を取得
# train_nodes, eval_nodes = get_graph_node_names(model)
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
    
    heatmap = cv2.resize(heatmap, (images.shape[2], images.shape[3]))  # 画像サイズに合わせてリサイズ
    
    return heatmap

# 各クラスごとに正しい予測と誤った予測を10枚ずつ取得
correct_images = {i: [] for i in range(4)}
correct_labels = {i: [] for i in range(4)}
incorrect_images = {i: [] for i in range(4)}
incorrect_labels = {i: [] for i in range(4)}
incorrect_preds = {i: [] for i in range(4)}  # 誤った予測のクラスも保存

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(images.size(0)):
            label = labels[i].item()
            pred = preds[i].item()

            # 正しい予測の場合
            if pred == label and len(correct_images[label]) < 10:
                correct_images[label].append(images[i].unsqueeze(0))
                correct_labels[label].append(labels[i].item())
            # 誤った予測の場合
            elif pred != label and len(incorrect_images[label]) < 10:
                incorrect_images[label].append(images[i].unsqueeze(0))
                incorrect_labels[label].append(labels[i].item())
                incorrect_preds[label].append(pred)

        # 全クラスで10枚集まったらループを抜ける
        if all(len(v) >= 10 for v in correct_images.values()) and all(len(v) >= 10 for v in incorrect_images.values()):
            break

# Grad-CAM の可視化と保存
def show_cam_on_image(img, mask):
    # グレースケール画像の場合、チャンネル数を3に拡張
    if len(img.shape) == 2:  # (H, W) の場合
        img = np.stack([img, img, img], axis=-1)  # (H, W, 3) に拡張
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class_names = ['Q1', 'Q2', 'Q3', 'Q4']  # クラス名を定義

# 正しい予測画像の保存
for class_idx, img_tensors in correct_images.items():
    for idx, img_tensor in enumerate(img_tensors):
        # 3ch 画像の場合----
        img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 0.229 + 0.485, 0, 1)  # 正規化を元に戻す
        # 1ch 画像の場合----
        # img = img_tensor.squeeze().cpu().numpy()  # グレースケールは(H, W)
        # img = np.clip(img * 0.5 + 0.5, 0, 1)  # 正規化を元に戻す

        heatmap = grad_cam(model, img_tensor, 'features')
        cam_img = show_cam_on_image(img, heatmap)

        # 元画像を保存
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f'Original Image {class_names[class_idx]}', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig(f'../result/heatmap/original_class_{class_names[class_idx]}_pred_{class_names[class_idx]}_{idx}.png', bbox_inches='tight')
        plt.close()

        # ヒートマップを保存
        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap, cmap='jet')
        plt.title(f'Grad-CAM {class_names[class_idx]}', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig(f'../result/heatmap/heatmap_class_{class_names[class_idx]}_pred_{class_names[class_idx]}_{idx}.png', bbox_inches='tight')
        plt.close()

# 誤った予測画像の保存
for class_idx, img_tensors in incorrect_images.items():
    for idx, img_tensor in enumerate(img_tensors):
        # 3ch 画像の場合----
        img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 0.229 + 0.485, 0, 1)  # 正規化を元に戻す
        # 1ch 画像の場合----
        # img = img_tensor.squeeze().cpu().numpy()  # グレースケールは(H, W)
        # img = np.clip(img * 0.5 + 0.5, 0, 1)  # 正規化を元に戻す

        pred_label = incorrect_preds[class_idx][idx]
        heatmap = grad_cam(model, img_tensor, 'features')
        cam_img = show_cam_on_image(img, heatmap)

        # 元画像を保存（誤った予測ラベルもファイル名に付与）
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f'Original Image {class_names[class_idx]} (Pred: {class_names[pred_label]})', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig(f'../result/heatmap/original_class_{class_names[class_idx]}_pred_{class_names[pred_label]}_{idx}.png', bbox_inches='tight')
        plt.close()

        # ヒートマップを保存（誤った予測ラベルもファイル名に付与）
        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap, cmap='jet')
        plt.title(f'Grad-CAM {class_names[class_idx]} (Pred: {class_names[pred_label]})', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig(f'../result/heatmap/heatmap_class_{class_names[class_idx]}_pred_{class_names[pred_label]}_{idx}.png', bbox_inches='tight')
        plt.close()

print("[LOG] Heatmaps of correct and incorrect predictions were successfully saved.")
