import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
# from architect.efficientnet_v2_s_test import efficientnet_v2_s
from architect.adjust1ch import update_model_channels
from architect.input_1ch import modify_input_layer_to_grayscale
# Dataset の選択
# from dataset1 import MusicDatasets
# from dataset3 import MusicDatasets
# from dataset import MusicDatasets
# from dataset7 import MusicDatasets
from dataset9 import MusicDatasets

# データセットパスとモデルパス
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/9grayscale" #/grayscale
sets = '2048s'
kind = "gray_raw9_input1ch_decre90"
base_path = "../model/Best_EfficientnetV2_" + sets
seeds = [11, 22, 33, 44, 55]
model_paths = [base_path + '_' + str(i) + kind + '.pth' for i in seeds] #kind

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

accuracys = []
sum_conf_matrix = 0

for index, model_path in enumerate(model_paths):
    seed = (index + 1) * 11

    test_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=False, random_seed=seed)
    test_loader = DataLoader(dataset = test_datasets, batch_size=batch_size, shuffle=False)

    # モデルの構築
    model = efficientnet_v2_s(weights=None)
    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更

    #--- input_1ch ------------------------------
    model = modify_input_layer_to_grayscale(model)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更
    #-------------------------------------

    #--- gray_1chs-----------------------------------------------------------------------------------------
    # model = update_model_channels(model)
    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更
    #--------------------------------------------------------------------------------------------------

    # 自作モデルの場合
    # model = efficientnet_v2_s(num_classes=4)
    model.load_state_dict(torch.load(model_path))

    # デバイスの指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Using device: {device}")
    print('[LOG] Complete parameter adaptation from ' + model_path + ' .')

    # モデルをGPUに移動
    model.to(device)
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
    accuracys.append(accuracy)

    # 混同行列の作成
    conf_matrix = confusion_matrix(all_labels, all_preds)
    sum_conf_matrix += conf_matrix

# Test Accuracyの平均を計算
sum_accuracy = sum(accuracys)
mean_accuracy = sum_accuracy/len(accuracys)
print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")

# 混同行列の平均を計算
mean_conf_matrix = sum_conf_matrix/len(model_paths)
print("--- Mean Confusion Matrix ---")
print(mean_conf_matrix)

# 混同行列から真のラベルと予測ラベルを再現する
y_true = []
y_pred = []

for i, row in enumerate(mean_conf_matrix):
    for j, value in enumerate(row):
        y_true.extend([i] * int(value))
        y_pred.extend([j] * int(value))

# precision, recall, f1-score を計算
class_names = ['Q1', 'Q2', 'Q3', 'Q4']
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))

# 結果を表示
for i, class_name in enumerate(class_names):
    print(f"{class_name}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-Score={f1[i]:.2f}")

# 混同行列を割合に変換
conf_matrix_percent = mean_conf_matrix / mean_conf_matrix.sum(axis=1, keepdims=True) * 100

# 混同行列の可視化（割合表示）
class_names = ['Q1', 'Q2', 'Q3', 'Q4']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 14})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix (Percentage)', fontsize=16)
plt.show()

# 混同行列の保存
plt.savefig('../result/' + kind + '/' + sets + '/confusion_matrix_' + sets + '_' + 'mean' + kind  + '.png')