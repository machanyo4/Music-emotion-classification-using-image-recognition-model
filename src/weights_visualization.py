import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms
from dataset import MusicDatasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.models import efficientnet_v2_s
# from architect.efficientnet_v2_s_test import efficientnet_v2_s

def visualize_final_layer_weights(model, sets, seed):
    # 最終層の重みを取得
    final_layer_weights = None
    for name, param in model.named_parameters():
        if 'classifier' in name and 'weight' in name:
            final_layer_weights = param.data
            break

    if final_layer_weights is None:
        print("Final layer weights not found.")
        return

    # 最終層の重みを可視化
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_layer_weights.cpu().numpy(), cmap='viridis')
    plt.title("Final Layer Weights")
    plt.xlabel("Output Neurons")
    plt.ylabel("Input Neurons")
    plt.show()
    plt.savefig('../result/weights' + sets + '_' + str(seed) + '.png')


# データセットパスとモデルパス
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/"
sets = "1024s"
base_path = "../model/Best_EfficientnetV2_" + sets
seed = 11
model_path = base_path + '_' + str(seed) + '.pth'

# ハイパーパラメータ
batch_size = 64

# 前処理
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# モデルの構築
model = efficientnet_v2_s(num_classes=4)

# モデルの重みを読み込み
model.load_state_dict(torch.load(model_path))

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# テストデータセットの読み込み
test_dataset = MusicDatasets(dataset_path, sets, transform=transform, train=False, random_seed=seed)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 予測と真のラベルを格納するリスト
all_preds = []
all_labels = []

# モデルを評価モードに
model.eval()

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
print(f"Accuracy: {accuracy * 100:.2f}%")

# 混同行列の作成
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

# 最終層の重みを可視化
visualize_final_layer_weights(model, sets, seed)
