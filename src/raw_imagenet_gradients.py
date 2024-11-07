import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
import matplotlib.pyplot as plt
import os

# EfficientNetV2をIMAGENET1K_V1の事前学習済み重みで読み込む
model = efficientnet_v2_s(weights='IMAGENET1K_V1')

# 畳み込み層の重みの大きさを取得
def visualize_weight_distribution(model, save_path='../result/prior/weight_distribution.png'):
    # 畳み込み層の重みの大きさをリストに格納
    weight_magnitudes = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if layer.weight is not None:
                weight_magnitudes.append(layer.weight.abs().cpu().numpy().flatten())

    # すべての重みを1つのリストにまとめる
    weight_magnitudes = [weight for weight_list in weight_magnitudes for weight in weight_list]

    # 重み分布をヒストグラムとして可視化
    plt.figure(figsize=(10, 6))
    plt.hist(weight_magnitudes, bins=50, color='green', alpha=0.7)
    plt.title('Weight Magnitude Distribution')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Frequency')

    # グラフを保存
    os.makedirs('../result/prior', exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    print(f"Weight distribution saved to {save_path}")

# 重みの分布を可視化
visualize_weight_distribution(model)
