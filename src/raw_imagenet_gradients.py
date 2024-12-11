import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
import matplotlib.pyplot as plt
import os
import numpy as np

# EfficientNetV2をIMAGENET1K_V1の事前学習済み重みで読み込む
model = efficientnet_v2_s(weights='IMAGENET1K_V1')

# model = efficientnet_v2_s(weights=None)
# model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更
# model_path = '/local/home/matsubara/EfficientNetV2_music_emotion_ctlex/model/Best_EfficientnetV2_1024s_11_decre90.pth'
# model.load_state_dict(torch.load(model_path))

# 畳み込み層の重みの大きさを取得
def visualize_weight_distribution(model, save_path='../result/prior/imagenet_weight_distribution_adjusted_detail.png'):
    # 畳み込み層の重みの大きさをリストに格納
    weight_magnitudes = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if layer.weight is not None:
                weight_magnitudes.append(layer.weight.abs().detach().cpu().numpy().flatten())

    # すべての重みを1つのリストにまとめる
    weight_magnitudes = [weight for weight_list in weight_magnitudes for weight in weight_list]

    # ビンの数を設定
    bins = 500
    # ヒストグラム範囲を取得
    hist_range = (0, 0.02)
    # ビンの幅を計算
    bin_width = (hist_range[1] - hist_range[0]) / bins
    print(f"Bin width for bins={bins}: {bin_width:.6f}")

    # 重み分布をヒストグラムとして可視化
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(weight_magnitudes, bins=bins, range=hist_range, color='green', alpha=0.7)
    plt.title('Weight Magnitude Distribution')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Frequency')

    # 横軸の範囲を制限する（例として0〜0.2に設定）
    plt.xlim(0, 0.2)

    # 横軸の目盛りを細かく設定（0.02刻みに設定）
    plt.xticks(np.arange(0, 0.21, 0.02))
    
    # グリッドを追加
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # グラフを保存
    os.makedirs('../result/prior', exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    print(f"Adjusted weight distribution saved to {save_path}")

# 重みの分布を可視化
visualize_weight_distribution(model)