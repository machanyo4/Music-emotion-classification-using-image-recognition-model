import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

def modify_input_layer_to_grayscale(model):
    """
    EfficientNetV2の最初の畳み込み層の入力チャンネルを3から1に変更し、
    事前学習済みの重みをグレースケール（1チャンネル）に対応するように平均化する。
    """
    # 最初の畳み込み層を取得
    first_conv_layer = model.features[0][0]
    
    # Conv2dであることを確認
    if isinstance(first_conv_layer, nn.Conv2d):
        # 重み (out_channels, in_channels, kernel_height, kernel_width)
        original_weight = first_conv_layer.weight.data
        
        # RGBチャンネル (3チャンネル) を次元平均して1チャンネルに変換
        # (out_channels, 1, kernel_height, kernel_width)
        new_weight = original_weight.mean(dim=1, keepdim=True)
        
        # 新しいConv2dを作成
        new_conv = nn.Conv2d(
            in_channels=1,  # グレースケールなので1チャンネル
            out_channels=first_conv_layer.out_channels, 
            kernel_size=first_conv_layer.kernel_size, 
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding, 
            bias=first_conv_layer.bias is not None
        )
        
        # 重みを新しいConv2dにセット
        new_conv.weight = nn.Parameter(new_weight)
        
        # バイアスがある場合はそのままコピー
        if first_conv_layer.bias is not None:
            new_conv.bias = first_conv_layer.bias
        
        # モデルの最初の畳み込み層を新しいConv2dに置き換える
        model.features[0][0] = new_conv
    
    return model