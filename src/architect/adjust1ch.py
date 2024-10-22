import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

def adjust_channels(layer, scale_factor=1/3):
    """
    Adjust the number of input and output channels for Conv2d and BatchNorm2d layers.
    """
    if isinstance(layer, nn.Conv2d):
        new_in_channels = max(1, int(layer.in_channels * scale_factor))
        new_out_channels = max(1, int(layer.out_channels * scale_factor))
        new_groups = max(1, int(layer.groups * scale_factor)) if layer.groups > 1 else 1
        
        # Adjust weights and bias
        new_weight = layer.weight[:new_out_channels, :new_in_channels, :, :].clone()
        new_bias = layer.bias[:new_out_channels].clone() if layer.bias is not None else None
        
        layer.in_channels = new_in_channels
        layer.out_channels = new_out_channels
        layer.groups = new_groups
        layer.weight = nn.Parameter(new_weight)
        if new_bias is not None:
            layer.bias = nn.Parameter(new_bias)

    if isinstance(layer, nn.BatchNorm2d):
        new_num_features = max(1, int(layer.num_features * scale_factor))
        
        layer.num_features = new_num_features
        layer.running_mean.data = layer.running_mean.data[:new_num_features].clone()
        layer.running_var.data = layer.running_var.data[:new_num_features].clone()
        layer.weight = nn.Parameter(layer.weight.data[:new_num_features].clone())
        layer.bias = nn.Parameter(layer.bias.data[:new_num_features].clone())

def adjust_classifier(model, scale_factor=1/3):
    """
    Adjust the classifier layer of the model to match the new number of features.
    """
    in_features = model.classifier[1].in_features
    out_features = model.classifier[1].out_features
    new_in_features = max(1, int(in_features * scale_factor))
    
    # Create a new Linear layer with adjusted input features
    model.classifier[1] = nn.Linear(new_in_features, out_features)

def update_model_channels(model, scale_factor=1/3):
    """
    Update the model to handle reduced channels.
    """
    for name, layer in model.named_modules():
        adjust_channels(layer, scale_factor)

    # Adjust the first convolutional layer separately to accept 1 input channel
    first_conv = list(model.modules())[1]
    if isinstance(first_conv, nn.Conv2d):
        first_conv.in_channels = 1
        new_weight = first_conv.weight.mean(dim=1, keepdim=True).clone()  # Average across input channels
        first_conv.weight = nn.Parameter(new_weight)
    
    adjust_classifier(model, scale_factor)
    
    return model

model = efficientnet_v2_s(weights='IMAGENET1K_V1')  # 'IMAGENET1K_V1'
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更


#--- 1ch ------------------------------
model = update_model_channels(model)
#-------------------------------------

# print('model : ', model)

# 全ての畳み込み層の入力チャネル数を確認
def check_conv_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"{name}: in_channels={module.in_channels}, out_channels={module.out_channels}")

# check_conv_layers(model)

# パラメータ数の表示
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)