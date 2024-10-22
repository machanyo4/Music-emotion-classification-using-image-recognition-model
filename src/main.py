import torch
from torch.utils.data import Dataset
import os
import random
from PIL import Image
from dataset import MusicDatasets
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
# 既存モデルの使用
from torchvision.models import efficientnet_v2_s
# 独自モデルの使用
# from architect.efficientnet_v2_s_mine import efficientnet_v2_s
#　1ch適応モジュールの読み込み
from architect.adjust1ch import update_model_channels

# Dir_Path
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/"
os.makedirs('../result', exist_ok=True)
os.makedirs('../model', exist_ok=True)
sets = '2048s'
seed = 55
kind = "_gray1chs_pl_decre90"

# ハイパーパラメータ
batch_size = 64
learning_rate = 0.001
num_epochs = 50


# データセットの読み込みと前処理
transform = transforms.Compose(
    [
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        # grayscale1ch 画像の場合----
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # grayscale3ch 画像の場合---
        # transforms.Grayscale(num_output_channels=3),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # calor 画像の場合---
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_full_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=True, random_seed=seed)
train_indexes, valid_indexes = train_test_split(range(len(train_full_datasets)), test_size=0.2, random_state=seed)
train_datasets = Subset(train_full_datasets, train_indexes)
valid_datasets = Subset(train_full_datasets, valid_indexes)
test_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=False, random_seed=seed)

train_loader = DataLoader(dataset = train_datasets, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_datasets, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset = test_datasets, batch_size=batch_size, shuffle=False)

# モデルの構築
# 既存モデルの場合
model = efficientnet_v2_s(weights=None)  # 'IMAGENET1K_V1'
# model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更


#--- 1ch ------------------------------
model = update_model_channels(model)
plmodel_path = '/local/home/matsubara/EfficientNetV2_music_emotion_ctlex/model/prior/imagenet_priorln.pth'
model.load_state_dict(torch.load(plmodel_path))
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更
#-------------------------------------

# 自作モデルの場合
# model = efficientnet_v2_s(num_classes=4)

# print('model : ', model)

# # 全ての畳み込み層の入力チャネル数を確認
# def check_conv_layers(model):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             print(f"{name}: in_channels={module.in_channels}, out_channels={module.out_channels}")

# check_conv_layers(model)

# パラメータ数の表示
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total number of trainable parameters: {count_parameters(model):,}")

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルをGPUに移動
model.to(device)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練およびテストの損失と精度を記録するリスト
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []

# # 最高のテスト精度を保存する変数
# best_valid_accuracy = 0.0  
# best_train_accuracy = 0.0  # 最高の訓練精度を保存する変数
# best_epoch = 0  # 最高のテスト精度を達成したエポックを保存する変数

# 学習スケジューラー LambdaLR
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.90 ** epoch)

# 学習のループ
for epoch in range(num_epochs):
    # 訓練モード
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)

    # テストモード
    model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total_valid += labels.size(0)
            correct_valid += predicted.eq(labels).sum().item()

    valid_accuracy = 100 * correct_valid / total_valid
    valid_loss /= len(valid_loader)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_accuracy)

    # 学習率の取得
    lr = scheduler.get_last_lr()[0]

    # 結果の表示
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%, LR: {lr:.6f}') #, LR: {lr:.6f}
    

    scheduler.step()


    # # 最高の精度を持つモデルを保存
    # if valid_accuracy > best_valid_accuracy or (valid_accuracy == best_valid_accuracy and train_accuracy > best_train_accuracy):
    #     best_valid_accuracy = valid_accuracy
    #     best_train_accuracy = train_accuracy
    #     best_epoch = epoch
    #     torch.save(model.state_dict(), '../model/Best_EfficientnetV2_' + sets + '_' + str(seed) + kind + '.pth')

    # 最終epochのモデルを保存
    if epoch + 1 == num_epochs:
        torch.save(model.state_dict(), '../model/Best_EfficientnetV2_' + sets + '_' + str(seed) + kind + '.pth')
        best_valid_accuracy = valid_accuracy
        best_epoch = epoch

# 最終的な結果の表示
print(f"Valid Accuracy of {best_valid_accuracy:.2f}% achieved at Epoch {best_epoch + 1}. Model saved.")

# グラフの表示
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(valid_loss_list, label='Valid Loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(valid_acc_list, label='Valid Accuracy')
plt.title('Accuracies')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('../result/training_results_' + sets + '_' + str(seed) + kind + '.png')
plt.show()