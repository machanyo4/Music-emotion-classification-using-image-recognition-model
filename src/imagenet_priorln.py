import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import efficientnet_v2_s
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
#　1ch適応モジュールの読み込み
from architect.adjust1ch import update_model_channels

# データセットパスを設定
train_dir = "/local/home/Data/train"
val_dir = "/local/home/Data/val"

# ハイパーパラメータ
batch_size = 64
learning_rate = 0.001
num_epochs = 100

# 結果保存ディレクトリ
os.makedirs('../result/prior', exist_ok=True)
os.makedirs('../model/prior', exist_ok=True)

# データの前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # grayscale1ch 画像の場合----
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# データセットの作成
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=transform),
    'val': datasets.ImageFolder(val_dir, transform=transform),
}

# DataLoaderの作成
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
}


# モデルの読み込み（EfficientNetV2）
model = efficientnet_v2_s(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1000)
model = update_model_channels(model)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.90 ** epoch)

# 学習のループ
train_loss_list, valid_loss_list = [], []
train_acc_list, valid_acc_list = [], []

for epoch in range(num_epochs):
    # 訓練モード
    model.train()
    train_loss, correct_train, total_train = 0.0, 0, 0

    for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch",leave=False):
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
    train_loss /= len(dataloaders['train'])
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)

    # テストモード
    model.eval()
    valid_loss, correct_valid, total_valid = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch",leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total_valid += labels.size(0)
            correct_valid += predicted.eq(labels).sum().item()

    valid_accuracy = 100 * correct_valid / total_valid
    valid_loss /= len(dataloaders['val'])
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_accuracy)

    # 学習率の取得
    # lr = scheduler.get_last_lr()[0]

    # 結果の表示
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%')
    
    # scheduler.step()

    # モデル保存
    if epoch + 1 == num_epochs:
        torch.save(model.state_dict(), f'../model/prior/imagenet_priorln.pth')

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

plt.savefig('../result/prior/training_results.png')
plt.show()