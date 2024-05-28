import torch
from torch.utils.data import Dataset
import os
import random
from PIL import Image
from dataset import MusicDatasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from collections import Counter
from sklearn.model_selection import train_test_split

# Dir_Path
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/"
os.makedirs('../result', exist_ok=True)
os.makedirs('../model', exist_ok=True)
sets = '2048s'
seed = 44

# ハイパーパラメータ
batch_size = 64
learning_rate = 0.001
num_epochs = 50


# データセットの読み込みと前処理
transform = transforms.Compose(
    [
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_full_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=True, random_seed=seed)
train_indexes, valid_indexes = train_test_split(range(len(train_full_datasets)), test_size=0.2, random_state=seed)
train_datasets = Subset(train_full_datasets, train_indexes)
valid_datasets = Subset(train_full_datasets, valid_indexes)
test_datasets = MusicDatasets(dataset_path, sets, transform=transform, train=False, random_seed=seed)

print('len train', len(train_datasets))
print('len valid', len(valid_datasets))
print('len test', len(test_datasets))

src_lists = test_datasets.img_path_and_label
for item in src_lists:
    print(item)

# train_loader = DataLoader(dataset = train_datasets, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(dataset = valid_datasets, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset = test_datasets, batch_size=batch_size, shuffle=False)

# テストデータセット内のすべてのバッチを取得
# for images, labels in test_loader:
#     print('image path :', images, 'label :', labels)