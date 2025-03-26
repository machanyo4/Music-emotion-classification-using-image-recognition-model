import torch
from torch.utils.data import Dataset
import os
import random
import re
from PIL import Image

class MusicDatasets(Dataset):
    def __init__(self, directory=None, sets=None, transform=None, train=True, random_seed=55):
        self.directory = directory
        self.sets = sets
        self.transform = transform
        self.train = train
        self.random_seed = random_seed
        random.seed(random_seed)  # ランダムシードを設定
        self.img_path_and_label = self.ImgPathAndLabel()

    def __len__(self):
        return len(self.img_path_and_label)

    def __getitem__(self, index):
        img_path, label = self.img_path_and_label[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img, label

    def ImgPathAndLabel(self):
        img_path_and_labels = []

        for class_name in ['Q1', 'Q2', 'Q3', 'Q4']:
            class_num = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}[class_name]
            class_dir = os.path.join(self.directory, class_name, self.sets)

            if os.path.exists(class_dir):
                files = [file for file in sorted(os.listdir(class_dir)) if file.endswith(".png")]

                # ファイルリストを9ブロックごとに分割
                block_size = 9
                num_blocks = len(files) // block_size
                file_blocks = [files[i:i+block_size] for i in range(0, len(files), block_size)]

                if self.train:
                    # ブロックをシャッフルしてから8割のブロックを選択
                    random.shuffle(file_blocks)
                    num_selected_blocks = int(0.8 * num_blocks)
                    selected_blocks = file_blocks[:num_selected_blocks]
                else:
                    # 残りの2割のブロックを選択
                    selected_blocks = file_blocks[int(0.8 * num_blocks):]

                # 選択されたブロックからファイルを取得
                selected_files = []
                for block in selected_blocks:
                    selected_files.extend(block)
                
                for file in selected_files:
                    image_path = os.path.join(class_dir, file)
                    image_path_and_label = image_path, class_num
                    img_path_and_labels.append(image_path_and_label)

        return img_path_and_labels