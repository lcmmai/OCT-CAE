import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path


class CustomDataset(Dataset):
    def __init__(self, args):
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        data_path = Path(args.data.data_path)
        txt_file = data_path / 'train.txt'
        mode = 'train'

        # 读取 txt 文件
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                image_name, label = line.strip().split()
                image_path = data_path / mode / image_name
                self.data.append((image_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, image_path.name[:-4]


class PairSampler:
    def __init__(self, dataset, batch_size, num_samples_per_epoch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = {}
        self.num_samples_per_epoch = num_samples_per_epoch

        # 按类存储每个图像索引
        for idx, (_, label) in enumerate(self.dataset.data):
            if label not in self.labels:
                self.labels[label] = []
            self.labels[label].append(idx)

    def __iter__(self):
        batch_size_half = self.batch_size // 2
        keys = list(self.labels.keys())
        while True:
            # 确保 indices_a 始终来自类别 0
            class_a = 0
            indices_a = random.sample(self.labels[class_a], batch_size_half)

            # 从类别 1, 2, 3 中随机选择一个类别
            class_b = random.choice([1, 2, 3])
            indices_b = random.sample(self.labels[class_b], batch_size_half)
            yield indices_a + indices_b


def custom_collate_fn(batch):
    images, labels, names = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels, names


def create_dataloader(args):
    dataset = CustomDataset(args)
    num_samples_per_epoch = len(dataset) * 2
    # args.batch_size = 2
    sampler = PairSampler(dataset, args.batch_size, num_samples_per_epoch)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn)
    return dataloader
