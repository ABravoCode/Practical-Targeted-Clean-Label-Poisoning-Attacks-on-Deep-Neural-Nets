import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

import options
args = options.options()


def create_poison_data(new_imgs):
    poison_dataset = torch.tensor([img.cpu().detach().numpy() for img in new_imgs])
    for i in range(len(poison_dataset)):
        poison_dataset.data[i] = torch.clamp(poison_dataset.data[i], 0, 1)
        poison_dataset.data[i] = poison_dataset.data[i].reshape(3, 32, 32)
    torch.save(poison_dataset, "./poison_dataset_{}_{}.pt".format(int(args.clean_target_id),int(args.clean_label)))
    return poison_dataset


class UnlearnableDataset(Dataset):
    def __init__(self, clean_label):
        self.clean_label = clean_label
        self.transformations = transforms.Compose([])
        
    def __getitem__(self, index):
        data = torch.load("./poison_dataset_{}_{}.pt".format(int(args.clean_target_id),int(args.clean_label)))
        data = self.transformations(data)
        img = data[index].clone().detach().requires_grad_(True)
        _label_ = self.clean_label
        temp = (img, _label_)
        return temp
    
    def __len__(self):
        return args.num_poison
