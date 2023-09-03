import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import tqdm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = '../datasets/'

train_transform = [transforms.ToTensor()]
train_transform = transforms.Compose(train_transform)
clean_train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=512,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=0)



def find_similar_img(target, orig_label, k=20):
    trainDataLoader = torch.utils.data.DataLoader(dataset=clean_train_dataset, batch_size=1)
    processBar = tqdm(trainDataLoader, unit='step')
    sims = {}
    for index, (trainImgs, label) in enumerate(processBar):
        if label == orig_label:
            trainImgs = trainImgs.to(device)
            cos_sim = torch.nn.functional.cosine_similarity(target.flatten(), trainImgs.flatten(), dim=0)
            sims[index] = cos_sim
    sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    sim_group = []
    for i in range(k):
        sim_group.append(sims[i][0])

    return sim_group

def linear_interpolation(target_img, sim_id_group, alpha): 
    print("Interpolating images, Alpha {}".format(alpha))
    target_img = target_img.to(device).permute([2, 0, 1]) # [32, 32, 3] -> [3, 32, 32]
    ip_img = []
    sim_img = []
    trainDataLoader = torch.utils.data.DataLoader(dataset=clean_train_dataset, batch_size=1)
    processBar = tqdm(trainDataLoader, unit='step')
    for index, (trainImgs, labels) in enumerate(processBar):
        if index in sim_id_group:
            sim_img.append(trainImgs)
    for img in sim_img: 
        img = img.to(device).squeeze(0)  # [1, 3, 32, 32] -> [3, 32, 32]
        interpolation = (1-alpha) * img + alpha * target_img
        ip_img.append(interpolation)
    ip_img = torch.tensor([img.cpu().detach().numpy() for img in ip_img]).to(device)
    return ip_img
