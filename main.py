import options

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
torch.multiprocessing.set_start_method('spawn', force=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torchvision
from torchvision import datasets, transforms

from tqdm import tqdm
import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from Pertubation import PerturbationTool
from trainers import generate_noise
from tools import find_similar_img, linear_interpolation 
from poisoned_datasets import UnlearnableDataset, create_poison_data

from models.ResNet import ResNet18

import options
args = options.options()
print(args)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

train_transform = [
    transforms.ToTensor()
]
test_transform = [
    transforms.ToTensor()
]
train_transform = transforms.Compose(train_transform)
test_transform = transforms.Compose(test_transform)

clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
clean_test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)

clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=0)
clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=0)

clean_target = torch.tensor(clean_test_dataset.data[args.clean_target_id].astype(np.float32)/255).to(device)

_CLASS_ = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
    base_model = ResNet18()
    base_model = base_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=base_model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)

    similar_imgs = find_similar_img(clean_target, orig_label = int(args.clean_label), k = args.num_poison)
    target_train_dataset = linear_interpolation(clean_target, similar_imgs, alpha=args.alpha)
    raw_dataset = create_poison_data(target_train_dataset) # float, save into .pt
    unlearnable_train_dataset = UnlearnableDataset(args.clean_label)
    perturb_dataset = ConcatDataset([clean_train_dataset, unlearnable_train_dataset])
    purturb_loader = DataLoader(dataset=perturb_dataset, batch_size=args.batch_size,
                                    shuffle=False, pin_memory=False,
                                    drop_last=False, num_workers=0)

    # noise_generator = PerturbationTool(epsilon=0.03137254901960784, num_steps=20, step_size=0.0031372549019607846)
    noise_generator = PerturbationTool(epsilon=args.eps, num_steps=20, step_size=args.step_size)
    perturb_img, noise = generate_noise(base_model, criterion, optimizer, noise_generator, purturb_loader, max_iter=10)

    uni_train_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    uni_train_transform = transforms.Compose(uni_train_transform)
    perturb_untrained_dataset_image = []
    perturb_untrained_dataset_label = []
    poisoned_train_dataset = []

    for single_data in perturb_dataset:
        perturb_untrained_dataset_image.append(single_data[0])
        perturb_untrained_dataset_label.append(single_data[1])

    for index, single_data in enumerate(perturb_untrained_dataset_image):
        # if perturb_untrained_dataset_label == args.clean_label:
        if index > 49999:
            single_data = single_data + noise[index]
        poisoned_data = single_data.permute([1, 2, 0]).detach().to('cpu').clamp_(0, 1).numpy().astype(np.float32)
        poisoned_train_dataset.append((poisoned_data, perturb_untrained_dataset_label[index]))

    model = ResNet18()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

    poisoned_loader = DataLoader(dataset=poisoned_train_dataset, batch_size=args.batch_size,
                                    shuffle=False, pin_memory=False,
                                    drop_last=False, num_workers=0)

    epoch = 0
    condition = True
    while condition:
        # Train
        model.train(True)
        pbar = tqdm(poisoned_loader, total=len(poisoned_loader))
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            images = images.permute([0, 3, 1, 2])
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc*100, loss))
        scheduler.step()
        # optimizer.step()

        # Eval
        model.eval()
        model.train(False)
        correct, total = 0, 0
        clear_label_correct = 0
        for i, (images, labels) in enumerate(clean_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        # tqdm.write("Correct: %d, Total: %d" % (correct, total))
        tqdm.write('Epoch %d Clean Accuracy %.2f\n' % (epoch, acc*100))
        epoch += 1

        if acc > args.beta or epoch > 30:
            condition = False

    test_image = clean_target.permute([2, 0, 1]).unsqueeze(0)
    print(test_image.shape)
    logits = model(test_image)
    _, pred = torch.max(logits, 1)
    print("Original:", _CLASS_[int(args.clean_label)], "Prediction:", _CLASS_[int(pred)])
    print(logits)

