import torch
from tqdm import tqdm
import options
args = options.options()


def generate_noise(base_model, criterion, optimizer, generator, train_loader, ZO_method, max_iter=10):        
    noise = torch.zeros([50000+args.num_poison, 3, 32, 32])
    data_iter = iter(train_loader)
    condition = True
    train_idx = 0

    while condition:
    # optimize theta for M steps
        base_model.train()
        for param in base_model.parameters():
            param.requires_grad = True
        for j in range(0, max_iter):
            try:
                (images, labels) = next(data_iter)
            except:
                train_idx = 0
                data_iter = iter(train_loader)
                (images, labels) = next(data_iter)
            
            for i, _ in enumerate(images):
                # Update noise to images
                images[i] += noise[train_idx]
                train_idx += 1
            images, labels = images.cuda(), labels.cuda()
            base_model.zero_grad()
            optimizer.zero_grad()
            logits = base_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
            optimizer.step()

        # Perturbation over entire dataset
        idx = 0
        for param in base_model.parameters():
            param.requires_grad = False
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch_start_idx, batch_noise = idx, []
            for i, _ in enumerate(images):
                # Update noise to images
                batch_noise.append(noise[idx])
                idx += 1
            batch_noise = torch.stack(batch_noise).cuda()
            
            # Update sample-wise perturbation
            base_model.eval()
            images, labels = images.cuda(), labels.cuda()
            perturb_img, eta = generator.min_min_attack(images, labels, base_model, optimizer, criterion, ZO_method, random_noise=batch_noise)
            
            for i, delta in enumerate(eta):
                noise[batch_start_idx+i] = delta.clone().detach().cpu()
            
        # Eval stop condition
        eval_idx, total, correct = 0, 0, 0
        for i, (images, labels) in enumerate(train_loader):
            for i, _ in enumerate(images):
                # Update noise to images
                images[i] += noise[eval_idx]
                eval_idx += 1
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = base_model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total

        print('correct:', correct, 'total:', total)
        print('Accuracy %.4f' % (acc*100))

        if acc > 0.8000:
            condition=False
              
    return perturb_img, noise