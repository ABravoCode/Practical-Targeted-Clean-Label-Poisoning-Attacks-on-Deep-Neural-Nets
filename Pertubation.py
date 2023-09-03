import numpy as np
import torch
from torch.autograd import Variable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import options
args = options.options()

from estimation import grad_estimator

class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=[20, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack(self, images, labels, model, optimizer, criterion, ZO_method=False, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            
            if ZO_method:
                mu = 0.01 #smoothing parameter
                q  = 10  #no of random directions
                estimator = grad_estimator(model, labels, images.shape)
                
                images = images.view(args.batch_size, -1)
                d = args.batch_size*images.shape[1]
                perturb_img_grad = estimator.grad_est(images, mu, q, d)

            else:
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    if hasattr(model, 'classify'):
                        model.classify = True
                    logits = model(perturb_img)
                    loss = criterion(logits, labels)
                else:
                    logits, loss = criterion(model, perturb_img, labels, optimizer)
                perturb_img.retain_grad()
                loss.backward()
                perturb_img_grad = perturb_img.grad.data


            eta = self.step_size * perturb_img_grad.sign() * (-1) 
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)  # torch.Size([512, 3, 32, 32])
            
            images.data = images.data.reshape(perturb_img.data.shape)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta
    
