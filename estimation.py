import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

import options
args = options.options()

class grad_estimator():
    def __init__(self, model, true_label, orig_shape):
        self.model = model
        self.true_label = true_label
        self.orig_shape = orig_shape

    def objective_f(self, x):
        reshaped_x = torch.reshape(x, self.orig_shape)
        logits = self.model(reshaped_x)
        
        loss_function = torch.nn.CrossEntropyLoss()
        loss_hat= loss_function(logits, self.true_label)

        return loss_hat
    
    def grad_est(self, x, mu, q, d):
        m, sigma = 0, 100 # mean and standard deviation
        sum = 0
        f_origin = self.objective_f(x)
        for i in range(q):
            u = torch.normal(m, sigma, size=(args.batch_size, 3072)).to(device)
            u_norm = torch.norm(u, 2)
            u = u/u_norm
            sum += (self.objective_f(x+mu*u)-f_origin)*u
            
        result = (d*sum)/(mu*q)
        result = result.reshape(self.orig_shape)
        
        return result
    
