#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PINN_sph 
@File    ：utils.py
@Author  ：LiangL. Yan
@Date    ：2023/1/09 21:56
"""
import sys
import numpy as np
import torch
import os


def norm(x):
    """Normalization."""
    mean = np.mean(x)
    var = np.var(x)

    return (x - mean) / var


def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    else :
        return 'cpu'


def gradients(x, y, order=1):
    """Computer the gradient : Dy / Dx."""
    if order == 1:
        # return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
        #                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        return torch.autograd.grad(y, x,
                                   grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]

    else:
        return gradients(gradients(x, y), x, order=order-1)


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def restore_model(model, resume_epochs, model_save_dir):
    """Restore the trained PINNs/gPINNs."""
    print('Loading the trained models from step {}...'.format(resume_epochs))
    path = os.path.join(model_save_dir, '{}-pinn.ckpt'.format(resume_epochs))

    model.net.load_state_dict(torch.load(path))
    print("Success load model with epoch {} !!!".format(resume_epochs))


# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))
