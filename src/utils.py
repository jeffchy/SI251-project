import torch
from torch import nn

class Record():

    def __init__(self, H, grad_dim):
        self.objs = torch.zeros((H)).float()
        self.delta_objs = torch.zeros((H)).float()
        self.grads = torch.zeros((H, grad_dim)).float()
        self.grad_dim = grad_dim
        self.H = H


    def update_obj(self, obj_val):
        self.delta_objs = torch.cat([torch.tensor([obj_val-self.objs[0]]).float(), self.delta_objs[: -1]])
        # self.delta_objs = obj_val - self.objs
        self.objs = torch.cat([torch.tensor([obj_val]).float(), self.objs[: -1]])

    def update_grad(self, obj_grad):
        obj_grad = torch.tensor(obj_grad).float().view(1, self.grad_dim)
        self.grads = torch.cat([torch.tensor(obj_grad).float(), self.grads[: -1]])


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


