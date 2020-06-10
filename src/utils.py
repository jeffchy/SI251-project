import torch

class Record():

    def __init__(self, H, grad_dim):
        self.objs = torch.zeros((H)).float()
        self.delta_objs = torch.zeros((H)).float()
        self.grads = torch.zeros((H, grad_dim)).float()
        self.grad_dim = grad_dim
        self.H = H


    def update_obj(self, obj_val):
        self.delta_objs = torch.cat([torch.tensor([obj_val-self.objs[0]]).float(), self.delta_objs[: -1]])
        self.objs = torch.cat([torch.tensor([obj_val]).float(), self.objs[: -1]])

    def update_grad(self, obj_grad):
        obj_grad = torch.tensor(obj_grad).float().view(1, self.grad_dim)
        self.grads = torch.cat([torch.tensor(obj_grad).float(), self.grads[: -1]])

