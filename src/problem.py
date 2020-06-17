import numpy as np
import torch
from torch import nn
from src.utils import Record


def robust_loss(x, y):
    squered_loss = (x - y) ** 2
    loss = torch.mean(squered_loss / (squered_loss + 1))  # c = 1.0
    return loss


class LinearRegressionNN(nn.Module):
    def __init__(self, W, b, robust=True):
        super(LinearRegressionNN, self).__init__()
        self.W = nn.Parameter(W.float(), requires_grad=True)
        self.b = nn.Parameter(b.float(), requires_grad=True)
        self.loss = robust_loss if robust else torch.nn.MSELoss()

    def forward(self, x, y):
        out = torch.matmul(self.W, x) + self.b
        loss = self.loss(out, y)
        return loss


class LinearRegression():
    def __init__(self, num_feature, N, H, robust=True):
        """
        y = W x + b, W \in R^{num_features}, b \in R
        min_{W, b}
        """
        self.H = H
        self.N = N
        self.num_feature = num_feature
        self.problem_name = 'LR'
        self.state_dim = (num_feature + 1) + H * 1 + (num_feature + 1) * H
        self.action_dim = (num_feature + 1)
        self.max_step = 100
        self.end_obj = 0.34
        self.loss = robust_loss if robust else torch.nn.MSELoss()
        self.generate()
        self.reset()



    def reset(self):
        self.T = 0
        self.W = torch.randn((self.num_feature,))
        self.b = torch.randn((1))

        current_obj, current_grad = self.get_current_obj_grad()
        self.init_reward = -current_obj
        self.record = Record(self.H, grad_dim=self.num_feature+1)
        self.record.update_obj(current_obj)
        self.record.update_grad(current_grad)
        self.init_state = self.get_state()

    def get_current_obj_grad(self):
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        out = torch.matmul(self.W, self.x) + self.b
        loss = self.loss(out, self.y)
        loss.backward()
        grad = torch.cat([self.W.grad, self.b.grad])
        self.W.requires_grad_(False)
        self.b.requires_grad_(False)
        return loss.detach(), grad

    def get_state(self):
        state = torch.cat([
            torch.cat([self.W, self.b]).float(), # current location
            self.record.delta_objs,
            self.record.grads.view(-1)
        ])
        return state


    def step(self, action):
        '''
        :param action:
        :return: state, reward, done, info
        '''
        action = torch.tensor(action).float().squeeze().cpu()
        self.W = self.W + action[: self.num_feature]
        self.b = self.b + action[self.num_feature: ]
        current_obj, current_grad = self.get_current_obj_grad()
        self.record.update_obj(current_obj)
        self.record.update_grad(current_grad)
        # make current state vec
        state = self.get_state()
        # get current reward
        reward = -current_obj # we want to minimize the object
        # check done or not
        done = (current_obj <= self.end_obj)
        # info, init as None
        info = None

        self.T += 1

        return state, reward, done, info


    def generate(self):
        """
        generate problem instances
        """

        y = torch.randn((self.N)).float()
        x = torch.randn((self.num_feature, self.N)).float()

        self.x = x
        self.y = y


class NNCE():
    def __init__(self, num_feature, N, H, robust=True):
        """
        y = W x + b, W \in R^{num_features}, b \in R
        min_{W, b}
        """
        self.H = H
        self.N = N
        self.num_feature = num_feature
        self.problem_name = 'NNCE'
        self.action_dim = 2*(num_feature**2 + num_feature)
        self.state_dim = (1+H)*self.action_dim + H * 1
        self.max_step = 100
        self.end_obj = 0.7
        self.loss = torch.nn.CrossEntropyLoss()
        self.relu = torch.nn.ReLU()
        self.generate()
        self.reset()



    def reset(self):
        self.T = 0
        self.W = torch.randn((self.num_feature, self.num_feature))
        self.U = torch.randn((self.num_feature, self.num_feature))
        self.b = torch.randn((self.num_feature))
        self.c = torch.randn((self.num_feature))

        current_obj, current_grad = self.get_current_obj_grad()
        self.init_reward = -current_obj
        self.record = Record(self.H, grad_dim=self.action_dim)
        self.record.update_obj(current_obj)
        self.record.update_grad(current_grad)
        self.init_state = self.get_state()

    def get_current_obj_grad(self):
        self.W.requires_grad_(True)
        self.U.requires_grad_(True)
        self.b.requires_grad_(True)
        self.c.requires_grad_(True)


        out = torch.matmul(self.relu(torch.matmul(self.x, self.W) + self.b), self.U) + self.c
        loss = self.loss(out, self.y)
        loss += (self.W.norm(p='fro') * 0.00025)
        loss += (self.U.norm(p='fro') * 0.00025)
        loss.backward()
        grad = torch.cat([self.W.grad.view(-1, 1), self.U.grad.view(-1, 1), self.b.grad.view(-1, 1), self.c.grad.view(-1, 1)])

        self.W.requires_grad_(False)
        self.U.requires_grad_(False)
        self.b.requires_grad_(False)
        self.c.requires_grad_(False)

        return loss.detach(), grad

    def get_state(self):
        state = torch.cat([
            torch.cat([self.W.view(-1, 1), self.U.view(-1, 1), self.b.view(-1, 1), self.c.view(-1, 1)]).view(-1).float(),
            self.record.delta_objs,
            self.record.grads.view(-1),
        ])
        return state


    def step(self, action):
        '''
        :param action:
        :return: state, reward, done, info
        '''
        action = torch.tensor(action).float().squeeze().cpu()
        self.W = self.W + action[: self.num_feature**2].view_as(self.W)
        self.U = self.U + action[self.num_feature**2: 2*self.num_feature**2].view_as(self.U)
        self.b = self.b + action[2*self.num_feature**2: 2*self.num_feature**2+self.num_feature].view_as(self.b)
        self.c = self.c + action[2*self.num_feature**2+self.num_feature:].view_as(self.b)

        current_obj, current_grad = self.get_current_obj_grad()
        self.record.update_obj(current_obj)
        self.record.update_grad(current_grad)
        # make current state vec
        state = self.get_state()
        # get current reward
        reward = -current_obj # we want to minimize the object
        # check done or not
        done = (current_obj <= self.end_obj)
        # info, init as None
        info = None

        self.T += 1

        return state, reward, done, info


    def generate(self):
        """
        generate problem instances
        """

        y = torch.randint(low=0, high=self.num_feature, size=(self.N,))
        x = torch.randn((self.N, self.num_feature)).float()

        self.x = x
        self.y = y


class NNCENN(nn.Module):
    def __init__(self, W, U, b, c):
        super(NNCENN, self).__init__()
        self.W = nn.Parameter(W.float(), requires_grad=True)
        self.U = nn.Parameter(U.float(), requires_grad=True)
        self.b = nn.Parameter(b.float(), requires_grad=True)
        self.c = nn.Parameter(c.float(), requires_grad=True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.relu = torch.nn.ReLU()

    def forward(self, x, y):
        out = torch.matmul(self.relu(torch.matmul(x, self.W) + self.b), self.U) + self.c
        loss = self.loss(out, y)
        loss += (self.W.norm(p='fro') * 0.00025)
        loss += (self.U.norm(p='fro') * 0.00025)
        return loss





