import numpy as np
import torch
from src.utils import Record

class LinearRegression():
    def __init__(self, num_feature, N, H):
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
        self.end_obj = 1
        self.loss = torch.nn.MSELoss()
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
        action = torch.tensor(action).float()
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
        done = (current_obj <= self.end_obj) or (self.T >= self.max_step)
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









