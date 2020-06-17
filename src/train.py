from src.model import PPO, Memory
from src.problem import LinearRegression, LinearRegressionNN, NNCE, NNCENN
from torch import nn
import torch
import numpy as np

def L2O(args):
    # random_seed = None
    #############################################

    # creating environment
    if args.problem == 'LR':
        problem = LinearRegression(num_feature=4, N=100, H=15)

        ############## Hyperparameters LR ##############
        log_interval = 20  # print avg reward in the interval
        max_episodes = 500  # max training episodes
        max_timesteps = 40  # max timesteps in one episode
        update_timestep = 100  # update policy every n timesteps
        max_problems = 10

        action_std = 0.5  # constant std for action distribution (Multivariate Normal)
        K_epochs = 5  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.9  # discount factor

        lr = 0.001  # parameters for Adam optimizer
        betas = (0.9, 0.999)

    if args.problem == 'NNCE':
        problem = NNCE(num_feature=2, N=100, H=15)

        ############## Hyperparameters LR ##############
        log_interval = 20  # print avg reward in the interval
        max_episodes = 2000  # max training episodes
        max_timesteps = 100  # max timesteps in one episode
        update_timestep = 200  # update policy every n timesteps
        max_problems = 1

        action_std = 0.5  # constant std for action distribution (Multivariate Normal)
        K_epochs = 5  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.8  # discount factor

        lr = 0.001  # parameters for Adam optimizer
        betas = (0.9, 0.999)


    state_dim = problem.state_dim
    action_dim = problem.action_dim

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    for i_problems in range(1, max_problems + 1):

        problem.generate()
        problem.reset()

        # logging variables
        running_reward = 0
        time_step = 0
        init_rewards = []
        last_rewards = []
        n_done = 0.9

        # training loop
        for i_episode in range(1, max_episodes + 1):

            problem.reset()
            state = problem.init_state
            has_done = False

            for t in range(max_timesteps):
                time_step += 1
                # Running policy_old:
                action = ppo.select_action(state, memory)
                if t == 0:
                    init_rewards.append(problem.init_reward)

                state, reward, done, _ = problem.step(action)

                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if time_step % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    time_step = 0

                running_reward += reward

                if done:
                    has_done = True

            if has_done:
                n_done += 1

            last_rewards.append(reward)

            # logging
            if i_episode % log_interval == 0:
                done_rate = n_done / log_interval
                running_reward = int((running_reward / log_interval))
                n_done = 0
                print('Problem {}: {}\t Episode {} \t Done rate: {} \t Avg reward: {}, Avg last reward: {}, Max last reward: {}, Avg init reward: {}'.format(args.problem, i_problems, i_episode, done_rate, running_reward, np.mean(last_rewards), np.max(last_rewards), np.mean(init_rewards) ))
                running_reward = 0
                init_rewards = []
                last_rewards = []

                evaluate_ppo(ppo, problem, max_timesteps, memory)


def evaluate_ppo(ppo, problem, max_timesteps, memory):
    np.random.seed(0)
    torch.manual_seed(0)
    problem.generate()
    problem.reset()

    state = problem.init_state
    losses = []
    reward = None

    for t in range(max_timesteps):
        # Running policy_old:
        action = ppo.select_action(state, memory)

        state, reward, done, _ = problem.step(action)

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        losses.append(reward.detach().numpy())

    print('Evaluate | last reward: {}, all losses: {}'.format(reward, list(np.array(losses))))

def Baselines(args):

    np.random.seed(0)
    torch.manual_seed(0)

    if args.problem == 'LR':
        problem = LinearRegression(num_feature=4, N=100, H=15)
        problem.generate()
        problem.reset()
        model = LinearRegressionNN(problem.W, problem.b)

    elif args.problem == 'NNCE':
        problem = NNCE(num_feature=2, N=100, H=15)
        problem.generate()
        problem.reset()
        model = NNCENN(problem.W, problem.U, problem.b, problem.c)


    max_timesteps = 100  # max timesteps in one episode

    assert args.method in ['SGD', 'Adam', 'Momentum', 'LBFGS']
    lr = 1  # parameters for Adam optimizer
    if args.method == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.method == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif args.method == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters())

    losses = []

    for t in range(1, max_timesteps+1):

        if args.method in ['SGD', 'Momentum', 'Adam']:
            optimizer.zero_grad()
            loss = model(problem.x, problem.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())


        else:
            def closure():
                optimizer.zero_grad()
                loss = model(problem.x, problem.y)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure()
            losses.append(loss.detach().numpy())

    print('final loss: {}'.format(loss))
    print(list(np.array(losses)))

