import torch
import gym
from src.model import PPO, Memory
from src.problem import LinearRegression
import numpy as np

def main1():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v2"
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    # random_seed = None
    #############################################

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # if random_seed:
    #     print("Random Seed: {}".format(random_seed))
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)
    #     np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

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
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0




def main():
    ############## Hyperparameters ##############
    log_interval = 50  # print avg reward in the interval
    max_episodes = 1000  # max training episodes
    max_timesteps = 40  # max timesteps in one episode
    update_timestep = 50  # update policy every n timesteps
    max_problems = 10

    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 20  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.5  # discount factor

    lr = 0.0001  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    # random_seed = None
    #############################################

    # creating environment
    problem = LinearRegression(num_feature=4, N=100, H=15)
    state_dim = problem.state_dim
    action_dim = problem.action_dim

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    for i_problems in range(1, max_problems + 1):
        problem.generate()
        problem.reset()

        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0
        init_rewards = []
        last_rewards = []
        # training loop
        for i_episode in range(1, max_episodes + 1):

            problem.reset()
            state = problem.init_state


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
                    break

            avg_length += t
            last_rewards.append(reward)

            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length / log_interval)
                running_reward = int((running_reward / log_interval))
                print('Problem: {}\t Episode {} \t Avg length: {} \t Avg reward: {}, Avg last reward: {}, Avg init reward: {}'.format(i_problems, i_episode, avg_length, running_reward, np.mean(last_rewards), np.mean(init_rewards) ))
                running_reward = 0
                avg_length = 0
                init_rewards = []
                last_rewards = []


if __name__ == '__main__':
    main()