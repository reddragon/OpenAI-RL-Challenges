import gym
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cPickle as pickle
import os

from gym import wrappers

from torch.autograd import Variable

SEED = 1234
NUM_EPISODES = 20000

# Hyperparams
GAMMA = 0.99
REWARD_CONST = 3.0
LEARNING_RATE = 1e-2

ENV_NAME = 'FrozenLake-v0'
ENV_INTERNAL_NAME = 'frozen-lake-nn'
CHECKPOINT_FILE_PATH = '{}-ckpt'.format(ENV_INTERNAL_NAME)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x


def save(net, optimizer, epoch):
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    print("Saving checkpoint to file '{}'".format(CHECKPOINT_FILE_PATH))
    torch.save(state, CHECKPOINT_FILE_PATH)


# Returns net, optimizer, epoch
def load():
    net = Net()
    # optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, momentum=0.5)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    epoch = 0

    if os.path.isfile(CHECKPOINT_FILE_PATH):
        print("Loading checkpoint from file '{}'".format(CHECKPOINT_FILE_PATH))
        checkpoint = torch.load(CHECKPOINT_FILE_PATH)
        epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, epoch


# Boiler plate to get a gym object
def get_gym(record=False):
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    outdir = '{}-data'.format(ENV_INTERNAL_NAME)
    env = gym.make(ENV_NAME)
    if record:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    return env


# Return a one-hot vector with the idx bit turned on
def get_oh_vector(idx):
    state_arr = np.zeros((1, 16))
    state_arr[0][idx] = 1
    state_var = Variable(torch.Tensor(state_arr))
    return state_var


env = get_gym(record=True)
if SEED >= 0:
    env.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

episode = 0
non_zero_rewards = 0
rewards = []
print_stats = True
verbose = False

net, optimizer, _ = load()
pause_training = False

while episode < NUM_EPISODES and not pause_training:
    new_state = env.reset()
    done = False
    path_len = 0
    while not done:
        optimizer.zero_grad()
        prev_state = new_state
        state_var = get_oh_vector(prev_state)

        output_var = net(state_var)
        output_arr = output_var.data.numpy()

        if np.random.rand() < 1.0 / ((episode / 200.0) + 5.0):
            action = env.action_space.sample()
        else:
            action = np.argmax(output_arr)

        if verbose:
            print '\n== Episode {}, State: {} =='.format(episode, prev_state)
            print 'state vector: ', state_var.data.numpy()
            print 'output_arr: ', output_arr
            print 'action picked: {}'.format(action)

        path_len = path_len + 1
        new_state, reward, done, _ = env.step(action)
        new_state_var = get_oh_vector(new_state)
        new_output_var = net(new_state_var)
        new_output_arr = new_output_var.data.numpy()
        expected_q_val = np.max(new_output_arr) * GAMMA + reward * REWARD_CONST

        target_arr = np.copy(output_arr)
        target_arr[0][action] = expected_q_val
        target_var = Variable(torch.Tensor(target_arr))

        if verbose:
            print 'New State: {}, Reward: {}'.format(new_state, reward)
            print 'New State vec: ', new_state_var.data.numpy()
            print 'New Output Var: ', new_output_arr
            print 'argmax: ', np.max(new_output_arr)
            print 'expected_q_val: ', expected_q_val
            print 'target_arr: ', target_arr

        loss = ((target_var - output_var)**2).sum()
        loss.backward()
        optimizer.step()

        if verbose:
            print("Episode: {}, Loss: {}".format(episode,
                                                 loss.data.numpy()[0]))
            check_var = net(state_var)
            print 'After backprop output: ', check_var.data.numpy()

        if reward != 0:
            non_zero_rewards = non_zero_rewards + 1
            # pause_training = True
            # break

        if done:
            rewards.append(reward)
            last_hundred_epochs = rewards[-100:]
            success = sum(last_hundred_epochs) * 1.0 / len(last_hundred_epochs)
            overall_success = sum(rewards) * 1.0 / len(rewards)
            if print_stats and episode % 10 == 0:
                print(
                    "Episode: {}, Success Ratio in Last 100 Epochs: {}, Overall Ratio: {}".
                    format(episode, success, overall_success))
    episode = episode + 1

    if verbose:
        print '\n\n'

print('Non Zero Rewards: {}, Ratio: {}'.format(
    non_zero_rewards, non_zero_rewards * 1.0 / NUM_EPISODES))
env.close()
