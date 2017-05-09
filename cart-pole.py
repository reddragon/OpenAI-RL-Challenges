import gym
import logging
import sys
import numpy as np
from gym import wrappers

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import cPickle as pickle
import os

from math import sqrt, ceil

from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = (self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.sigmoid(x)

def get_action(net, input):
    prob = net(input).data.numpy()[0][0]
    x = np.random.uniform()
    if x > prob:
        return 0, prob
    return 1, prob

def save(net, optimizer, epoch):
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    print ("Saving checkpoint to file '{}'" . format(CHECKPOINT_FILE_PATH))
    torch.save(state, CHECKPOINT_FILE_PATH)

CHECKPOINT_FILE_PATH = 'rl_ckpt'
NUM_EPISODES = 300
MIN_ITERS = 100
LEARNING_RATE = 0.002
GAMMA = 0.99
REWARD_THRESHOLD = 195.0
REWARD_AVG_WINDOW_LEN = 100.0

# Returns net, optimizer, epoch
def load():
    net = Net()
    # optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, momentum=0.5)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    epoch = 0

    if os.path.isfile(CHECKPOINT_FILE_PATH):
        print ("Loading checkpoint from file '{}'" . format(CHECKPOINT_FILE_PATH))
        checkpoint = torch.load(CHECKPOINT_FILE_PATH)
        epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, epoch

# Boiler plate to get a gym object
def get_gym(record=False, outdir='rl-data'):
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    outdir = 'rl-data'
    env = gym.make('CartPole-v0')
    if record:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    return env

def discounted_rewards(r):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r
    # return list(reversed([y*(GAMMA**idx) for idx,y in enumerate(reversed(rewards))]))

env = get_gym()
env = get_gym(record=True)
net, optimizer, episode = load()

episode = 0
steps_in_episode = np.array([])
training_done = False

while episode < NUM_EPISODES and not training_done:
    obs = env.reset()
    running_reward = 0
    rewards = []
    actions = []
    obs_inps = []
    outs = []
    num_steps = 0
    while True:
        num_steps = num_steps + 1
        obs_np = np.expand_dims(np.array(obs), axis=0)
        input_var = Variable(torch.Tensor(obs_np))
        action, prob = get_action(net, input_var)
        obs_inps.append(obs)
        outs.append(prob)
        obs, reward, done, _ = env.step(action)
        running_reward += reward
        rewards.append(running_reward)
        actions.append(action)
        if done:
            disc_rewards = np.array(discounted_rewards(rewards))
            steps = len(actions)
            actions_var = Variable(torch.Tensor(actions))
            rewards_var = Variable(torch.Tensor(disc_rewards))

            optimizer.zero_grad()
            obs_inps = np.array(obs_inps)
            input_var = Variable(torch.Tensor(obs_inps))
            outs_var = net(input_var)
            outs = np.array(outs).reshape(-1, 1)
            loss =\
                -(
                    disc_rewards *
                    (
                        (1 - actions_var) * torch.log(1 - outs_var) +
                        (actions_var) * torch.log(outs_var)
                    )
                ).sum() * 1.0 / steps

            print(
                "Epoch: {}, Loss: {}, Len: {}"
                .format(episode, loss.data.numpy()[0], num_steps)
            )

            loss.backward()
            optimizer.step()

            if steps_in_episode.shape[0] >= REWARD_AVG_WINDOW_LEN:
                steps_in_episode = steps_in_episode[1:]

            steps_in_episode = np.append(steps_in_episode, steps)
            avg = np.average(steps_in_episode)
            print 'Running Reward:', avg

            if avg > REWARD_THRESHOLD and steps_in_episode.shape[0] == REWARD_AVG_WINDOW_LEN:
                print 'Done early because we reached an average of', avg
                training_done = True

            num_steps = 0
            break

    if episode % 25 == 0:
        save(net, optimizer, episode)
    episode = episode + 1
