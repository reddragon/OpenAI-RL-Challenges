import gym
import logging
import sys
import numpy as np
from gym import wrappers

SEED = 0
NUM_EPISODES = 3000

# Hyperparams
LEARNING_RATE = 0.75
GAMMA = 0.99
NOISE = 2.0

env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, 'frozen-lake', force=True)
if SEED >= 0:
    env.seed(SEED)
    np.random.seed(SEED)

episode = 0
non_zero_rewards = 0

# Initialize the Q-table with one row per state, and one column per action.
check = -1
Q = np.zeros((16, 4))
while episode < NUM_EPISODES:
    new_state = env.reset()
    done = False
    path_len = 0
    while not done:
        # env.render()
        path_len = path_len + 1
        prev_state = new_state

        if Q[prev_state].sum() > 0.0:
            action = np.argmax(Q[prev_state] + np.random.randn(
                1, env.action_space.n) * (NOISE / (episode + 1)))
        else:
            action = np.random.choice(4)

        new_state, reward, done, _ = env.step(action)

        if episode == check:
            print(
                'Episode: {} Prev State was: {}, Action taken was: {}, next state was: {}, Path Len: {}'.
                format(episode, prev_state, action, new_state, path_len))

        allow = False
        if allow or prev_state != new_state:
            prev_val = Q[prev_state][action]
            next_max = np.max(Q[new_state, :])
            after_val = prev_val + LEARNING_RATE * \
                (reward + GAMMA * next_max - prev_val)
            Q[prev_state][action] = after_val

            if episode == check:
                print(
                    "Prev_State:{} Next_State:{} Prev_Val: {} Next_Max: {} After_Val: {}, Reward: {}".
                    format(prev_state, new_state, prev_val, next_max,
                           after_val, reward))

        if reward != 0:
            # print("----> Reward wasnt zero in Episode {} at state: {}, from state: {}, taking: {}  <----".format(episode, new_state, prev_state, action))
            non_zero_rewards = non_zero_rewards + 1

    episode = episode + 1

print('Final State:')
print(Q)
print('Non Zero Rewards: {}, Ratio: {}'.format(
    non_zero_rewards, non_zero_rewards * 1.0 / NUM_EPISODES))
env.close()
