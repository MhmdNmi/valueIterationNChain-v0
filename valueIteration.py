import numpy as np
# import matplotlib.pyplot as plt
import gym
import random

# HYPERPARAMETERS
number_of_states = 25
slip_chance = 0.2
small_reward = 5
large_reward = 20
gamma = 0.9  #Discounting rate

env = gym.make("NChain-v0", slip=slip_chance, n=number_of_states, small=small_reward, large=large_reward)

V = np.zeros(number_of_states)
Vn = np.zeros(number_of_states)

# Training
diff = 1.5
V[0] = small_reward
V[number_of_states - 1] = large_reward

for i in range(number_of_states):
    Vn[i] = V[i]
#it = 0
while diff > 0.05:
    for i in range(1, number_of_states - 1):
        ns = min(i + 1, number_of_states - 1)
        Qf = (1 - slip_chance) * V[ns] + slip_chance * V[0]
        Qb = slip_chance * V[ns] + (1 - slip_chance) * V[0]
        if Qf >= Qb:
            Vn[i] = gamma * Qf
        else:
            Vn[i] = gamma * Qb

    diff = 0
    for j in range(number_of_states):
        diff += abs(V[j] - Vn[j])
#    print(V, Vn)
#    it += 1
#    print(it, diff)
    for j in range(1, number_of_states - 1):
        V[j] = Vn[j]

#print(diff)
print(V)

# Testing
test_steps = 10000
state = env.reset()
cumulative_rewards = 0

for step in range(test_steps):
    next_state = min(state + 1, number_of_states - 1)
    forward_value = (1 - slip_chance) * V[next_state] + slip_chance * V[0]
    backward_value = slip_chance * V[next_state] + (1 - slip_chance) * V[0]
    if forward_value > backward_value:
        action = 0
    else:
        action = 1

    new_state, reward, done, info = env.step(action)
#    print(state, action, new_state, reward)
    cumulative_rewards += reward
    state = new_state

print("Cumulative reward: {}".format(cumulative_rewards))
print("Average reward: {}".format(cumulative_rewards / test_steps))
