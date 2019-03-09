import gym
import pandas as pd 
import numpy as np 
from PG_PILCO import PolicyGradient
import matplotlib.pyplot as plt 
from gym.spaces import Discrete, Box

display_reward_threshold = 2000
render = False
learning_rate = 0.02
reward_decay = 0.995
max_episode = 1000
max_episode_step = 3000

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

if isinstance(env.observation_space, Discrete):
    S_DIM = env.observation_space.n
elif isinstance(env.observation_space, Box):
    S_DIM = env.observation_space.shape[0]

if isinstance(env.action_space, Discrete):
    A_DIM = env.action_space.n
elif isinstance(env.action_space, Box):
    A_DIM = env.action_space.shape[0]

policy = PolicyGradient(n_actions=A_DIM, n_features=S_DIM, learning_rate=learning_rate, reward_decay=reward_decay)

total_step = 0
all_ep_r = []
total_step_per_episode = []

for i_episode in range(max_episode):
	observation = env.reset()
	step = 0

	while True:
		if render:
			env.render()

		action = policy.choose_action(observation)

		observation_next, reward, done, info = env.step(action)

		policy.store_transition(observation, action, reward)

		step += 1

		if done or  step >= max_episode_step:
			total_step += step
			ep_rs_sum = sum(policy.ep_rs)
			# if 'running_reward' not in globals():
			# 	running_reward = ep_rs_sum
			# else:
			# 	running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
			if ep_rs_sum > display_reward_threshold:
				#render = True
				render = False #never render

			print("Episode: ", i_episode, " step in this episode:", step, " total step:", total_step, " reward: ", int(ep_rs_sum))

			vt = policy.learn()   # vt is the discounted reward for every time step
			vt = vt[0]            # the first element means the discounted reward for the total episode
			all_ep_r.append(ep_rs_sum)  # what we need is the reward (non discounted)
			total_step_per_episode.append(total_step)
			break

		# update observation
		observation = observation_next 

policy.save_model('./policy.pth')

print("Finished")
plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.show()

# to do 
# plot return for each step: xlabel: totalstep
plt.plot(total_step_per_episode, all_ep_r)
plt.xlabel('Total step')
plt.ylabel('Episode reward')
plt.show()

dataframe = pd.DataFrame({'reward': all_ep_r})
dataframe.to_csv("episode_reward.csv", index=True, sep=',')

dataframe2 = pd.DataFrame({'total_step': total_step_per_episode, 'reward': all_ep_r})
dataframe2.to_csv("step_reward.csv", index=False, sep=',')


