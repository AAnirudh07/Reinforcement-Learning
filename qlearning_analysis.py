from hashlib import new
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

learning_rate = 0.1
discount = 0.95     #future reward vs. current reward
episodes = 2000
epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

show_episode = 500

observation_size = [20,20]
observation_win_size = (env.observation_space.high - env.observation_space.low) / observation_size

q_table = np.random.uniform(low=-2, high=0, size=(observation_size + [env.action_space.n]))
episode_rewards = []
agg_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / observation_win_size 
    return tuple(discrete_state.astype(np.int64))


for episode in range(episodes):
    ep_reward = 0
    if episode % show_episode == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        ep_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render: 
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q_value = q_table[discrete_state + (action, )]
            new_q = (1 - learning_rate) * current_q_value + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    
    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value
    
    episode_rewards.append(ep_reward)
    if not episode % show_episode:
        average_reward = sum(episode_rewards[-show_episode:]) / len(episode_rewards[-show_episode:])
        agg_rewards['ep'].append(episode)
        agg_rewards['avg'].append(average_reward)
        agg_rewards['min'].append(min(episode_rewards[-show_episode:]))
        agg_rewards['max'].append(max(episode_rewards[-show_episode:]))



env.close()
plt.plot(agg_rewards['ep'], agg_rewards['avg'], label='avg')
plt.plot(agg_rewards['ep'], agg_rewards['min'], label='min')
plt.plot(agg_rewards['ep'], agg_rewards['max'], label='max')
plt.legend(loc=2)
plt.show()