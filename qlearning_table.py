from hashlib import new
import gym
import numpy as np

env = gym.make("MountainCar-v0")

learning_rate = 0.1
discount = 0.95     #future reward vs. current reward
episodes = 25000
epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

show_episode = 2000

observation_size = [20,20]
observation_win_size = (env.observation_space.high - env.observation_space.low) / observation_size

q_table = np.random.uniform(low=-2, high=0, size=(observation_size + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / observation_win_size 
    return tuple(discrete_state.astype(np.int64))


for episode in range(episodes):

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
env.close()