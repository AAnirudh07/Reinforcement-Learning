import gym 
import numpy as np
from stable_baselines import ACER
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

environment_name = 'LunarLander-v2'
env = gym.make(environment_name)


#test the environment
EPISODES = 10
for episode in range(EPISODES):
    initial_state = env.reset()
    done = False
    while not done:
        action = np.random.randint(0,env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        env.render()
env.done()

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = ACER('MlpPolicy', env)
model.learn(total_timesteps=100000)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()