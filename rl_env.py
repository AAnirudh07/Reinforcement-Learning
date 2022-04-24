from re import S
from cv2 import waitKey
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time


style.use("ggplot")

SIZE = 10
NO_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
EPSILON_DECAY = 0.998
SHOW_EPISODES = 3000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

epsilon = 0.9
start_q_table = None

d = {1: (255,175,0), 2:(0,255,0), 3:(0,0,255)}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def action(self, choice):
        if choice==0:
            self.move(x=1,y=1)
        elif choice==1:
            self.move(x=-1,y=-1)
        elif choice==2:
            self.move(x=-1,y=1)
        elif choice==3:
            self.move(x=1,y=-1)
        
    def move(self,x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1
        

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1,y1),(x2,y2))] = [np.random.uniform(-5,0) for i in range(4)]
        
else:
    with open(start_q_table, "rb") as f:
        pickle.load(f)

episode_rewards = []
for episode in range(NO_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EPISODES == 0:
        print(f"Episode No.{episode} epsilon:{epsilon}")
        print(f"{SHOW_EPISODES} episode mean:{np.mean(episode_rewards[-SHOW_EPISODES:])}")
        show = True
    else:
        show = False
    
    episode_reward = 0
    
    for i in range(200):
        observation = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[observation])
        else:
            action = np.random.randint(0,4)
        
        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        new_observation = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_observation])
        current_q = q_table[observation][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[observation][action] = new_q

        if show:
            env = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300),resample=Image.BOX)
            cv2.imshow("",np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            
            else:
                if waitKey(1) & 0xFF == ord("q"):
                    break
        
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
    
    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EPISODES,)) / SHOW_EPISODES, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.show()
