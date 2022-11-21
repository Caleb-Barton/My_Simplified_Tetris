from env_cb import Env_CB
from Agent import Agent
from Memory import Memory
import numpy as np
import time


max_iteration = 15000
logging_iteration = 100
learning = []
losses = []

# environment = Env_CB(4, 10, 20)
environment = Env_CB(3, 4, 7)
agent = Agent(environment)
memory = Memory(max_size=5000)

for iteration in range(1, max_iteration + 1):
    steps = 0
    done = False
    state = environment.reset()

    while not done:
        if iteration % logging_iteration == 0:
            action = agent.actWell(state)
            next_state, reward, done, *_ = environment.step(action)
            environment.render()
            time.sleep(0.25)
        else:
            action = agent.act(state)
            next_state, reward, done, *_ = environment.step(action)

        memory.push(element=(state, next_state, action, reward))

        state = next_state
        steps += 1

    # make a few updates
    for _ in range(64):
        memory_batch = memory.get_batch(batch_size=64)
        loss = agent.update(memory_batch)
    losses.append(loss)
    agent.update_randomness()

    learning.append(steps)
    if iteration % logging_iteration == 0:
        print(f"Iteration: {iteration}")
        print(f"  Moving-Average Steps: {np.mean(learning[-logging_iteration:]):.4f}")
        print(f"  Memory-Buffer Size: {len(memory.memory)}")
        print(f"  Agent Randomness: {agent.randomness:.3f}")
        print()

"""
import gym
import gym_simplifiedtetris
from env_cb import Env_CB

env = Env_CB(3, 4, 7)
obs = env.reset()


# Run 10 games of Tetris, selecting actions uniformly at random.
episode_num = 0
while episode_num < 5:
    env.render()

    # action = env.action_space.sample()
    print(obs)
    action = int(input())
    obs, reward, done, info = env.step(action)
    print(reward)

    if done:
        print(f"Episode {episode_num + 1} has terminated.")
        episode_num += 1
        obs = env.reset()

env.close()
"""