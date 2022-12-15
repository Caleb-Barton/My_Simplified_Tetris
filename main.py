from env_cb import Env_CB
from Agent import Agent
from Memory import Memory
import numpy as np
import time
import pickle


def runIt():
    max_iteration = 5000
    logging_iteration = 100
    learning = []
    losses = []

    polyomino = 3
    wellWidth = 4
    wellHeight = 7

    environment = Env_CB(polyomino, wellWidth, wellHeight)
    agent = Agent(environment)
    memory = Memory(max_size=1000)

    for iteration in range(1, max_iteration + 1):
        steps = 0
        done = False
        state = environment.reset()

        while not done:
            if iteration % logging_iteration == 0:
                action = agent.actWell(state)
                next_state, reward, done, *_ = environment.step(action)
                environment.render()
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
            print(f"{np.mean(learning[-logging_iteration:]):.4f}")

runIt()