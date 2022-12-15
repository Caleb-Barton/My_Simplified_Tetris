import gym
import gym_simplifiedtetris
import numpy as np


class Env_CB:
    def __init__(self, minoSize=4, wellWidth=10, wellHeight=20, reward=0):
        self.currStep = 0
        self.wellWidth = wellWidth
        self.wellHeight = wellHeight
        self.env = gym.make(f"simplifiedtetris-binary-{wellHeight}x{wellWidth}-{minoSize}-v0")
        self.coveredBlocks = 0
        self.observation_space = np.array([0.0] * (self.wellWidth + 1))
        self.action_space = self.env.action_space
        self.reward = reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward == 1:
            newCoveredBlocks = self.checkForCoveredSpaces(obs)
            self.checkHeight(obs)
            if newCoveredBlocks != self.coveredBlocks:
                rewardAdjust = (self.coveredBlocks - newCoveredBlocks) /4  # NegNum if more than before,
                self.coveredBlocks = newCoveredBlocks
                reward += rewardAdjust
        return self.simplifyObs(obs), reward, done, info

    def simplifyObs(self, obsCmplx):
        newObs = [np.float32(self.wellHeight)] * (self.wellWidth + 1)  # First element will be the new piece
        newObs[-1] = np.float32(obsCmplx[-1])
        for col in range(self.wellWidth):
            row = 0
            while row < self.wellHeight:
                if obsCmplx[col * self.wellHeight + row] != 0:
                    newObs[col] = np.float32(row) # Each column will be labeled with the number of row that had an element
                    break
                else:
                    row += 1
        self.observation_space = np.array(newObs)
        return np.array(newObs)

    def checkForCoveredSpaces(self, obs) -> int:
        coveredSpaceCount = 0
        for col in range(self.wellWidth):
            blockFound = False
            for space in range(self.wellHeight):
                if obs[col * self.wellHeight + space] > 0:
                    blockFound = True
                elif blockFound:
                    coveredSpaceCount += 1
        return coveredSpaceCount

    def checkHeight(self, obs) -> int:
        topBlock = 0
        for col in range(self.wellWidth):
            for space in range(self.wellHeight):
                if obs[col * self.wellHeight + space] > 0:
                    height = self.wellHeight - space
                    if (height > topBlock):
                        topBlock = height
        # print(f"Height is {topBlock}")
        return topBlock

    def blockCount(self, obs) -> int:
        count = 0
        for col in range(self.wellWidth):
            for space in range(self.wellHeight):
                if obs[col * self.wellHeight + space] > 0:
                    count += 1
        return count

    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        self.simplifyObs(obs)
        return self.observation_space

    def close(self):
        self.env.close()
