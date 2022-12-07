import gym
import gym_simplifiedtetris
import numpy as np
from gym import spaces


class Env_CB:
    def __init__(self, minoSize=4, wellWidth=10, wellHeight=20, maxSteps=500):
        self.maxSteps = maxSteps
        self.currStep = 0
        self.wellWidth = wellWidth
        self.wellHeight = wellHeight
        self.env = gym.make(f"simplifiedtetris-binary-{wellHeight}x{wellWidth}-{minoSize}-v0")
        self.coveredBlocks = 0

        # This should probably be somewhere else...
        self.__NumOfPiecesList = [1, 1, 1, 2, 7, 18, 60, 196, 704]  # 0 blocks, 1 piece, 3 blocks, 7 pieces
        self.NumOfPieces = self.__NumOfPiecesList[minoSize]

        self.observation_space = np.array([0] * (self.wellWidth + self.NumOfPieces))
        print("My os? ",self.observation_space)
        print("Their os? ",self.env.observation_space)

        # I should make the action space an array?
        self.action_space = spaces.Tuple((
            eval("spaces.Discrete(2)," * self.env.action_space.n)   # .n is the number of options in the action space. Neat
        ))
        print(self.action_space)
        # self.action_space = self.env.action_space
        print(self.env.action_space)

    def step(self, action):
        # There will be an error here
        obs, reward, done, info = self.env.step(action)
        newCoveredBlocks = self.checkForCoveredSpaces(obs)
        self.checkHeight(obs)
        if newCoveredBlocks != self.coveredBlocks:
            rewardAdjust = (self.coveredBlocks - newCoveredBlocks) / 10  # NegNum if more than before,
            self.coveredBlocks = newCoveredBlocks
            reward += rewardAdjust
        return self.simplifyObs(obs), reward, done, info

    def simplifyObs(self, obsCmplx):
        newObs = [np.int32(self.wellHeight)] * (self.wellWidth + self.NumOfPieces)
        for i in range(- self.NumOfPieces, 0): # The last describes the piece
            newObs[i] = 0
        currPiece = obsCmplx[-1] + 1   # +1 because it starts at zero, and I can't do [-0] (Used to be currPiece = (np.float32.apply(int))(obsCmplx[-1]) + 1)
        newObs[- currPiece] = 1  # Gosh I hope this works.
        for col in range(self.wellWidth):
            row = 0
            while row < self.wellHeight:
                if obsCmplx[col * self.wellHeight + row] != 0:
                    newObs[col] = np.int32(row) # Each column will be labeled with the number of row that had an element
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
