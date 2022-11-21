import numpy as np
from QualityNN import QualityNN
import torch


class Agent(object):
    def __init__(self, environment):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = QualityNN(environment.observation_space.shape[0], environment.action_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        self.decay = 0.999
        self.randomness = 1.00
        self.min_randomness = 0.001

    def act(self, state):
        # move the state to a Torch Tensor
        # print(type(state))
        # state = np.array([np.float32(1), np.float32(1), np.float32(1), np.float32(1), np.float32(1)])
        state = torch.from_numpy(state).to(self.device)

        # find the quality of both actions
        # print(state)
        qualities = self.model(state).cpu()
        # print("-")

        # sometimes take a random action
        if np.random.rand() <= self.randomness:
            action = np.random.randint(low=0, high=qualities.size(dim=0))
        else:
            action = torch.argmax(qualities).item()

        # return that action
        return action

    def actWell(self, state):
        # move the state to a Torch Tensor
        # print(type(state))
        # state = np.array([np.float32(1), np.float32(1), np.float32(1), np.float32(1), np.float32(1)])
        state = torch.from_numpy(state).to(self.device)

        # find the quality of both actions
        # print(state)
        qualities = self.model(state).cpu()
        # print("-")

        # sometimes take a random action
        action = torch.argmax(qualities).item()


        # return that action
        return action

    def update(self, memory_batch):
        # unpack our batch and convert to tensors
        states, next_states, actions, rewards = self.unpack_batch(memory_batch)

        # compute what the output is (old expected qualities)
        # Q(S, A)
        old_targets = self.old_targets(states, actions)

        # compute what the output should be (new expected qualities)
        # reward + max_a Q(S', a)
        new_targets = self.new_targets(states, next_states, rewards, actions)

        # compute the difference between old and new estimates
        loss = torch.nn.functional.smooth_l1_loss(old_targets, new_targets)

        # apply difference to the neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for logging
        return loss.item()

    def old_targets(self, states, actions):
        # model[states][action]
        return self.model(states).gather(1, actions)

    def new_targets(self, states, next_states, rewards, actions):
        # reward + max(model[next_state])
        return rewards + torch.amax(self.model(next_states), dim=1, keepdim=True)

    def unpack_batch(self, batch):
        states, next_states, actions, rewards = zip(*batch)

        states = torch.tensor(states).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)

        # unsqueeze(1) makes 2d array. [1, 0, 1, ...] -> [[1], [0], [1], ...]
        # this is required because the first array is for the batch, and
        #   the inner arrays are for the elements
        # the states and next_states are already in this format so we don't
        #   need to do anything to them
        # .long() for the actions because we are using them as array indices
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().unsqueeze(1).to(self.device)

        return states, next_states, actions, rewards

    def update_randomness(self):
        self.randomness *= self.decay
        self.randomness = max(self.randomness, self.min_randomness)