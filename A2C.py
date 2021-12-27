# Imports
from torch.nn.functional import softmax, relu
from torch.autograd import Variable
from torch.nn import Linear, Module
from torch.optim import Adam
import numpy as np
import torch
import gym

# Hyperparameters & Variables
LEARNING_RATE = 3e-4
MAX_EPISODES = 3_000
HIDDEN_SIZE = 256
NUM_STEPS = 300
GAMMA = 0.99

# A2C Class
class A2C(Module):
    def __init__(self, env):
        super(A2C, self).__init__()

        self.env = env

        self.actor_layer1 = Linear(env.observation_space.shape[0], HIDDEN_SIZE)
        self.actor_layer2 = Linear(HIDDEN_SIZE, env.action_space.n)

        self.critic_layer1 = Linear(env.observation_space.shape[0], HIDDEN_SIZE)
        self.critic_layer2 = Linear(HIDDEN_SIZE, 1)

        self.optimizer = Adam(self.parameters(), lr = LEARNING_RATE)

    # Forward Pass
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        
        critic_layer1_output = relu(self.critic_layer1(state))
        critic_layer2_output = self.critic_layer2(critic_layer1_output)

        actor_layer1_output = relu(self.actor_layer1(state))
        actor_layer2_output = softmax(self.actor_layer2(actor_layer1_output), dim = 1)

        return critic_layer2_output, actor_layer2_output

    # Back Pass
    def backward(self, values, next_value, log_probabilities, entropy, rewards):
        next_values = np.zeros_like(values)
        for q in reversed(range(len(rewards))):
            next_value = rewards[q] + (GAMMA * next_value)
            next_values[q] = next_value

        values = torch.FloatTensor(values)
        next_values = torch.FloatTensor(next_values)
        log_probabilities = torch.stack(log_probabilities)
        advantage = next_values - values

        actor_loss = (-log_probabilities * advantage).mean()
        critic_loss = advantage.pow(2).mean() / 2
        a2c_loss = actor_loss + critic_loss + 0.001 * entropy

        self.optimizer.zero_grad()
        a2c_loss.backward()
        self.optimizer.step()

    # Train Model
    def train(self):
        entropy = 0
        for episode in range(MAX_EPISODES):
            log_probabilities = []
            rewards = []
            values = []

            state = env.reset()
            for steps in range(NUM_STEPS):
                value, policy = self.forward(state)
                values.append(value.detach().numpy()[0, 0])
                dist = policy.detach().numpy()

                action = np.random.choice(self.env.action_space.n, p = np.squeeze(dist))
                log_probabilities.append(torch.log(policy.squeeze(0)[action]))
                entropy -= np.sum(np.mean(dist) * np.log(dist))
                next_state, reward, done, _ = env.step(action)

                rewards.append(reward)
                state = next_state

                if done or steps == NUM_STEPS - 1:
                    next_value, _ = self.forward(next_state)
                    next_value = next_value.detach().numpy()[0, 0]
                    if episode % 10 == 0:
                        print(f'Episode: {episode}, Reward: {int(sum(rewards))}')
                    break
            
            self.backward(values, next_value, log_probabilities, entropy, rewards)

# Train
env = gym.make("CartPole-v0")
a2c = A2C(env)
a2c.train()
