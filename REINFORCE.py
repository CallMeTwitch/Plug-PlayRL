# Imports
from torch.nn.functional import relu, softmax
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from torch.nn import Linear, Module
from torch.optim import Adam
import numpy as np
import torch
import gym

# Hyperparameters & Variables
ACTOR_LEARNING_RATE = 5e-4
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 128
NUM_GAMES = 3e3
GAMMA = 0.99

# Actor Class
class Actor(Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()

        self.layer1 = Linear(input_size, HIDDEN_SIZE1)
        self.layer2 = Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)
        self.layer3 = Linear(HIDDEN_SIZE2, output_size)

        self.optimizer = Adam(self.parameters(), lr = ACTOR_LEARNING_RATE)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Forward Pass
    def forward(self, state):
        layer1_output = relu(self.layer1(state))
        layer2_output = relu(self.layer2(layer1_output))
        layer3_output = self.layer3(layer2_output)

        return layer3_output

    # Back Pass
    def backward(self, rewards, actions):
        self.optimizer.zero_grad()

        G = np.zeros_like(rewards, dtype = np.float64)
        for q in range(len(rewards)):
            G_sum = 0
            discount = 1
            for w in range(q, len(rewards)):
                G_sum += rewards[w] * discount
                discount *= GAMMA
            G[q] = G_sum
        G = torch.tensor(G, dtype = torch.float).to(self.device)

        loss = sum([-g * logprob for g, logprob in zip(G, actions)])
        loss.backward()
        self.optimizer.step()

# REINFORCE Class
class REINFORCE:
    def __init__(self, env):
        self.env = env

        self.reward_mem = []
        self.action_mem = []

        self.actor = Actor(env.observation_space.shape[0], env.action_space.n)

    # Sample Policy for Action
    def choose_action(self, state):
        state = torch.Tensor([state]).to(self.actor.device)
        probabilities = softmax(self.actor.forward(state), dim = 1)
        action_probs = Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_mem.append(log_probs)

        return action.item()

    # Memorize
    def store(self, rewards):
        self.reward_mem.append(rewards)

    # Plot
    def plot_learning_curve(self, x, scores):
        running_avg = np.zeros(len(scores))
        for q in range(len(running_avg)):
            running_avg[q] = np.mean(scores[max(0, q - 100):(q + 1)])
        plt.plot(x, running_avg)
        plt.title('Running Average of Previous 100 Scores')
        plt.show()

    # Train
    def train(self):
        scores = []
        for q in range(NUM_GAMES):
            done = False
            state = self.env.reset()
            score = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.store(reward)
                state = next_state
            self.actor.backward(self.reward_mem, self.action_mem)
            self.reward_mem, self.action_mem = [], []

            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print(f'Episode: {q}, Score: {round(score, 2)}, Average Score: {round(avg_score, 2)}')

        self.plot_learning_curve(scores, [q + 1 for q in range(len(scores))])

# Train
env = gym.make('LunarLander-v2')
reinforce = REINFORCE(env)
reinforce.train()