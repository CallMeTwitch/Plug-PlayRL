# Imports
from torch.nn.functional import relu, softmax, smooth_l1_loss
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from torch.nn import Module, Linear
from torch.optim import Adam
import torch
import gym

# Hyperparameters & Variables
UPDATES_PER_EPOCH = 2
LEARNING_RATE = 5e-4
DEPRECIATION = 0.98
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
PRINT_EVERY = 20
EPSILON = 0.05
MAX_STEPS = 20
LAMBDA = 0.95
EPOCHS = 2e3

# PPO Class
class PPO(Module):
    def __init__(self, env):
        super(PPO, self).__init__()

        self.env = env

        self.layer1 = Linear(env.observation_space.shape[0], HIDDEN_SIZE1)
        self.layer2 = Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)

        self.policy_layer = Linear(HIDDEN_SIZE2, env.action_space.n)
        self.value_layer = Linear(HIDDEN_SIZE2, 1)
        
        self.optimizer = Adam(self.parameters(), lr = LEARNING_RATE)

        self.data = []

    # Forward Pass
    def forward(self, input):
        layer1_output = relu(self.layer1(input)).view(-1, 1, HIDDEN_SIZE1)
        layer2_output = self.layer2(layer1_output)

        policy = softmax(self.policy_layer(layer2_output), dim = 2)
        value = self.value_layer(layer2_output)
        return policy, value

    # Memorize Data
    def store(self, data):
        self.data.append(data)

    # Retrieve Data
    def get_data(self):
        states, actions, rewards, next_states, action_probs, dones = [], [], [], [], [], []
        for transition in self.data:
            state, action, reward, next_state, prob_action, done = transition

            action_probs.append([prob_action])
            next_states.append(next_state)
            rewards.append([reward])
            actions.append([action])
            states.append(state)

            dones.append([0 if done else 1])
        
        next_states = torch.tensor(next_states, dtype = torch.float)
        states = torch.tensor(states, dtype = torch.float)
        dones = torch.tensor(dones, dtype = torch.float)
        action_probs = torch.tensor(action_probs)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        self.data = []

        return states, actions, rewards, next_states, dones, action_probs 

    # Back Pass
    def backward(self):
        state, action, reward, next_state, done, prob_action = self.get_data()

        for _ in range(UPDATES_PER_EPOCH):
            next_value = self.forward(next_state)[1].squeeze(1)
            target_reward = reward + DEPRECIATION * next_value * done
            value = self.forward(state)[1].squeeze(1)
            diffs = (target_reward - value).detach().numpy()
            
            advantage = 0
            advantage_lst = []
            for diff in diffs[::-1]:
                advantage = DEPRECIATION * LAMBDA * advantage + diff[0]
                advantage_lst.append([advantage])
            
            advantage_lst.reverse()
            advantage_lst = torch.tensor(advantage_lst, dtype = torch.float)

            policy = self.forward(state)[0]
            policy = policy.squeeze(1).gather(1, action)

            ratio = torch.exp(torch.log(policy) - torch.log(prob_action))

            loss1 = ratio * advantage_lst
            loss2 = torch.clamp(ratio, (1 - EPSILON), (1 + EPSILON)) * advantage_lst
            final_loss = -torch.min(loss1, loss2) + smooth_l1_loss(value, target_reward.detach())

            self.optimizer.zero_grad()
            final_loss.mean().backward(retain_graph = True)
            self.optimizer.step()

    # Train
    def train(self):
        rewards = []
        score = 0

        for epoch in range(EPOCHS):
            state = self.env.reset()
            done = False

            while not done:
                for _ in range(MAX_STEPS):
                    policy = self.forward(torch.from_numpy(state).float())[0]
                    categorical = Categorical(policy)
                    policy = policy.view(-1)

                    action = categorical.sample().item()

                    next_state, reward, done, _ = self.env.step(action)

                    self.store((state, action, reward / 100, next_state, policy[action].item(), done))

                    state = next_state

                    score += reward

                    if done:
                        break

                self.backward()
            
            if epoch % PRINT_EVERY == 0 and epoch != 0:
                print(f'Epoch: {epoch}, Average Reward: {round(score / PRINT_EVERY, 2)}')
                rewards.append(score / PRINT_EVERY)
                score = 0

            self.env.close

        plt.plot(rewards, color = 'black')
        plt.show()

# Train
env = gym.make('CartPole-v1')
ppo = PPO(env)
ppo.train()