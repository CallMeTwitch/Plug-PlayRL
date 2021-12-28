# Imports
from torch.nn.functional import relu, mse_loss
from torch.nn import Module, Linear
from torch.optim import Adam
import numpy as np
import torch
import gym

# Settings
gym.logger.set_level(40)

# Hyperparameters & Variables
CRITIC_LEARNING_RATE = 0.001
ACTOR_LEARNING_RATE = 0.001
MEM_SIZE = 1_000_000
HIDDEN_SIZE1 = 400
HIDDEN_SIZE2 = 300
UPDATE_ACTOR = 2
BATCH_SIZE = 100
NUM_GAMES = 1_000
WARMUP = 1_000
GAMMA = 0.99
NOISE = 0.1
TAU = 0.005

# Memory Class
class ReplayBuffer:
    def __init__(self, state_shape, action_size):
        self.mem_cntr = 0

        self.next_state_mem = np.zeros((MEM_SIZE, *state_shape))
        self.terminal_mem = np.zeros(MEM_SIZE, dtype = np.bool)
        self.state_mem = np.zeros((MEM_SIZE, *state_shape))
        self.action_mem = np.zeros((MEM_SIZE, action_size))
        self.reward_mem = np.zeros(MEM_SIZE)

    # Memorize
    def store(self, state, action, reward, next_state, done):
        index = self.mem_cntr % MEM_SIZE

        self.next_state_mem[index] = next_state
        self.terminal_mem[index] = done
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.state_mem[index] = state

        self.mem_cntr += 1

    # Retrieve from Memory
    def sample(self):
        max_mem = min(self.mem_cntr, MEM_SIZE)

        batch = np.random.choice(max_mem, BATCH_SIZE)

        next_states = self.next_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]
        states = self.state_mem[batch]

        return states, actions, rewards, next_states, dones

# Critic Class
class Critic(Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        
        self.layer1 = Linear(state_size + action_size, HIDDEN_SIZE1)
        self.layer2 = Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)
        self.layer3 = Linear(HIDDEN_SIZE2, 1)

        self.optimizer = Adam(self.parameters(), lr = CRITIC_LEARNING_RATE)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Forward Pass
    def forward(self, state, action):
        layer1_output = relu(self.layer1(torch.cat([state, action], dim = 1)))
        layer2_output = relu(self.layer2(layer1_output))
        layer3_output = self.layer3(layer2_output)
        
        return layer3_output

# Actor Class
class Actor(Module):
    def __init__(self, input_shape, output_size):
        super(Actor, self).__init__()

        self.layer1 = Linear(*input_shape, HIDDEN_SIZE1)
        self.layer2 = Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)
        self.layer3 = Linear(HIDDEN_SIZE2, output_size)

        self.optimizer = Adam(self.parameters(), lr = ACTOR_LEARNING_RATE)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Forward Pass
    def forward(self, state):
        layer1_output = relu(self.layer1(state))
        layer2_output = relu(self.layer2(layer1_output))
        layer3_output = torch.tanh(self.layer3(layer2_output))

        return layer3_output
    
    # Backward Pass
    def backward(self, critic, state):
        self.optimizer.zero_grad()
        value = critic.forward(state, self.forward(state))
        loss = -torch.mean(value)
        loss.backward()
        self.optimizer.step()

# TD3 Class
class TD3:
    def __init__(self, env):
        self.env = env

        self.step_cntr = 0
        self.time_step = 0

        self.actor = Actor(env.observation_space.shape, env.action_space.shape[0])
        self.critic1 = Critic(env.observation_space.shape[0], env.action_space.shape[0])
        self.critic2 = Critic(env.observation_space.shape[0], env.action_space.shape[0])

        self.target_actor = Actor(env.observation_space.shape, env.action_space.shape[0])
        self.target_critic1 = Critic(env.observation_space.shape[0], env.action_space.shape[0])
        self.target_critic2 = Critic(env.observation_space.shape[0], env.action_space.shape[0])

        self.memory = ReplayBuffer(env.observation_space.shape, env.action_space.shape[0])

        self.update_network_parameters(tau = 1)

    # Sample Action from Policy
    def choose(self, state):
        if self.time_step < WARMUP:
            mu = torch.tensor(np.random.normal(scale = NOISE, size = self.env.action_space.shape[0]))
        else:
            state = torch.tensor(state, dtype = torch.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        mu_prime = mu + torch.tensor(np.random.normal(scale = NOISE), dtype = torch.float)
        mu_prime = torch.clamp(mu_prime, self.env.action_space.low[0], self.env.action_space.high[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy() 

    # Memorize
    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    # Critic Backward Pass
    def critic_backward(self, target, value1, value2):
        self.critic1.zero_grad()
        self.critic2.zero_grad()

        loss1 = mse_loss(target, value1)
        loss2 = mse_loss(target, value2)
        loss = loss1 + loss2
        loss.backward()

        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

    # Backward Pass
    def backward(self):
        if self.memory.mem_cntr < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.memory.sample()

        next_state = torch.tensor(next_state, dtype = torch.float).to(self.critic1.device)
        reward = torch.tensor(reward, dtype = torch.float).to(self.critic1.device)
        action = torch.tensor(action, dtype = torch.float).to(self.critic1.device)
        state = torch.tensor(state, dtype = torch.float).to(self.critic1.device)
        done = torch.tensor(done).to(self.critic1.device)

        target_actions = self.target_actor.forward(next_state) + torch.clamp(torch.tensor(np.random.normal(scale = 0.2)), -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.env.action_space.low[0], self.env.action_space.high[0])

        target_value1 = self.target_critic1.forward(next_state, target_actions)
        target_value2 = self.target_critic2.forward(next_state, target_actions)

        target_value1[done], target_value2[done] = 0.0, 0.0

        target_value = torch.min(target_value1.view(-1), target_value2.view(-1))
        target_value = (reward + GAMMA * target_value).view(BATCH_SIZE, 1)

        value1 = self.critic1.forward(state, action)
        value2 = self.critic2.forward(state, action)

        self.critic_backward(target_value, value1, value2)

        self.step_cntr += 1

        if self.step_cntr % UPDATE_ACTOR:
            return
        
        self.actor.backward(self.critic1, state)

        self.update_network_parameters()

    # Update Target Networks
    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = TAU

        target_critic1_parameters = self.target_critic1.named_parameters()
        target_critic2_parameters = self.target_critic2.named_parameters()
        target_actor_parameters = self.target_actor.named_parameters()
        critic1_parameters = self.critic1.named_parameters()
        critic2_parameters = self.critic2.named_parameters()
        actor_parameters = self.actor.named_parameters()

        target_critic1_state_dict = dict(target_critic1_parameters)
        target_critic2_state_dict = dict(target_critic2_parameters)
        target_actor_state_dict = dict(target_actor_parameters)
        critic1_state_dict = dict(critic1_parameters)
        critic2_state_dict = dict(critic2_parameters)
        actor_state_dict = dict(actor_parameters)

        for name in critic1_state_dict:
            critic1_state_dict[name] = tau * critic1_state_dict[name].clone() + (1 - tau) * target_critic1_state_dict[name].clone()

        for name in critic2_state_dict:
            critic2_state_dict[name] = tau * critic2_state_dict[name].clone() + (1 - tau) * target_critic2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic1.load_state_dict(critic1_state_dict)
        self.target_critic2.load_state_dict(critic2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    # Train
    def train(self):
        score_history = []

        for i in range(NUM_GAMES):
            observation = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.choose(observation)
                observation_, reward, done, _ = self.env.step(action)
                self.remember(observation, action, reward, observation_, done)
                self.backward()
                score += reward
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            print(f'Episode: {i}, Score: {round(score, 2)}, Average: {round(avg_score, 2)}')

# Train
env = gym.make('BipedalWalker-v3')
td3 = TD3(env)
td3.train()
