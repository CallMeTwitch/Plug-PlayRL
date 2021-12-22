# Imports
from torch.nn import Module, Linear, LayerNorm
from torch.nn.functional import relu, mse_loss
from matplotlib import pyplot as plt
from torch.optim import Adam
import numpy as np
import torch
import gym

# Hyperparameters & Variables
CRITIC_LEARNING_RATE = 1e-3
ACTOR_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
HIDDEN_SIZE1 = 400
HIDDEN_SIZE2 = 300
NUM_GAMES = 1e3
BATCH_SIZE = 64
MAX_SIZE = 1e6
GAMMA = 0.99
TAU = 1e-3

# Critic Class
class Critic(Module):
    def __init__(self, input_shape, action_size):
        super(Critic, self).__init__()
        
        self.layer1 = Linear(*input_shape, HIDDEN_SIZE1)
        self.layer2 = Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)

        self.normal_layer1 = LayerNorm(HIDDEN_SIZE1)
        self.normal_layer2 = LayerNorm(HIDDEN_SIZE2)

        self.action_value = Linear(action_size, HIDDEN_SIZE2)

        self.value = Linear(HIDDEN_SIZE2, 1)

        self.optimizer = Adam(self.parameters(), lr = CRITIC_LEARNING_RATE, weight_decay = WEIGHT_DECAY)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')

        self.to(self.device)
    
    # Forward Pass
    def forward(self, state, action):
        layer1_output = self.layer1(state)
        normal_layer1_output = relu(self.normal_layer1(layer1_output))

        layer2_output = self.layer2(normal_layer1_output)
        normal_layer2_output = self.normal_layer2(layer2_output)

        action_value = self.action_value(action)

        return self.value(relu(torch.add(normal_layer2_output, action_value)))

# Actor Class
class Actor(Module):
    def __init__(self, input_shape, action_size):
        super(Actor, self).__init__()

        self.layer1 = Linear(*input_shape, HIDDEN_SIZE1)
        self.layer2 = Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)

        self.normal_layer1 = LayerNorm(HIDDEN_SIZE1)
        self.normal_layer2 = LayerNorm(HIDDEN_SIZE2)

        self.mu = Linear(HIDDEN_SIZE2, action_size)

        self.optimizer = Adam(self.parameters(), lr = ACTOR_LEARNING_RATE)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    # Forward Pass
    def forward(self, state):
        layer1_output = self.layer1(state)
        normal_layer1_output = relu(self.normal_layer1(layer1_output))

        layer2_output = self.layer2(normal_layer1_output)
        normal_layer2_output = relu(self.normal_layer2(layer2_output))

        return torch.tanh(self.mu(normal_layer2_output))

# Noise Generator Class
class Ornstein_Uhlenbeck:
    def __init__(self, mu, sigma = 0.15, theta = 0.2, dt = 1e-2, x0 = None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    # When Called
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
        self.x_prev = x
        return x

    # Reset Function
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# Memory Class
class ReplayBuffer:
    def __init__(self, input_shape, action_size):
        self.mem_center = 0

        self.next_state_memory = np.zeros((MAX_SIZE, *input_shape))
        self.terminal_memory = np.zeros(MAX_SIZE, dtype = np.bool) 
        self.state_memory = np.zeros((MAX_SIZE, *input_shape))
        self.action_memory = np.zeros((MAX_SIZE, action_size))
        self.reward_memory = np.zeros(MAX_SIZE)

    # Memorize
    def store(self, state, action, reward, next_state, done):
        index = self.mem_center % MAX_SIZE

        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.state_memory[index] = state

        self.mem_center += 1
    
    # Retrieve Random
    def sample(self, batch_size):
        max_mem = min(self.mem_center, MAX_SIZE)

        batch = np.random.choice(max_mem, batch_size)

        return self.state_memory[batch], self.action_memory[batch], self.reward_memory[batch], self.next_state_memory[batch], self.terminal_memory[batch]

# DDPG Class
class DDPG:
    def __init__(self, env):
        self.env = env

        self.gamma = 0.99

        self.target_critic = Critic(env.observation_space.shape, env.action_space.shape[0])
        self.target_actor = Actor(env.observation_space.shape, env.action_space.shape[0])
        self.critic = Critic(env.observation_space.shape, env.action_space.shape[0])
        self.actor = Actor(env.observation_space.shape, env.action_space.shape[0])

        self.noise = Ornstein_Uhlenbeck(np.zeros(env.action_space.shape[0]))
        self.memory = ReplayBuffer(env.observation_space.shape, env.action_space.shape[0])

        self.update_network_parameters(tau = 1)

    # Sample Action from Policy
    def choose_action(self, state):
        self.actor.eval()
        state = torch.tensor([state], dtype = torch.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_ = mu + torch.tensor(self.noise(), dtype = torch.float).to(self.actor.device)
        self.actor.train()

        return mu_.cpu().detach().numpy()[0]

    # Memorize
    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    # Backward Pass
    def backward(self):
        if self.memory.mem_center < BATCH_SIZE:
            return

        states, actions, rewards, next_states, done = self.memory.sample(BATCH_SIZE)

        next_states = torch.tensor(next_states, dtype = torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype = torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype = torch.float).to(self.actor.device)
        states = torch.tensor(states, dtype = torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(next_states)
        next_critic_value = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        next_critic_value[done] = 0.0
        next_critic_value = next_critic_value.view(-1)

        target = rewards + self.gamma * next_critic_value
        target = target.view(BATCH_SIZE, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    # Update Target Networks
    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = TAU

        target_critic_params = self.target_critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        critic_params = self.critic.named_parameters()
        actor_params = self.actor.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    # Plot
    def plot_learning_curve(self, x, scores):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.show()

    # Train
    def train(self):
        best_score = self.env.reward_range[0]
        score_history = []
        for i in range(NUM_GAMES):
            observation = self.env.reset()
            done = False
            score = 0
            self.noise.reset()
            while not done:
                action = self.choose_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                self.remember(observation, action, reward, observation_, done)
                self.backward()
                score += reward
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score

            print(f'Episode: {i}, Score: {round(score, 2)}, Average Score: {round(avg_score, 2)}')

        x = [i + 1 for i in range(NUM_GAMES)]
        self.plot_learning_curve(x, score_history)

# Train
env = gym.make('LunarLanderContinuous-v2')
ddpg = DDPG(env)
ddpg.train()