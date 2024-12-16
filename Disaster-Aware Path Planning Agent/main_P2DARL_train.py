import numpy as np
import torch
import torch.nn as nn
import random
import itertools
from collections import deque
from scipy.spatial.distance import euclidean
import heapq


class RoadEnvironment:
    def __init__(self, road_data_file, node_data_file, adjacency_file, direction_file):
        # Load data files
        self.road_data = np.load(road_data_file)  # ID, x, y, disaster_severity
        self.node_data = np.load(node_data_file)  # ID, x, y
        self.adjacency_matrix = np.load(adjacency_file)  # N×N matrix
        self.direction_data = np.load(direction_file)  # ID + 8 directional connections

        self.n_nodes = len(self.node_data)
        self.current_node = None
        self.target_node = None
        self.visited_nodes = None
        self.path = None

    def reset(self):
        # Randomly select start and target nodes
        available_nodes = list(range(self.n_nodes))
        self.current_node = random.choice(available_nodes)
        self.target_node = random.choice([n for n in available_nodes if n != self.current_node])
        self.visited_nodes = {self.current_node}
        self.path = [self.current_node]

        # State: [current_x, current_y, target_x, target_y]
        state = self._get_state()
        return state

    def _get_state(self):
        current_pos = self.node_data[self.current_node][1:3]  # x, y of current node
        target_pos = self.node_data[self.target_node][1:3]  # x, y of target node
        return np.concatenate([current_pos, target_pos])

    def _dijkstra(self, start, end):
        distances = {i: float('infinity') for i in range(self.n_nodes)}
        distances[start] = 0
        pq = [(0, start)]
        previous = {i: None for i in range(self.n_nodes)}

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_node == end:
                break

            if current_distance > distances[current_node]:
                continue

            for next_node in range(self.n_nodes):
                if self.adjacency_matrix[current_node][next_node] > 0:
                    distance = current_distance + self.adjacency_matrix[current_node][next_node]

                    if distance < distances[next_node]:
                        distances[next_node] = distance
                        previous[next_node] = current_node
                        heapq.heappush(pq, (distance, next_node))

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return distances[end], path

    def _calculate_r_distant(self):
        # Calculate shortest path length using Dijkstra
        r_min, _ = self._dijkstra(self.path[0], self.target_node)

        # Calculate current path length
        r_now = 0
        for i in range(len(self.path) - 1):
            r_now += self.adjacency_matrix[self.path[i]][self.path[i + 1]]

        return 1 - r_now / r_min if r_min > 0 else -1

    def _calculate_r_disaster(self):
        if len(self.path) < 2:
            return 0

        # Calculate midpoint of path endpoints
        start_pos = self.node_data[self.path[0]][1:3]
        end_pos = self.node_data[self.path[-1]][1:3]
        center = (start_pos + end_pos) / 2

        # Calculate path length
        path_length = 0
        for i in range(len(self.path) - 1):
            path_length += self.adjacency_matrix[self.path[i]][self.path[i + 1]]

        # Calculate disaster impact
        r_disaster = 0
        for i in range(len(self.road_data)):
            node_pos = self.road_data[i][1:3]
            pop = self.road_data[i][3]  # Using disaster_severity as population
            dist = euclidean(center, node_pos)
            if dist <= path_length:
                r_disaster += pop * (1 - dist / path_length)

        return r_disaster

    def step(self, action):
        # action is 0-7 representing 8 directions
        next_node = self.direction_data[self.current_node][action + 1]

        if next_node == -1:  # Invalid move (dead end)
            return self._get_state(), -5, True, {}

        # Update current position and path
        self.current_node = next_node
        self.visited_nodes.add(next_node)
        self.path.append(next_node)

        # Calculate rewards
        r_distant = self._calculate_r_distant()
        r_disaster = self._calculate_r_disaster()

        # Check if reached target
        done = (self.current_node == self.target_node)

        # Calculate total reward
        reward = 10 * (r_distant - 0.1 * r_disaster)  # α = 10
        if done:
            reward += 10  # Target reward

        # Prevent loops
        if len(self.visited_nodes) > 100:  # Max steps
            done = True

        return self._get_state(), reward, done, {}


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.network(x)

    def act(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self(state_tensor)
            return q_values.argmax().item()


# ReplayMemory class remains unchanged
class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def add_memo(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))

    def sample(self):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state, action, reward, done, next_state = zip(*batch)

        return (
            torch.FloatTensor(state),
            torch.LongTensor(action).unsqueeze(1),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(done).unsqueeze(1),
            torch.FloatTensor(next_state)
        )


class Agent:
    def __init__(self, n_state, n_action, batch_size=64):
        self.online_net = DQN(n_state, n_action)
        self.target_net = DQN(n_state, n_action)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.0001)
        self.memo = ReplayMemory(capacity=100000, batch_size=batch_size)
        self.GAMMA = 0.95


# Training parameters
EPSILON_START = 0.5
EPSILON_END = 0.0
EPSILON_DECAY = 50000
TARGET_UPDATE_FREQUENCY = 300

# Initialize environment and agent
env = RoadEnvironment(
    road_data_file='road_data.npy',
    node_data_file='node_data.npy',
    adjacency_file='adjacency.npy',
    direction_file='direction.npy'
)

n_state = 4  # current_x, current_y, target_x, target_y
n_action = 8  # 8 directions

agent = Agent(n_state=n_state, n_action=n_action)

# Training loop
n_episode = 50000
n_time_step = 100
REWARD_BUFFER = np.zeros(shape=n_episode)

for episode_i in range(n_episode):
    episode_reward = 0
    state = env.reset()

    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i,
                            [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])

        # Action selection
        if random.random() <= epsilon:
            action = random.randint(0, n_action - 1)
        else:
            action = agent.online_net.act(state)

        # Environment step
        next_state, reward, done, _ = env.step(action)
        agent.memo.add_memo(state, action, reward, done, next_state)
        state = next_state
        episode_reward += reward

        if done:
            REWARD_BUFFER[episode_i] = episode_reward
            break

        # Training step
        if len(agent.memo.memory) >= agent.memo.batch_size:
            batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

            # Compute targets
            target_q_values = agent.target_net(batch_s_)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

            # Compute Q-values
            q_values = agent.online_net(batch_s)
            a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

            # Compute loss and optimize
            loss = nn.functional.smooth_l1_loss(a_q_values, targets)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

    # Update target network
    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print(f"Episode: {episode_i}")
        print(f"Average Reward: {np.mean(REWARD_BUFFER[max(0, episode_i - 100):episode_i + 1])}")