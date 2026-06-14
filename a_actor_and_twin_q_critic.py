import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    # Quanser state 5차원, action 1차원 기본값
    def __init__(self, n_features: int = 5, n_actions: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_actions)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_v = torch.tanh(self.out(x))   # 출력 [-1, 1] (QuanserEnv가 ×ACTION_TO_PWM)
        return mu_v

    def get_action(self, x: torch.Tensor, exploration: bool = True,
                   noise_scale: float = 0.1) -> np.ndarray:
        # 추론 시 그래프 생성 방지 (100Hz 제어 루프 부담↓)
        with torch.no_grad():
            mu_v = self.forward(x)
        action = mu_v.detach().cpu().numpy()

        if exploration:
            # Pendulum 의 scale=1.0 은 Quanser 엔 과함 -> noise_scale(기본 0.1)
            noises = np.random.normal(loc=0.0, scale=noise_scale, size=self.n_actions)
            action = action + noises

        action = np.clip(action, a_min=-1.0, a_max=1.0)
        return action


class TwinQCritic(nn.Module):
    def __init__(self, n_features: int = 5, n_actions: int = 1, hidden_dim: int = 256):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(n_features, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim + n_actions, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_fc1 = nn.Linear(n_features, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim + n_actions, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
        self.to(DEVICE)

    def forward(self, x, action) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        q1 = F.relu(self.q1_fc1(x))
        q1 = torch.cat([q1, action], dim=-1)
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = torch.cat([q2, action], dim=-1)
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        return q1, q2

    def q1_value(self, x, action) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        q1 = F.relu(self.q1_fc1(x))
        q1 = torch.cat([q1, action], dim=-1)
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        return q1


Transition = collections.namedtuple(
    typename="Transition", field_names=["observation", "action", "next_observation", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self) -> Transition:
        return self.buffer.pop()

    def clear(self) -> None:
        self.buffer.clear()

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        observations, actions, next_observations, rewards, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )

        observations = np.array(observations)
        next_observations = np.array(next_observations)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)

        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        # ★ 연속 행동이므로 float32 (int64면 [-1,1]이 -1/0/1로 잘려 학습 붕괴) ★
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
