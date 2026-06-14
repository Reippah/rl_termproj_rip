"""
b_td3_train.py
  - 에피소드 사이 일괄 업데이트 (제어 루프 100Hz 보호; 루프 중 학습 X)
  - warmup(learning_starts) 동안 랜덤 행동으로 탐색
  - validation 5회, best 이어받기(파일명 보상 역산), Ctrl+C 시 interrupt 저장
"""
import os
import re
import time
from datetime import datetime
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from a_actor_and_twin_q_critic import MODEL_DIR, Actor, TwinQCritic, ReplayBuffer, Transition, DEVICE


class TD3:
    def __init__(self, env, test_env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="TD3_{0}".format(self.env_name),
                                    name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.validation_time_steps_interval = config["validation_time_steps_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]       # warmup 스텝(랜덤 행동)
        self.utd_ratio = config["utd_ratio"]                   # 환경 스텝당 업데이트 수
        self.exploration_noise = config["exploration_noise"]   # 행동 수집 시 가우시안 노이즈

        # TD3 고유
        self.policy_update_delay = config["policy_update_delay"]
        self.target_policy_noise = config["target_policy_noise"]
        self.target_policy_noise_clip = config["target_policy_noise_clip"]

        n_features = env.observation_space.shape[0]   # 5
        n_actions = env.action_space.shape[0]         # 1

        self.actor = Actor(n_features=n_features, n_actions=n_actions)
        self.target_actor = Actor(n_features=n_features, n_actions=n_actions)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.twin_q_critic = TwinQCritic(n_features=n_features, n_actions=n_actions)
        self.target_twin_q_critic = TwinQCritic(n_features=n_features, n_actions=n_actions)
        self.target_twin_q_critic.load_state_dict(self.twin_q_critic.state_dict())
        self.twin_q_critic_optimizer = optim.Adam(self.twin_q_critic.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        self.time_steps = 0
        self.training_time_steps = 0
        self.total_train_start_time = None

        # 최고 검증 성능 추적 + 이전 best 이어받기
        self.best_validation_reward = -float("inf")
        try:
            best_files = [f for f in os.listdir(MODEL_DIR)
                          if f.startswith("td3_{0}_".format(self.env_name)) and f.endswith("_best.pth")]
            nums = []
            for f in best_files:
                m = re.search(r"_(-?\d+\.?\d*)_\d{4}-\d{2}-\d{2}", f)
                if m:
                    try:
                        nums.append(float(m.group(1)))
                    except ValueError:
                        pass
            if nums:
                self.best_validation_reward = max(nums)
                print("[INIT] 이전 best 이어받기: {0:.1f}".format(self.best_validation_reward))
        except FileNotFoundError:
            pass

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()
        policy_loss = critic_loss = mu_v = 0.0
        next_validation_at = max(self.learning_starts, self.validation_time_steps_interval)

        try:
            for n_episode in range(1, self.max_num_episodes + 1):
                episode_reward = 0
                episode_steps = 0
                observation, _ = self.env.reset()
                done = False

                while not done:
                    self.time_steps += 1
                    episode_steps += 1

                    # warmup: 랜덤 행동 / 이후: actor + 탐색 노이즈
                    if self.time_steps <= self.learning_starts:
                        action = self.env.action_space.sample()
                    else:
                        action = self.actor.get_action(
                            observation, exploration=True, noise_scale=self.exploration_noise
                        )

                    next_observation, reward, terminated, truncated, _ = self.env.step(action)
                    episode_reward += reward

                    # done 에는 terminated 만 (truncated=시간만료는 부트스트랩 유지)
                    self.replay_buffer.append(
                        Transition(observation, action, next_observation, reward, terminated)
                    )
                    observation = next_observation
                    done = terminated or truncated

                # 에피소드 끝 -> 일괄 업데이트 (제어 루프 중엔 학습 안 함)
                if self.replay_buffer.size() >= self.batch_size and self.time_steps >= self.learning_starts:
                    n_updates = max(1, int(episode_steps * self.utd_ratio))
                    for _ in range(n_updates):
                        policy_loss, critic_loss, mu_v = self.train()

                # validation
                if self.time_steps >= next_validation_at:
                    next_validation_at += self.validation_time_steps_interval
                    _, validation_reward_avg = self.validate()
                    if validation_reward_avg > self.best_validation_reward:
                        self.best_validation_reward = validation_reward_avg
                        self.model_save(validation_reward_avg, tag="best", update_latest=True)
                        print("  [SAVE] td3_{0}_{1:.1f}_{2}_best.pth  (-> latest)".format(
                            self.env_name, validation_reward_avg, self.current_time))

                if self.use_wandb:
                    self.log_wandb(self.best_validation_reward, episode_reward, episode_steps,
                                   policy_loss, critic_loss, mu_v, n_episode)

                if n_episode % self.print_episode_interval == 0:
                    print("[Epi. {:3,}, Steps {:6,}]".format(n_episode, self.time_steps),
                          "Reward: {:>8.3f},".format(episode_reward),
                          "Policy L.: {:>7.3f},".format(policy_loss),
                          "Critic L.: {:>7.3f},".format(critic_loss),
                          "Train Steps: {:,}".format(self.training_time_steps))

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Ctrl+C 감지 — 현재 정책 저장 중...")
            self.model_save(0.0, tag="interrupt", update_latest=False)
            print("  [SAVE] td3_{0}_0.0_{1}_interrupt.pth".format(self.env_name, self.current_time))

        total = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.total_train_start_time))
        print("Total Training End : {}".format(total))
        if self.use_wandb:
            self.wandb.finish()

    def train(self) -> tuple[float, float, float]:
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # ── CRITIC UPDATE ──
        with torch.no_grad():
            # Target Policy Smoothing: 타겟 액션에 클리핑된 노이즈
            noise = (torch.randn_like(actions) * self.target_policy_noise).clamp(
                -self.target_policy_noise_clip, self.target_policy_noise_clip)
            next_actions = (self.target_actor(next_observations) + noise).clamp(-1.0, 1.0)

            # Twin Critics: 두 Q 최솟값 (과대평가 방지)
            next_q1, next_q2 = self.target_twin_q_critic(next_observations, next_actions)
            next_q_values = torch.min(next_q1, next_q2).squeeze(dim=-1)
            next_q_values[dones] = 0.0
            target_values = rewards.squeeze(dim=-1) + self.gamma * next_q_values

        q1_values, q2_values = self.twin_q_critic(observations, actions)
        q1_values = q1_values.squeeze(dim=-1)
        q2_values = q2_values.squeeze(dim=-1)

        critic_loss = F.mse_loss(target_values, q1_values) + F.mse_loss(target_values, q2_values)
        self.twin_q_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.twin_q_critic_optimizer.step()

        # DELAYED POLICY UPDATE
        actor_loss = torch.tensor(0.0)
        mu_v = torch.tensor(0.0)
        if self.training_time_steps % self.policy_update_delay == 0:
            mu_v = self.actor(observations)
            q_v = self.twin_q_critic.q1_value(observations, mu_v)
            actor_loss = -1.0 * q_v.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_synchronize_models(self.actor, self.target_actor, self.soft_update_tau)
            self.soft_synchronize_models(self.twin_q_critic, self.target_twin_q_critic, self.soft_update_tau)

        return actor_loss.item(), critic_loss.item(), mu_v.mean().item()

    def soft_synchronize_models(self, source_model, target_model, tau):
        src = source_model.state_dict()
        tgt = target_model.state_dict()
        for k, v in src.items():
            tgt[k] = tau * tgt[k] + (1.0 - tau) * v
        target_model.load_state_dict(tgt)

    def model_save(self, validation_reward_avg: float, tag: str = "best",
                   update_latest: bool = True) -> None:
        filename = "td3_{0}_{1:.1f}_{2}_{3}.pth".format(
            self.env_name, validation_reward_avg, self.current_time, tag)
        torch.save(self.actor.state_dict(), os.path.join(MODEL_DIR, filename))
        if update_latest:
            copyfile(src=os.path.join(MODEL_DIR, filename),
                     dst=os.path.join(MODEL_DIR, "td3_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)
        for i in range(self.validation_num_episodes):
            episode_reward = 0
            observation, _ = self.test_env.reset()
            done = False
            while not done:
                action = self.actor.get_action(observation, exploration=False)   # 결정론적
                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)
                episode_reward += reward
                observation = next_observation
                done = terminated or truncated
            episode_reward_lst[i] = episode_reward

        avg = float(np.average(episode_reward_lst))
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.total_train_start_time))
        print("[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
            episode_reward_lst, avg, elapsed))
        return episode_reward_lst, avg

    def log_wandb(self, validation_reward_avg, episode_reward, episode_steps,
                  policy_loss, critic_loss, mu_v, n_episode):
        self.wandb.log({
            "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(self.validation_num_episodes): validation_reward_avg,
            "[TRAIN] Episode Reward": episode_reward,
            "[TRAIN] Episode Steps": episode_steps,
            "[TRAIN] Policy Loss": policy_loss,
            "[TRAIN] Critic Loss": critic_loss,
            "[TRAIN] mu_v": mu_v,
            "[TRAIN] Replay Buffer": self.replay_buffer.size(),
            "Training Episode": n_episode,
            "Training Steps": self.training_time_steps,
            "Time Steps": self.time_steps,
        })

def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    from envs.quanser_env import QuanserEnv

    env_name = "QuanserQube"
    env = QuanserEnv(max_steps=800, verbose_reset=False)
    test_env = env   # 같은 하드웨어 공유

    config = {
        "env_name": env_name,
        "max_num_episodes": 200_000,
        "batch_size": 256,
        "replay_buffer_size": 1_000_000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.995,
        "print_episode_interval": 20,
        "validation_time_steps_interval": 10_000,
        "validation_num_episodes": 5,
        "episode_reward_avg_solved": 99_999,     # 자동 종료 안 함
        "learning_starts": 5_000,                # warmup(랜덤 행동) 스텝
        "utd_ratio": 1.0,                        # 환경 스텝당 업데이트 수
        "exploration_noise": 0.3,                # 행동 수집 가우시안 노이즈
        "policy_update_delay": 2,
        "target_policy_noise": 0.2,
        "target_policy_noise_clip": 0.5,
    }

    use_wandb = True
    td3 = TD3(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    try:
        td3.train_loop()
    finally:
        env.close()

if __name__ == "__main__":
    main()
