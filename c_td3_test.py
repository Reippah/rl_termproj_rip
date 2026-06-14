import os
import torch
from a_actor_and_twin_q_critic import MODEL_DIR, Actor

def test(env, actor: Actor, num_episodes: int) -> None:
    for i in range(num_episodes):
        episode_reward = 0
        observation, _ = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            action = actor.get_action(observation, exploration=False)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            observation = next_observation
            done = terminated or truncated
        print("[EPISODE: {0}] EPISODE_STEPS: {1:4d}, EPISODE REWARD: {2:6.1f}".format(
            i, episode_steps, episode_reward))

def main_play(num_episodes: int, env_name: str) -> None:
    from envs.quanser_env import QuanserEnv

    env = QuanserEnv(max_steps=2000, verbose_reset=False)

    n_features = env.observation_space.shape[0]   # 5
    n_actions = env.action_space.shape[0]         # 1

    actor = Actor(n_features=n_features, n_actions=n_actions)
    model_path = os.path.join(MODEL_DIR, "td3_QuanserQube_2162.7_2026-06-14_20-56-34_best.pth")
    actor.load_state_dict(torch.load(model_path, weights_only=True))
    actor.eval()
    print("[TEST] loaded:", os.path.basename(model_path))

    try:
        test(env, actor, num_episodes=num_episodes)
    finally:
        env.close()

if __name__ == "__main__":
    NUM_EPISODES = 5
    ENV_NAME = "QuanserQube"
    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
