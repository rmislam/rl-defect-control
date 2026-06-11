#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../../src'
import sys
sys.path.insert(1, dir_path + '/../../environments')
sys.path.insert(1, src_path + '/python')
import atexit
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from active_nematic_multibinary_env import ActiveNematicEnv
from param_utils import readParam

SEND_PIPE_PATH = src_path + '/cpp/python_done'
RECV_PIPE_PATH = src_path + '/cpp/c_done'

GEOM_NAME = readParam('GEOM_NAME')
tb_name = 'ppo-active-nematic-multibinary-' + GEOM_NAME

def exit_handler():
    os.remove(SEND_PIPE_PATH)
    os.remove(RECV_PIPE_PATH)
    print('Removed named pipes')

atexit.register(exit_handler)

os.mkfifo(SEND_PIPE_PATH)
os.mkfifo(RECV_PIPE_PATH)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class TensorboardActionLogger(BaseCallback):
    def __init__(self, log_freq=100):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        try:
            obs = self.model._last_obs
        except Exception:
            obs = self.training_env.reset()

        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]

        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor).distribution
            self.logger.record("action_probs/min", dist.probs.min().item())
            self.logger.record("action_probs/mean", dist.probs.mean().item())
            self.logger.record("action_probs/max", dist.probs.max().item())
            self.logger.record("action_entropy/mean", dist.entropy().mean().item())

        return True


if __name__ == '__main__':
    n_steps = 512
    tensorboard_callback = TensorboardActionLogger(log_freq=n_steps)

    checkpoint_callback = CheckpointCallback(
      save_freq=n_steps,
      save_path=dir_path + '/../saved_models/',
      name_prefix=tb_name,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
        net_arch={'pi': [256, 128], 'vf': [256, 128]},
    )

    env = ActiveNematicEnv(SEND_PIPE_PATH, RECV_PIPE_PATH)
    env.initSim()
    model = PPO("CnnPolicy", env, n_steps=n_steps, batch_size=int(n_steps / 2), learning_rate=2.5e-4, ent_coef=0.005, gae_lambda=0.95, clip_range=0.2, n_epochs=10, policy_kwargs=policy_kwargs, device='cpu', verbose=1, tensorboard_log=dir_path + '/../tensorboard_logs/' + tb_name)
    print(model.policy)
    model.learn(total_timesteps=1e6, callback=[checkpoint_callback, tensorboard_callback], tb_log_name=tb_name)
    model.save(tb_name)

    # Test
    state, _ = env.reset()
    total_reward = 0.0

    while True:
        action, _ = model.predict(state, deterministic=True)
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward

        if is_done or is_trunc:
            env.endSim()
            break

    print("Total reward: %.2f" % total_reward)
