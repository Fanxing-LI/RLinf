# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import gymnasium as gym
import torch
from omegaconf import open_dict

from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv


class IsaaclabBaseEnv(gym.Env):
    """
    Class for isaaclab in rlinf. Different from other lab enviromnent, the output of isaaclab is all tensor on
    cuda.

    IsaacLab resets its environments itself: with ``auto_reset``, the last step of an action chunk resets the
    environments whose episodes ended in the chunk. An ended environment keeps stepping until then, which does
    not count for its episode. It needs an IsaacLab whose ``step`` takes ``auto_reset``.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        self.cfg = cfg
        self.isaaclab_env_id = self.cfg.init_params.id
        self.num_envs = num_envs

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.video_cfg = cfg.video_cfg
        self._init_isaaclab_env()
        self.device = self.env.device()

        self.task_description = cfg.init_params.task_description
        self._is_start = True  # if this is first time for simulator
        self.auto_reset = cfg.auto_reset
        self.prev_step_reward = torch.zeros(self.num_envs).to(self.device)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32).to(
            self.device
        )
        # Environments whose episodes ended since their last reset.
        self._ended = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.ignore_terminations = cfg.ignore_terminations

    def _make_env_function(self):
        raise NotImplementedError

    def _init_isaaclab_env(self):
        env_fn = self._make_env_function()
        self.env = SubProcIsaacLabEnv(env_fn)
        self.env.reset(seed=self.seed)

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.fail_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.returns = torch.zeros(self.num_envs).to(self.device)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool).to(self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | (step_reward > 0)
        # batch level
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def reset(self, seed: Optional[int] = None):
        obs, _ = self.env.reset(seed=seed)
        infos = {}
        obs = self._wrap_obs(obs)
        self._reset_metrics()
        self._ended[:] = False
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        auto_reset = auto_reset and self.auto_reset
        obs, step_reward, terminations, truncations, infos = self.env.step(
            actions, auto_reset
        )
        final_obs = infos.pop("final_obs")

        obs = self._wrap_obs(obs)

        self._elapsed_steps += 1

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations

        # An episode ends once: what its environment does until its reset does not count.
        step_reward = step_reward * ~self._ended
        terminations = terminations & ~self._ended
        truncations = truncations & ~self._ended

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False

        self._ended |= terminations | truncations
        if auto_reset and self._ended.any():
            # IsaacLab has reset the environments whose episodes it ended: their observations are those of
            # the new episodes. An environment that reached max_episode_steps only goes on from its state.
            ended = self._ended.clone()
            infos = {
                "final_observation": obs
                if final_obs is None
                else self._wrap_obs(final_obs),
                "final_info": infos,
                "_final_info": ended,
                "_final_observation": ended,
                "_elapsed_steps": ended,
            }
            self._reset_metrics(ended)
            self._ended[:] = False

        return (
            obs,
            step_reward,
            terminations,
            truncations,
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=i == chunk_size - 1
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations).to(
                self.device
            )
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations).to(self.device)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _wrap_obs(self, obs):
        raise NotImplementedError

    def close(self):
        self.env.close()

    def update_reset_state_ids(self):
        """
        No muti task.
        """
        pass

    """
    Below codes are all copied from libero, thanks to the author of libero!
    """

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps.to(self.device)

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward
