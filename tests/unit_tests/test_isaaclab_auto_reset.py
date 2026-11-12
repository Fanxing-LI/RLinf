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

import torch
from omegaconf import OmegaConf

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv


class _FakeSimulation:
    """Stands in for the simulation process of two environments.

    The first environment ends ``ends_at`` steps after its reset and keeps saying so, as IsaacLab does until it
    resets the environment; every step pays both environments 1.
    """

    def __init__(self, ends_at: int, final_obs: bool):
        self.ends_at, self.final_obs = ends_at, final_obs
        self.steps, self.auto_resets = 0, []

    def device(self) -> str:
        return "cpu"

    def reset(self, seed=None):
        self.steps = 0
        return {"state": "start"}, {}

    def step(self, action, auto_reset):
        self.steps += 1
        self.auto_resets.append(auto_reset)
        terminated = torch.tensor([self.steps >= self.ends_at, False])
        final_obs = None
        if auto_reset and terminated.any():
            self.steps = 0
            final_obs = {"state": "last"} if self.final_obs else None
        obs = {"state": "start" if self.steps == 0 else "running"}
        extras = {"final_obs": final_obs}
        return obs, torch.ones(2), terminated, torch.zeros(2, dtype=torch.bool), extras


class _Env(IsaaclabBaseEnv):
    def __init__(self, simulation: _FakeSimulation, **overrides):
        self._simulation = simulation
        cfg = {
            "seed": 0,
            "auto_reset": True,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "max_episode_steps": 100,
            "video_cfg": {"save_video": False},
            "init_params": {"id": "Fake", "task_description": ""},
        }
        super().__init__(OmegaConf.create({**cfg, **overrides}), 2, 0, 1, None)

    def _init_isaaclab_env(self):
        self.env = self._simulation

    def _wrap_obs(self, obs):
        return obs


def test_only_the_last_step_of_a_chunk_resets():
    for auto_reset, expected in ((True, [False, False, True]), (False, [False] * 3)):
        simulation = _FakeSimulation(ends_at=10, final_obs=True)
        _Env(simulation, auto_reset=auto_reset).chunk_step(torch.zeros(2, 3, 1))

        assert simulation.auto_resets == expected


def test_episode_ends_once_and_its_final_observation_comes_with_the_reset():
    env = _Env(_FakeSimulation(ends_at=2, final_obs=True))
    obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
        torch.zeros(2, 4, 1)
    )

    # The first environment ends in the second step: its later steps count for nothing.
    assert rewards.tolist() == [[1.0, 1.0, 0.0, 0.0], [1.0] * 4]
    assert terminations.tolist() == [[False, False, False, True], [False] * 4]
    assert not truncations.any()
    # IsaacLab reset it in the last step, which returns the new episode and the old one's end.
    infos = infos_list[-1]
    assert obs_list[-1] == {"state": "start"}
    assert infos["final_observation"] == {"state": "last"}
    assert infos["final_info"]["episode"]["return"].tolist() == [2.0, 4.0]
    assert infos["_final_info"].tolist() == [True, False]
    assert (env.returns.tolist(), env.elapsed_steps.tolist()) == ([0.0, 4.0], [0, 4])

    # The new episode counts again.
    rewards = env.chunk_step(torch.zeros(2, 1, 1))[1]
    assert rewards.tolist() == [[1.0], [1.0]]


def test_environment_at_the_step_limit_goes_on_from_its_state():
    env = _Env(_FakeSimulation(ends_at=10, final_obs=True), max_episode_steps=2)
    obs_list, _, _, truncations, infos_list = env.chunk_step(torch.zeros(2, 2, 1))

    # IsaacLab did not end the episodes, so it reset nothing: the final observation is the current one.
    assert truncations.tolist() == [[False, True], [False, True]]
    assert infos_list[-1]["final_observation"] is obs_list[-1]
    assert env.elapsed_steps.tolist() == [0, 0]
