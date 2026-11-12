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

from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    seed_queue: mp.Queue,
):
    parent_remote.close()
    env_fn = env_fn_wrapper.x
    isaac_env, sim_app = env_fn()
    try:
        while True:
            try:
                cmd = child_remote.recv()
            except EOFError:
                child_remote.close()
                break
            if cmd == "reset":
                obs_queue.put(isaac_env.reset(seed=seed_queue.get()))
            elif cmd == "step":
                input_action, auto_reset = action_queue.get()
                obs, reward, terminated, truncated, extras = isaac_env.step(
                    input_action, auto_reset=auto_reset
                )
                # IsaacLab keeps the final observations of an automatic reset in
                # its extras, so they are sent with the step of that reset only.
                final_obs = extras.pop("final_obs", None)
                obs_queue.put(
                    (
                        obs,
                        reward,
                        terminated,
                        truncated,
                        {**extras, "final_obs": final_obs},
                    )
                )
            elif cmd == "close":
                isaac_env.close()
                child_remote.close()
                sim_app.close()
                break
            elif cmd == "device":
                child_remote.send(isaac_env.device)
            else:
                child_remote.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        child_remote.close()
    finally:
        try:
            isaac_env.close()
        except Exception as e:
            print(f"IsaacLab Env Closed with error: {e}")


class SubProcIsaacLabEnv:
    def __init__(self, env_fn):
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.action_queue = ctx.Queue()
        self.obs_queue = ctx.Queue()
        self.seed_queue = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.obs_queue,
            self.seed_queue,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        self.isaac_lab_process.start()
        self.child_remote.close()

    def reset(self, seed=None):
        self.parent_remote.send("reset")
        self.seed_queue.put(seed)
        obs, info = self.obs_queue.get()
        return obs, info

    def step(self, action: torch.Tensor, auto_reset: bool = True):
        """
        action : (bs, action_dim)
        auto_reset : whether IsaacLab resets, in this step, the environments
            whose episodes ended since their last reset. The extras of the step
            then hold their observations from before the reset as ``final_obs``,
            if the environment computes them, and ``None`` otherwise.
        """
        self.parent_remote.send("step")
        self.action_queue.put((action, auto_reset))
        env_step_result = self.obs_queue.get()
        return env_step_result

    def close(self):
        self.parent_remote.send("close")
        self.isaac_lab_process.join()
        self.isaac_lab_process.terminate()

    def device(self):
        self.parent_remote.send("device")
        return self.parent_remote.recv()
