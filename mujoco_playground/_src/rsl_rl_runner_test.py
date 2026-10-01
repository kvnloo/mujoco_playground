# Copyright 2026 Google LLC
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

"""CPU smoke tests for production configs and the real RSL-RL runner.

The test-local environment avoids physics compilation; these tests do not cover
robot learning quality, the JAX/Torch wrapper, or legacy checkpoint conversion.
Install the learning extra to run them rather than skip them.
"""

import importlib.util
from pathlib import Path
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

_RSL_INSTALLED = importlib.util.find_spec("rsl_rl") is not None
if _RSL_INSTALLED:
  from rsl_rl.runners import OnPolicyRunner
  from tensordict import TensorDict
  import torch


class _RunnerEnv:
  """Test-local tensor environment; not a registered Playground task."""

  def __init__(self, asymmetric):
    self.num_envs = 4
    self.num_actions = 1
    self.max_episode_length = 100
    self.device = "cpu"
    self.cfg = {}
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    self._asymmetric = asymmetric
    self._obs = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16

  def get_observations(self):
    obs = {"state": self._obs}
    if self._asymmetric:
      obs["privileged_state"] = torch.cat(
          [self._obs, self._obs[:, :2]], dim=-1
      )
    return TensorDict(obs, batch_size=[self.num_envs], device=self.device)

  def step(self, actions):
    self.episode_length_buf += 1
    self._obs = self._obs + 0.01 * actions
    rewards = self._obs[:, 0] - actions[:, 0].square()
    dones = torch.zeros(self.num_envs)
    return self.get_observations(), rewards, dones, {"time_outs": dones}


@absltest.skipUnless(_RSL_INSTALLED, "Requires the learning extra (RSL-RL).")
class RslRlRunnerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("locomotion_state", "locomotion", False),
      ("locomotion_privileged", "locomotion", True),
      ("manipulation_state", "manipulation", False),
      ("manipulation_privileged", "manipulation", True),
  )
  def test_train_infer_and_restore_checkpoint(self, suite, asymmetric):
    factory, env_name = (
        (locomotion_params.rsl_rl_config, "Go1Getup")
        if suite == "locomotion"
        else (manipulation_params.rsl_rl_config, "LeapCubeReorient")
    )
    cfg = factory(env_name).to_dict()
    cfg["obs_groups"] = {
        "actor": ["state"],
        "critic": ["privileged_state" if asymmetric else "state"],
    }
    # Shorten the run, not the production model/normalization configuration.
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 1

    with tempfile.TemporaryDirectory() as directory:
      with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        env = _RunnerEnv(asymmetric)
        runner = OnPolicyRunner(env, cfg, directory, device="cpu")
        self.assertEqual(runner.alg.actor.obs_dim, 4)
        self.assertEqual(runner.alg.critic.obs_dim, 6 if asymmetric else 4)
        self.assertTrue(runner.alg.actor.obs_normalization)
        self.assertTrue(runner.alg.critic.obs_normalization)
        before = {
            name: value.detach().clone()
            for name, value in runner.alg.actor.named_parameters()
        }
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in runner.alg.actor.named_parameters()
            )
        )
        policy = runner.get_inference_policy(device="cpu")
        with torch.no_grad():
          actions = policy(env.get_observations())
        self.assertEqual(tuple(actions.shape), (4, 1))
        self.assertTrue(torch.isfinite(actions).all().item())

        models = (runner.alg.actor, runner.alg.critic)
        snapshots = [
            {name: value.clone() for name, value in model.state_dict().items()}
            for model in models
        ]
        checkpoint = str(Path(directory) / "round_trip.pt")
        marker = {"test": "v5-round-trip"}
        runner.save(checkpoint, infos=marker)
        # Alter both weights and normalization buffers to make reload observable.
        with torch.no_grad():
          for model in models:
            for value in model.state_dict().values():
              value.zero_()
        self.assertEqual(runner.load(checkpoint), marker)
        for model, expected in zip(models, snapshots):
          for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


if __name__ == "__main__":
  absltest.main()
