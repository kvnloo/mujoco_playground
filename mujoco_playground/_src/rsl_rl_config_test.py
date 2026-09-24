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

"""Tests for RSL-RL configuration compatibility."""

from absl.testing import absltest

from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params


class RslRlConfigTest(absltest.TestCase):

  def _assert_v5_schema(self, cfg):
    self.assertNotIn("policy", cfg)
    self.assertNotIn("empirical_normalization", cfg)
    self.assertEqual(cfg.actor.class_name, "MLPModel")
    self.assertEqual(cfg.critic.class_name, "MLPModel")
    self.assertTrue(cfg.actor.obs_normalization)
    self.assertTrue(cfg.critic.obs_normalization)
    self.assertEqual(
        cfg.actor.distribution_cfg.class_name, "GaussianDistribution"
    )
    self.assertEqual(cfg.actor.distribution_cfg.init_std, 1.0)
    self.assertEqual(cfg.actor.distribution_cfg.std_type, "scalar")

  def test_locomotion_config_uses_rsl_rl_v5_schema(self):
    self._assert_v5_schema(locomotion_params.rsl_rl_config("Go1Getup"))

  def test_manipulation_config_uses_rsl_rl_v5_schema(self):
    self._assert_v5_schema(
        manipulation_params.rsl_rl_config("LeapCubeReorient")
    )


if __name__ == "__main__":
  absltest.main()
