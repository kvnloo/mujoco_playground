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

"""Regression tests for configured and simulated timestep agreement."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import numpy as np

from mujoco_playground import registry


class TimestepTest(parameterized.TestCase):
  """Verify both model representations and elapsed environment time."""

  @parameterized.named_parameters(
      ("ball_default", "BallInCup", None),
      ("ball_finer", "BallInCup", 0.001),
      ("ball_coarser", "BallInCup", 0.004),
      ("barkour_default", "BarkourJoystick", None),
      ("barkour_xml_matching", "BarkourJoystick", 0.002),
      ("barkour_finer", "BarkourJoystick", 0.001),
  )
  def test_timestep_reaches_models_and_simulation(self, env_name, sim_dt):
    overrides = {"impl": "jax"}
    if sim_dt is not None:
      overrides["sim_dt"] = sim_dt
    env = registry.load(env_name, config_overrides=overrides)

    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    initial_time = float(state.data.time)
    step = jax.jit(env.step)
    action = jp.zeros(env.action_size)
    elapsed = []
    for count in range(1, 3):
      state = step(state, action)
      state.data.time.block_until_ready()
      elapsed.append(float(state.data.time) - initial_time)
    print("CLOCK_RECEIPT " + json.dumps({
        "env": env_name, "override": sim_dt,
        "configured_dt": env.sim_dt,
        "host_dt": env.mj_model.opt.timestep,
        "compiled_dt": float(env.mjx_model.opt.timestep),
        "substeps": env.n_substeps, "control_dt": env.dt,
        "elapsed": elapsed,
    }), flush=True)
    self.assertAlmostEqual(env.mj_model.opt.timestep, env.sim_dt)
    np.testing.assert_allclose(
        env.mjx_model.opt.timestep, env.sim_dt, rtol=1e-6, atol=1e-8
    )
    # These cases have integral ratios. Rounding policy is separate work.
    self.assertAlmostEqual(env.n_substeps * env.sim_dt, env.dt)

    for count, actual in enumerate(elapsed, 1):
      np.testing.assert_allclose(
          actual, count * env.dt, rtol=1e-5, atol=1e-7
      )



if __name__ == "__main__":
  absltest.main()
