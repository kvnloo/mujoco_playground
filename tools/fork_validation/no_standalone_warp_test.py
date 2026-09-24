"""Real-package gate. No standalone mujoco-warp installation is permitted.

Checks model conversion and all tree leaves; it does not run Warp trajectories.
"""
import importlib.util

import jax
import mujoco
from mujoco import mjx
from mujoco.mjx.warp import mujoco_warp as backend
import numpy as np

from mujoco_playground._src import mjx_env


def test_native_backend_without_standalone_and_only_warning_changes():
    assert importlib.util.find_spec("mujoco_warp") is None, (
        "Use a clean environment without standalone mujoco-warp for this gate."
    )
    assert backend is not None
    assert hasattr(backend, "OverflowType"), "This gate requires the modern overflow-enum backend."
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.003" iterations="17" ls_iterations="5"/>'
        '<worldbody><body><joint type="slide"/>'
        '<geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>'
    )
    before = mjx.put_model(model, impl="warp")
    after = mjx_env.put_model(model, impl="warp")
    target = int(backend.OverflowType.ITERATIONS | backend.OverflowType.LS_ITERATIONS)
    raw_bits = int(before.opt._impl.warn_overflow)
    assert raw_bits & target == target, "Control must exercise both warning bits."
    expected_bits = raw_bits & ~target
    assert int(after.opt._impl.warn_overflow) == expected_bits
    normalized = before.tree_replace({"opt._impl.warn_overflow": after.opt._impl.warn_overflow})
    expected, expected_tree = jax.tree_util.tree_flatten(normalized)
    actual, actual_tree = jax.tree_util.tree_flatten(after)
    assert actual_tree == expected_tree
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
