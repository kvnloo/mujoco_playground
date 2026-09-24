"""Fork-only red/green checks against real MuJoCo, never mocks or skips."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

BASE = "4057c147714b6ac09b395377f1a1724bbeacc4d3"
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "validation-results"
CLOCK_PREFIX = "test_timestep_reaches_models_and_simulation_"
CLOCK_NAMES = {CLOCK_PREFIX + n for n in (
    "ball_default", "ball_finer", "ball_coarser", "barkour_default",
    "barkour_xml_matching", "barkour_finer",
)}
CLOCK_BAD = {CLOCK_PREFIX + n for n in (
    "ball_finer", "ball_coarser", "barkour_default", "barkour_finer",
)}
WARP_NAMES = {"test_warn_overflow_filtered_warp", "test_put_model_warn_overflow_filtered_warp"}
WARP_CONTROL = "test_native_backend_without_standalone_and_only_warning_changes"


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def verify_report(path: Path, names: set[str], expected_failures: set[str], mode: str) -> dict:
    nodes = list(ET.parse(path).getroot().iter("testcase"))
    observed = [c.get("name") for c in nodes]
    if len(observed) != len(names) or set(observed) != names:
        raise RuntimeError(f"Wrong/missing/duplicate tests: {observed}; expected {sorted(names)}")
    failed = set()
    for case in nodes:
        name = case.get("name")
        if case.find("skipped") is not None or case.find("error") is not None:
            raise RuntimeError(f"Skipped or errored case: {name}")
        failure = case.find("failure")
        if failure is not None:
            failed.add(name)
            detail = ET.tostring(failure, encoding="unicode")
            expected_type = "ModuleNotFoundError" if mode == "warp" else "AssertionError"
            if expected_type not in detail:
                raise RuntimeError(f"Unexpected failure mechanism in {name}: {detail}")
            if mode == "warp" and "mujoco_warp" not in detail:
                raise RuntimeError(f"Wrong missing dependency: {detail}")
    if failed != expected_failures:
        raise RuntimeError(f"Failures {sorted(failed)} != expected {sorted(expected_failures)}")
    return {"passed": sorted(names - failed), "expected_failures": sorted(failed), "skipped": 0}


def run_cases(phase: str, mode: str) -> dict:
    if mode == "warp":
        path = "mujoco_playground/_src/dm_control_suite/dm_control_suite_test.py::TestSuite::"
        targets = [path + n for n in sorted(WARP_NAMES)]
        names = WARP_NAMES.copy()
        if phase == "candidate":
            targets.append(str(HERE / "no_standalone_warp_test.py"))
            names.add(WARP_CONTROL)
        expected_failures = WARP_NAMES if phase == "baseline" else set()
    else:
        targets = [str(HERE / "timestep_test.py")]
        names = CLOCK_NAMES
        expected_failures = CLOCK_BAD if phase == "baseline" else set()
    xml = OUT / f"{mode}-{phase}.xml"
    log = OUT / f"{mode}-{phase}.log"
    with log.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-s", *targets, f"--junitxml={xml}"],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=1000, check=False,
        )
    text = log.read_text()
    print(text[-12000:], flush=True)
    expected_code = 1 if expected_failures else 0
    if result.returncode != expected_code:
        raise RuntimeError(f"{phase}: pytest exit {result.returncode}, expected {expected_code}")
    report = verify_report(xml, names, expected_failures, mode)
    report["returncode"] = result.returncode
    report["clock_measurements"] = []
    for line in text.splitlines():
        if "CLOCK_RECEIPT {" in line:
            report["clock_measurements"].append(json.loads(line.split("CLOCK_RECEIPT ", 1)[1]))
    if mode == "clocks" and len(report["clock_measurements"]) != 6:
        raise RuntimeError("Missing actual-clock measurements")
    print(f"{mode} {phase}: {json.dumps(report)}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("warp", "clocks"))
    mode = parser.parse_args().mode
    OUT.mkdir(exist_ok=True)
    report = {"mode": mode, "status": "blocked", "base": BASE, "python": sys.version}
    try:
        # Only validation plumbing may differ from the pinned production source.
        command("git", "diff", "--exit-code", BASE, "HEAD", "--", "mujoco_playground", "pyproject.toml")
        command("git", "diff", "--exit-code")
        report["head"] = command("git", "rev-parse", "HEAD")
        report["packages"] = {n: metadata.version(n) for n in (
            "mujoco", "mujoco-mjx", "warp-lang", "jax", "jaxlib", "pytest", "flax",
        )}
        if report["packages"]["mujoco"] != "3.14.0" or report["packages"]["mujoco-mjx"] != "3.14.0":
            raise RuntimeError("This first native gate is pinned to MuJoCo/MJX 3.14.0")
        if importlib.util.find_spec("mujoco_warp") is not None:
            raise RuntimeError("Standalone mujoco_warp must be absent")
        sys.path.insert(0, str(ROOT))
        import jax
        import mujoco_playground
        if not Path(mujoco_playground.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError("A different Playground checkout was imported")
        if jax.default_backend() != "cpu":
            raise RuntimeError("This gate must use real CPU execution")
        if mode == "clocks":
            from mujoco_playground._src import mjx_env
            mjx_env.ensure_menagerie_exists()
            assets = Path(mjx_env.MENAGERIE_PATH)
            asset_sha = command("git", "-C", str(assets), "rev-parse", "HEAD")
            if asset_sha != mjx_env.MENAGERIE_COMMIT_SHA:
                raise RuntimeError("Wrong Menagerie asset version")
            report["menagerie"] = asset_sha
        report["baseline"] = run_cases("baseline", mode)
        patch = HERE / f"{mode}.patch"
        command("git", "apply", "--check", str(patch))
        command("git", "apply", str(patch))
        (OUT / "candidate.diff").write_text(command("git", "diff") + "\n")
        report["candidate"] = run_cases("candidate", mode)
        report["status"] = "passed"
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
