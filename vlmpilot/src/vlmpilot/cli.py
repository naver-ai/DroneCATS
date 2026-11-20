# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""cli.py — process entry point.

Parses arguments, hands them to ``mission``, and turns top-level exceptions into
exit codes. Both ``python -m vlmpilot`` and the ``vlmpilot`` console script land in
``main()`` (R-05).
"""

from __future__ import annotations

import argparse
import signal
import sys
import traceback

import yaml

# mission imports airsim at module load. The CLI must still start on the real-Tello laptop (no airsim
# installed), so airsim is imported lazily, only inside the airsim branch.
# The two constants below duplicate mission's values; keep them in sync.
DEFAULT_CONFIG_PATH = "configs/airsim/single-drone/airsim_gemini.yaml"
TELLO_DEFAULT_CONFIG_PATH = "configs/real/tello_gemini38_or.yaml"   # = controller/real/tello_mission.DEFAULT_CONFIG_PATH
TEST_IMAGE_PATH = "frame_airsim_test.jpg"
MODES = ("airsim", "pegasus", "tello")


def is_multi_config(config_path: str | None) -> bool:
    """True iff the YAML has a non-empty ``drone_names`` list.

    **Any** failure (no path, missing file, broken YAML, non-mapping root) is ``False`` = the
    single path — a config typo must never take single-drone evaluation down with it.
    """
    if not config_path:
        return False
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return False
        names = data.get("drone_names")
        return isinstance(names, (list, tuple)) and len(names) > 0
    except Exception:  # noqa: BLE001 — read failures fall back to single on purpose
        return False


def is_agent_config(config_path: str | None) -> bool:
    """True iff the YAML asks for the head/body multi-agent mode (``agent_mode: head_body``).

    Read the same defensive way as :func:`is_multi_config`: any failure means the key is not
    there, so a broken config falls back to the lockstep team path rather than taking the
    process down. Only consulted once the config is already known to be multi-drone.
    """
    if not config_path:
        return False
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return False
        return str(data.get("agent_mode", "") or "").strip().lower() == "head_body"
    except Exception:  # noqa: BLE001 — same rule as is_multi_config
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmpilot",
        description="VLM-driven closed-loop drone navigation on AirSim, Isaac Sim (Pegasus + PX4 SITL) or a DJI Tello.",
    )
    parser.add_argument("mode", choices=MODES, help="flight mode: 'airsim' (simulator), 'pegasus' (Isaac Sim + PX4 SITL) or 'tello' (real drone)")
    parser.add_argument(
        "--config", type=str, default=None,
        help=(f"config YAML path, relative to cwd (airsim default: {DEFAULT_CONFIG_PATH}; "
              f"tello default: {TELLO_DEFAULT_CONFIG_PATH}; required for pegasus)"),
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="AirSim RPC port; overrides the optional $MY_PORT environment default and the config's airsim_rpc_port (ignored in tello mode)",
    )
    parser.add_argument("--debug", action="store_true", help="extra progress logs around queue waits")
    parser.add_argument(
        "--test", action="store_true",
        help=f"one inference on the static image {TEST_IMAGE_PATH} in cwd, then exit",
    )
    return parser


def _ensure_sigint_is_handled() -> None:
    """The verifier runner stops an episode with SIGINT. A process launched from a
    non-interactive shell in the background (`nohup … &`) inherits SIGINT=SIG_IGN and
    Python then installs no KeyboardInterrupt handler — the runner's SIGINT is silently
    dropped and the drone keeps flying until the watchdog SIGKILLs us (seen in real flight).
    Restore the default handler so SIGINT always reaches the mission loop."""
    try:
        if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
            signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError):  # not main thread / unsupported platform
        pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_sigint_is_handled()
    try:
        if args.mode == "pegasus":
            # Isaac Sim + PX4 SITL path. Decided before --test and multi-drone, which are AirSim-only.
            from vlmpilot.controller.isaacsim import pegasus_mission  # lazy import: keeps the airsim path independent

            try:
                pegasus_mission.require_config_path(args)
            except ValueError as exc:
                print(f"[CLI] {exc}", flush=True)
                return 2
            if args.test:
                print("[CLI] --test is AirSim-only; pegasus ignores it and runs the closed loop", flush=True)
            return pegasus_mission.fly(args)
        if args.mode == "tello":
            # Real-drone path: avoid mission/multi_mission, which import airsim
            # (the real-drone laptop has no airsim package). --test/--port ignored.
            from vlmpilot.controller.real import tello_mission  # lazy import: keeps the airsim path independent

            return tello_mission.fly(args)
        from vlmpilot.controller.airsim import mission  # airsim path only; imports airsim at module load

        if args.test:
            return mission.single_shot(args)  # --test is always the single path
        config_path = args.config or DEFAULT_CONFIG_PATH
        if is_multi_config(config_path):
            if is_agent_config(config_path):
                from vlmpilot.controller.airsim import agent_mission  # head/body mode: N body threads + one head

                return agent_mission.fly_agents(args)
            from vlmpilot.controller.airsim import multi_mission  # imported only when needed: keeps the single path independent

            return multi_mission.fly_team(args)
        return mission.fly(args)
    except KeyboardInterrupt:
        # Safety net for Ctrl+C before the mission's own try block (R-04).
        print("[CLI] user interrupt — exiting before takeoff.", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 — any other failure is exit 1 (R-03)
        print(f"[CLI] exiting on fatal error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return 1
