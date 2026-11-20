#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Run the executor (B) with the scripted/oracle backend, without modifying vlmpilot.

    python3 verifier/isaacsim/run_scripted_episode.py \
        --config <config with api_provider: scripted> [--instruction "..."]   # scripted
    python3 verifier/isaacsim/run_scripted_episode.py \
        --config vlmpilot/configs/isaacsim/single-drone/pegasus_oracle.yaml   [--instruction "..."]   # oracle

How it works
  1. `scripted_backend.register()` / `oracle_backend.register()` add `"scripted"` /
     `"oracle"` to `vlm_backends._BACKEND_CLASSES`. `build_backend` looks up that
     table, so `api_provider: scripted|oracle` in the config works with no code change.
  2. `vlmpilot.cli.main()` is called as is, so the full executor path runs.

The executor reads the instruction from **stdin** (same interface as `verifier/`), so
`--instruction` is pushed into stdin. Without it, the terminal stdin is used.

Process A must already be running. The oracle reads A's GT sidecar, so targets are needed too:
    $ISAACSIM/python.sh scripts/isaacsim/sim/launch_brownstone.py --variant Morning \
        --gt --gt-dir <run> --targets 'trashcan:orange' --vehicles 1 --render-hz 10
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    instruction = None
    if "--instruction" in argv:
        i = argv.index("--instruction")
        if i + 1 >= len(argv):
            print("[SCRIPTED] --instruction needs a value")
            return 2
        instruction = argv[i + 1]
        del argv[i:i + 2]

    sys.path.insert(0, os.path.join(REPO, "vlmpilot", "src"))
    sys.path.insert(0, os.path.join(REPO, "scripts", "isaacsim"))

    from sim import oracle_backend, scripted_backend
    scripted_backend.register()
    oracle_backend.register()

    if instruction is not None:
        # The executor reads one instruction line from stdin. With EOF after it,
        # the script emits finished and exits cleanly.
        sys.stdin = io.StringIO(instruction + "\n")
        print(f"[SCRIPTED] stdin instruction: {instruction!r}", flush=True)

    from vlmpilot import cli
    return cli.main(["pegasus"] + argv)


if __name__ == "__main__":
    sys.exit(main())
