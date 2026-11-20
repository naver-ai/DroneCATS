# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""``python -m vlmpilot`` — propagate ``main()``'s return value as the exit code (R-05)."""

from vlmpilot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
