# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""agent_mission.py — one head/body multi-agent episode (``agent_mode: head_body``).

Sibling of :func:`vlmpilot.controller.airsim.multi_mission.fly_team`, whose loop is lockstep — one capture and
one VLM call for N actions. Here each drone gets its own thread, endpoint and pace, and a head
re-plans on a timer or on a declaration. One RPC client per consumer, because a msgpack-rpc
client is not thread-safe: ``main`` for the head's grabs and calls, ``body-<name>`` ×N on
``fleet.clients[i]``, and each FlightController's own ``cmd-worker``."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from vlmpilot import artifacts
from vlmpilot.controller.airsim import flight_controller as _fc, mission, multi_mission
from vlmpilot.policy import vlm_backends
from vlmpilot.policy.body_policy import DEFAULT_ORDER_TTL_STEPS, BodyAgent, line_atomic_stdout, say
from vlmpilot.policy.camera import PinholeCamera
from vlmpilot.controller.airsim.fleet import Fleet, grab_scene_bgr
from vlmpilot.policy.head_policy import (DEFAULT_HISTORY_TURNS, DEFAULT_MAX_CALLS, DEFAULT_POLL_INTERVAL_S,
                          PROMPT_DEFAULT, HeadAgent, resolve_prompt)
from vlmpilot.policy.policy import ACTION_RESPONSE_SCHEMA, ACTION_SCHEMA_NAME, VlmPolicy

# Not the generic 8 s backend default: a head call takes tens of seconds, and a body that times
# out holds its drone.
DEFAULT_HEAD_TIMEOUT_S = 60.0
DEFAULT_BODY_TIMEOUT_S = 15.0
# Teardown must outlast a body's VLM call, or abandoned threads wake up and drive RPCs into a
# fleet that is already landing.
BODY_JOIN_MARGIN_S = 3.0


def _section(config: Mapping[str, Any], key: str) -> dict:
    value = config.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _opt_num(value: Any, default: float) -> float:
    """Config number that may legitimately be 0: ``x or default`` would eat ``poll_interval_s: 0``."""
    return default if value is None or value == "" else float(value)


def _sensor_client_factory(port: int) -> Any:
    """A maker of fresh RPC connections, to be CALLED on the thread that will use the result:
    each carries its own tornado IOLoop, safe exactly as long as one thread drives it."""
    def make() -> Any:
        client = _fc._api().MultirotorClient(port=port)
        client.confirmConnection()
        return client
    return make


def _body_timeout_s(body_cfg: Mapping[str, Any]) -> float:
    """A body's VLM deadline — also the longest a body can block, so teardown waits on it."""
    return float(body_cfg.get("vlm_timeout", DEFAULT_BODY_TIMEOUT_S) or DEFAULT_BODY_TIMEOUT_S)


def _order_ttl(config: Mapping[str, Any], body_cfg: Mapping[str, Any]) -> Any:
    """A body's step budget per order: the ``body:`` key, else the fleet-wide one. Three
    readers (headless clamp, body construction, head) and they must never drift apart."""
    return body_cfg.get("order_ttl_steps", config.get("order_ttl_steps", DEFAULT_ORDER_TTL_STEPS))


# ``head.enabled: false`` (or ``head.model_name: none``) removes the commander entirely — no
# backend, no client, no call. Every body then flies the mission text alone (head-less baseline).
_HEAD_OFF_NAMES = ("none", "off", "disabled")


def _head_enabled(head_cfg: Mapping[str, Any]) -> bool:
    if not bool(head_cfg.get("enabled", True)):
        return False
    return str(head_cfg.get("model_name", "") or "").strip().lower() not in _HEAD_OFF_NAMES


def _headless_body_cfg(config: Mapping[str, Any], body_cfg: Mapping[str, Any],
                       reason: str = "no head") -> dict:
    """Body config for an arm whose head will not speak again: the order TTL must be off.

    Only the head can clear the TTL, so with no head — or with a one-shot head, which is the
    same thing from t0 on — a non-zero one is a countdown to a fleet frozen on ``THINK(awaiting
    fresh orders)``. Forced here rather than trusted to the yaml, because that failure is
    silent and costs a whole batch."""
    cfg = dict(body_cfg)
    ttl = int(_order_ttl(config, cfg) or 0)
    if ttl > 0:
        say(f"[CONFIG] {reason} — lowering body.order_ttl_steps={ttl} to 0 (unlimited): "
            "with no commander to renew orders, every drone would freeze in HOLD once the TTL expires")
    cfg["order_ttl_steps"] = 0
    if bool(cfg.get("wait_after_finished", False)):
        say(f"[CONFIG] ⚠ {reason} + body.wait_after_finished=true — there is no commander to wake a drone "
            "that declared, so it stays put until the episode ends")
    return cfg


# Keys under `head:` the head *loop* reads; everything else there is a backend option.
_HEAD_LOOP_KEYS = frozenset({"enabled", "api_provider", "model_name", "poll_interval_s",
                             "max_calls", "timeout_s", "history_turns", "prompt", "map"})


def _head_backend(config: Mapping[str, Any], head_cfg: Mapping[str, Any]) -> Any:
    # Same shared inheritance as _body_backends: without it an `openai` head silently hits the
    # default endpoint.
    cfg = {k: v for k, v in (config or {}).items() if k not in ("head", "body")}
    # Then every non-orchestration key under `head:` on top. Scoped here, not at the top level:
    # the bodies read that same dict and must stay byte-identical to the centralized arm.
    cfg.update({k: v for k, v in (head_cfg or {}).items() if k not in _HEAD_LOOP_KEYS})
    cfg.update(
        api_provider=str(head_cfg.get("api_provider", "gemini") or "gemini"),
        model_name=str(head_cfg.get("model_name", "") or ""),
        vlm_timeout=float(head_cfg.get("timeout_s", DEFAULT_HEAD_TIMEOUT_S) or DEFAULT_HEAD_TIMEOUT_S))
    if not vlm_backends.supports_multi_image(cfg):
        raise vlm_backends.VlmConfigError(
            f"head api_provider={cfg['api_provider']!r} cannot send multiple images, so it cannot be the commander")
    return vlm_backends.build_backend(cfg)


# Providers that read a per-drone address out of ``body.base_urls``. Only `openai` does — one
# vLLM per drone, and a missing list would silently serialize the fleet onto one endpoint. A
# hosted provider has no address to split; one backend (one connection pool) per drone suffices.
_ENDPOINT_PROVIDERS = ("openai",)


def _body_backends(config: Mapping[str, Any], body_cfg: Mapping[str, Any],
                   drone_names: Sequence[str]) -> list[Any]:
    """One backend per drone, requesting exactly what the centralized arm's single VLM does —
    only the endpoint and the timeout differ."""
    provider = str(body_cfg.get("api_provider", "openai") or "openai")
    base_urls = list(body_cfg.get("base_urls") or [])
    # Given at all → one per drone, whatever the provider. Omitted → only a hosted provider may.
    if (base_urls or provider in _ENDPOINT_PROVIDERS) and len(base_urls) < len(drone_names):
        raise vlm_backends.VlmConfigError(
            f"body.base_urls has only {len(base_urls)} entries — {len(drone_names)} drones need one each")
    base = {k: v for k, v in (config or {}).items() if k not in ("head", "body")}
    base.update(api_provider=provider, model_name=str(body_cfg.get("model_name", "") or ""),
                vlm_timeout=_body_timeout_s(body_cfg))
    backends = []
    for i, name in enumerate(drone_names):
        cfg = dict(base)
        if base_urls:
            cfg["openai_base_url"] = str(base_urls[i])
        else:
            # A top-level openai_base_url is the centralized arm's — never a hosted body's.
            cfg.pop("openai_base_url", None)
        backends.append(vlm_backends.build_backend(
            cfg, response_schema=ACTION_RESPONSE_SCHEMA, schema_name=ACTION_SCHEMA_NAME))
        endpoint = base_urls[i] if base_urls else "{}:{}".format(provider, cfg["model_name"])
        say(f"[BODY] endpoint drone={name} {endpoint}")
    return backends


# Knobs read by VlmPolicy. Missing from the body section → fall back to the fleet-wide value. The same config
# keys must yield the same body as the centralized arm, so the only difference between the two arms is "is there a commander".
_POLICY_KEYS = ("point_order", "point_scale", "history_len", "intrinsics_in_prompt",
                "plan_schema", "mm_history", "mm_history_scale", "think_action")


def _body_policies(config: Mapping[str, Any], body_cfg: Mapping[str, Any],
                   camera: PinholeCamera, backends: Sequence[Any]) -> list[Any]:
    """One standalone policy per drone. Built before the Fleet: the controller must record command
    length / travel distance through this policy's note_command so the history matches the standalone arm.

    Backend knobs come from the body section only (fleet-wide values belong to the commander / centralized arm)."""
    cfg = {k: config[k] for k in _POLICY_KEYS if k in config}
    cfg.update({k: body_cfg[k] for k in _POLICY_KEYS if k in body_cfg})
    cfg["api_provider"] = str(body_cfg.get("api_provider", "openai") or "openai")
    cfg["model_name"] = str(body_cfg.get("model_name", "") or "")
    return [VlmPolicy.from_config(dict(cfg), camera, b) for b in backends]


def _build_bodies(
    *,
    config: Mapping[str, Any],
    body_cfg: Mapping[str, Any],
    fleet: Fleet,
    policies: Sequence[Any],
    port: int,
    stop_event: threading.Event,
    declaration_event: threading.Event,
    on_decision: Any,
    on_declared: Any,
) -> list[BodyAgent]:
    camera_name = str(config.get("camera_name", mission.DEFAULT_CAMERA_NAME))
    arrival_frames = int(config.get("arrival_frames", mission.DEFAULT_ARRIVAL_FRAMES)
                         or mission.DEFAULT_ARRIVAL_FRAMES)
    return [
        BodyAgent(
            name=name,
            index=i,
            controller=fleet.controllers[i],
            policy=policies[i],
            # fleet.clients[i] is the controller's; a body senses on its own connection.
            make_sensor_client=_sensor_client_factory(port),
            grab_frame=lambda c, n=name: grab_scene_bgr(c, camera_name, n),
            stop_event=stop_event,
            declaration_event=declaration_event,
            order_ttl_steps=int(_order_ttl(config, body_cfg)),
            arrival_frames=arrival_frames,
            # When on, a drone that emitted FINISHED holds until a new command (default off: keeps existing arms reproducible).
            wait_after_finished=bool(body_cfg.get("wait_after_finished", False)),
            on_decision=on_decision,
            on_declared=on_declared,
        )
        for i, name in enumerate(fleet.drone_names)
    ]


def _fly_headless(bodies: Sequence[BodyAgent], mission_text: str, stop_event: threading.Event) -> None:
    """The head-less baseline: hand every body the mission verbatim, then just wait. Nothing
    re-plans, so the main thread only holds the episode open until ``stop_event``."""
    for body in bodies:
        body.set_orders(mission_text)
        say(f"[NOHEAD] {body.name} <- {mission_text}")
    say(f"[NOHEAD] {len(bodies)} drones each run the original mission text with no commander — no re-orders")
    stop_event.wait()


def fly_agents(args: Any) -> int:
    """Head/body multi-drone body. Returns the process exit code."""
    config_path = getattr(args, "config", None) or mission.DEFAULT_CONFIG_PATH
    say(f"[CONFIG] config file: {config_path}")
    config = mission.load_mission_config(config_path)
    port = mission.resolve_port(getattr(args, "port", None), config)

    drone_names = [str(n) for n in (config.get("drone_names") or [])]
    if not drone_names:
        raise RuntimeError("head_body mode but the config has no drone_names")
    n = len(drone_names)
    head_cfg = _section(config, "head")
    body_cfg = _section(config, "body")
    head_on = _head_enabled(head_cfg)
    if not head_on:
        body_cfg = _headless_body_cfg(config, body_cfg)
    elif _opt_num(head_cfg.get("poll_interval_s"), DEFAULT_POLL_INTERVAL_S) < 0:
        # After the initial order the watch loop never runs, so nobody refreshes the TTL, exactly as with no head.
        # Same silent freeze, so the same enforcement applies here (not left to a single yaml line).
        body_cfg = _headless_body_cfg(config, body_cfg, "head one-shot(poll_interval_s<0)")
    # Before the scene loads: a head.prompt typo surfacing later has already cost a placement.
    head_prompt = resolve_prompt(head_cfg.get("prompt", PROMPT_DEFAULT)) if head_on else PROMPT_DEFAULT
    say(f"[CONFIG] Drones: {', '.join(drone_names)}")
    say(f"[CONFIG] agent_mode=head_body head={head_cfg.get('model_name') if head_on else 'none (disabled)'} "
        f"body={body_cfg.get('model_name')}")
    if head_on and head_prompt != PROMPT_DEFAULT:
        say(f"[CONFIG] head.prompt={head_prompt} — the commander prompt is not the default")
    if bool(body_cfg.get("wait_after_finished", False)):
        say("[CONFIG] body.wait_after_finished=true — a drone that declared stops until the commander's next order")

    camera_name = str(config.get("camera_name", mission.DEFAULT_CAMERA_NAME))
    recording_root = str(config.get("recording_root", mission.DEFAULT_RECORDING_ROOT))
    output_root = str(config.get("output_root", mission.DEFAULT_OUTPUT_ROOT))
    start_poses = config.get("start_poses") or {}
    continue_after_arrival = os.environ.get(mission.ENV_CONTINUE_AFTER_ARRIVAL) == "1"
    record_wanted = bool(config.get("auto_record", False)) and os.environ.get(mission.ENV_NO_RECORD) != "1"
    say(f"[CONFIG] camera={camera_name!r} continue_after_arrival={continue_after_arrival} "
        f"record={record_wanted} start_poses={'yes' if start_poses else 'no'}")

    stop_event = threading.Event()
    declaration_event = threading.Event()
    save_lock = threading.Lock()  # N body threads writing decision_*.jpg into one run dir
    state = multi_mission._TeamState(drone_names)
    threads: list[threading.Thread] = []
    # Outlives the loop: bodies keep printing while they drain, so it closes in the finally.
    stdout_guard = contextlib.ExitStack()
    try:
        # 2N: first N for the drones, last N for the commander's parallel capture (no commander → no last N).
        state.clients = multi_mission._connect_all(port, 2 * n if head_on else n)
        head_clients = state.clients[n:]
        probe = multi_mission.probe_cameras(
            dict(zip(drone_names, state.clients)), camera_name, drone_names)
        if probe is None:
            return 1
        width, height, needs_confirm = probe
        if needs_confirm and not mission.confirm_low_resolution():
            return 1

        state.run_dir = artifacts.create_run_dir(output_root)
        say(f"[RUN] output folder: {state.run_dir}")
        run_dir = state.run_dir

        def _on_decision(record: Any) -> None:
            with save_lock:
                artifacts.save_decision(run_dir, record)

        def _on_declared(body: BodyAgent) -> None:
            state.declared_once = True
            if not continue_after_arrival:  # like the single/team paths: first arrival ends it
                state.reached = True
                stop_event.set()

        camera = PinholeCamera(width, height, float(config.get("fov_horizontal", mission.DEFAULT_FOV_HORIZONTAL_DEG)))
        state.backend = _head_backend(config, head_cfg) if head_on else None
        backends = _body_backends(config, body_cfg, drone_names)
        policies = _body_policies(config, body_cfg, camera, backends)
        state.fleet = Fleet(
            config=config,
            drone_names=drone_names,
            image_width=width,
            image_height=height,
            run_dir=state.run_dir,
            team_policy=None,  # no centralized policy in this mode — each body decides for itself
            camera=camera,
            on_decision=None,
            on_command=lambda i, duration_s, distance_m=None: policies[i].note_command(duration_s, distance_m),
            clients=state.clients[:n],
        )

        state.instruction = mission.read_instruction(mission.INSTRUCTION_PROMPT)
        if not state.instruction:
            say("[INPUT] instruction is empty or input was closed — exiting without takeoff (exit 2)")
            return 2
        state.airborne_phase = True
        say(f"[START] instruction: {state.instruction!r} — taking off in {multi_mission.START_DELAY_S:g}s")
        time.sleep(multi_mission.START_DELAY_S)
        if start_poses:
            state.fleet.apply_start_poses(start_poses)
        state.fleet.takeoff_all()
        if start_poses:
            state.fleet.climb_to_start_alt(start_poses)
        if record_wanted:
            multi_mission._start_recording(state, recording_root)

        stdout_guard.enter_context(line_atomic_stdout())
        bodies = _build_bodies(
            config=config, body_cfg=body_cfg, fleet=state.fleet, policies=policies,
            port=port, stop_event=stop_event, declaration_event=declaration_event,
            on_decision=_on_decision, on_declared=_on_declared)
        head = None
        if head_on:
            head = HeadAgent(
                backend=state.backend, bodies=bodies, clients=head_clients, camera=camera,
                camera_name=camera_name, mission_text=state.instruction, run_dir=state.run_dir,
                poll_interval_s=_opt_num(head_cfg.get("poll_interval_s"), DEFAULT_POLL_INTERVAL_S),
                max_calls=int(_opt_num(head_cfg.get("max_calls"), DEFAULT_MAX_CALLS)),
                history_turns=int(head_cfg.get("history_turns", DEFAULT_HISTORY_TURNS) or DEFAULT_HISTORY_TURNS),
                order_ttl_steps=int(_order_ttl(config, body_cfg)),
                api_provider=str(head_cfg.get("api_provider", "gemini") or "gemini"),
                model_name=str(head_cfg.get("model_name", "") or ""),
                prompt=head_prompt,
                # control-tower arm: text-free overview map as the last image (head_map);
                # ``scene`` is the geometry the runner wrote into this episode's config
                map_on=bool(head_cfg.get("map", False)),
                scene=config.get("scene") or None)
        say("[LOOP] commander/drone loop started: each drone flies at its own pace, the commander re-orders periodically and on each declaration" if head_on
            else "[LOOP] drone loop started: no commander, each drone gets the original mission text and flies at its own pace")
        say("[LOOP] a single Ctrl+C runs the wrap-up (grid · video · landing)")
        threads = [threading.Thread(target=b.run, name=f"vlmpilot-body-{b.name}", daemon=True)
                   for b in bodies]
        for thread in threads:
            thread.start()
        if head is None:
            _fly_headless(bodies, state.instruction, stop_event)
        else:
            head.assign_initial()  # nothing moves until this lands
            head.poll_loop(stop_event, declaration_event)
    except KeyboardInterrupt:
        say("[LOOP] interrupt received — moving to wrap-up")
    except Exception as exc:  # noqa: BLE001 — same two-phase rule as mission.fly / fly_team
        if not state.airborne_phase:
            raise
        say(f"[LOOP] unexpected error, ending the episode: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
    finally:
        _stop_bodies(stop_event, declaration_event, threads,
                     _body_timeout_s(body_cfg) + BODY_JOIN_MARGIN_S)
        stdout_guard.close()
        # Grid frames come from the bodies; leaving state.backend (the head) would label them as the head.
        state.backend = None
        if continue_after_arrival:
            # If assembly (400+ jpg) exceeds CLEANUP_GRACE (120s), the runner's kill corrupts the sim RPC.
            say("[LOOP] bench mode — skipping grid/video assembly, landing only")
            if state.fleet is not None:
                try:
                    state.fleet.shutdown()
                except (Exception, KeyboardInterrupt) as exc:
                    say(f"[LOOP] error during fleet shutdown (ignored): {type(exc).__name__}: {exc}")
        else:
            multi_mission._finalize(
                state, recording_root, model=str(body_cfg.get("model_name", "") or "") or None)
    return 0


def _stop_bodies(stop_event: threading.Event, declaration_event: threading.Event,
                 threads: Sequence[threading.Thread], join_timeout: float) -> None:
    """Signal every body and wait out its longest blocking call before landing.

    ``join_timeout`` is a wall-clock budget for the whole fleet, not per thread: the bodies wake
    at once. A body still inside a call when it runs out is abandoned — they are daemons."""
    stop_event.set()
    declaration_event.set()
    deadline = time.monotonic() + float(join_timeout)
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [t.name for t in threads if t.is_alive()]
    if alive:
        say(f"[BODY] {', '.join(alive)} did not finish within {join_timeout:g}s — giving up on them and moving to landing")
