# --- Usage Example ---
"""
Updated Usage (messages JSON):
  python scene_to_prompt.py scenes/scenes_B/basic/scene_B_basic_1.json -t 1.50
Produces to stdout:
  [
    {"role":"system","content":"...instructions..."},
    {"role":"user","content":"=== Scene ===\n..."}
  ]

Programmatic:
  from scene_to_prompt import generate_scene_messages
  msgs = generate_scene_messages(scene_cfg, [1.5])
  # send msgs to OpenAI client

Irregular convex polygons (irregular_design=True) are described with CCW-ordered vertices when provided; otherwise treated as convex polygons with slight vertex perturbations.

Settings:
- list: reveal a list of timestamps; require predict_position() with no args returning nested list per timestamp
- function: hide timestamps; require predict_position(t) returning positions for time t
"""


# Optional helper (not auto-run)
def example_minimal():
    """Print a prompt for a minimal synthetic scene (for quick testing)."""
    minimal_scene = {
        "physics": {"gravity_enabled": False, "gravity_x": 0.0, "gravity_y": -9.8},
        "meta": {},
        "boxes": [
            {
                "center": [400, 300],
                "radius": 160,
                "sides": 8,
                "rotation": 0.0,
                "rotation_speed": 0.4
            }
        ],
        "balls": [
            {"position": [420, 300], "velocity": [60, 0], "radius": 18, "sides": 0}
        ]
    }
    print(generate_scene_prompt(minimal_scene, [1.0]))


import json
import math
import argparse
from typing import Dict, Any, List, Optional
from .math_parser import parse_numeric_value

AXIS_LABELS = {
    "A": "Inner Rotation",
    "B": "Outer Rotation",
    "C": "Outer Translation",
    "D": "Gravity",
    "E": "Irregular Inner Shape",
    "F": "Multiple Containers",
    "G": "Multiple Internal Polygons"
}


def _is_time_varying_rotation(profile: Optional[Dict[str, Any]]) -> bool:
    if not profile:
        return False
    return any(abs(profile.get(k, 0)) > 1e-6 for k in ("omega_amp",))  # simple heuristic


def _summarize_rotation_speed(val) -> str:
    # Parse val for comparison, but keep original for display
    val_numeric = parse_numeric_value(val, 0.0)
    if abs(val_numeric) < 1e-4:
        return "no rotation"
    # If it's a string expression, show it directly; otherwise format as number
    if isinstance(val, str):
        return f"constant angular velocity = {val} rad/s"
    return f"constant angular velocity ≈ {val:.3f} rad/s"


def _summarize_angular_velocity(val) -> str:
    # Parse for comparison, but keep original for display
    try:
        val_numeric = parse_numeric_value(val, 0.0)
    except (ValueError, TypeError):
        return f"angular velocity: {val} rad/s"  # Keep as string if can't convert

    if abs(val_numeric) < 1e-4:
        return "no angular velocity"
    
    # If it's a string expression, show it directly; otherwise format as number
    if isinstance(val, str):
        return f"initial angular velocity = {val} rad/s"
    return f"initial angular velocity ≈ {val:.3f} rad/s"


def _summarize_rotation_profile(profile: Optional[Dict[str, Any]]) -> Optional[str]:
    if not profile:
        return None
    base = profile.get("omega_base", 0.0)
    amp = profile.get("omega_amp", 0.0)
    if abs(amp) < 1e-6:
        return f"angular velocity ≈ {base:.3f} rad/s"
    return f"time-varying angular velocity ω(t)= {base:.3f} + {amp:.3f}·sin(t) rad/s"


def _summarize_translation_path(path: Optional[Dict[str, Any]]) -> Optional[str]:
    if not path:
        return None
    t = path.get("type")
    if t == "sin1d":
        axis = path.get("axis", "x")
        amp = path.get("amplitude", 0)
        freq = path.get("freq", 1.0)
        return f"1D sinusoidal translation along {axis}-axis (amplitude {amp}m, freq {freq}Hz)"
    if t == "lissajous":
        ax = path.get("ax", 0)
        ay = path.get("ay", 0)
        fx = path.get("fx", 1.0)
        fy = path.get("fy", 1.0)
        return f"2D Lissajous path (ax={ax}m, ay={ay}m, fx={fx}Hz, fy={fy}Hz)"
    if t == "piecewise":
        segs = path.get("segments", 4)
        amp = path.get("amplitude", 0)
        return f"piecewise linear oscillation (segments={segs}, amplitude {amp}m)"
    return "custom translation path"


def _summarize_gravity(physics: Dict[str, Any], meta: Dict[str, Any]) -> str:
    enabled = physics.get("gravity_enabled", True)
    gx = physics.get("gravity_x", 0.0)
    gy = physics.get("gravity_y", -9.8)
    if not enabled or (abs(gx) < 1e-6 and abs(gy) < 1e-6):
        base = "no effective gravity (treated as zero)"
    else:
        base = f"constant gravity vector ({gx:.2f}, {gy:.2f}) m/s^2"
    prof = meta.get("gravity_profile", {})
    tv = prof.get("time_variation") if isinstance(prof, dict) else None
    if tv:
        amp = tv.get("amp", 0.0)
        freq = tv.get("freq", 1.0)
        if abs(amp) > 1e-6:
            base += f"; vertical component oscillates with amplitude {amp:.2f} and frequency {freq:.2f}Hz"
    return base


def _infer_axis(scene: Dict[str, Any]) -> str:
    physics = scene.get("physics", {})
    meta = scene.get("meta", {}) or {}
    boxes = scene.get("boxes", [])
    balls = scene.get("balls", [])
    # Flags
    has_gravity = physics.get("gravity_enabled", True) and (abs(parse_numeric_value(physics.get("gravity_x"), 0)) > 1e-6 or abs(
        parse_numeric_value(physics.get("gravity_y"), -9.8) + 9.8) > 1e-6 or "gravity_profile" in meta)
    outer_rotation = any(abs(parse_numeric_value(b.get("rotation_speed"), 0)) > 1e-6 for b in boxes) or "outer_rotation_profile" in meta
    outer_translation = any(b.get("translation_path") for b in boxes)
    multi_containers = len(boxes) > 1
    multi_polys = len(balls) > 1
    irregular_inner = any(b.get("irregular_design") for b in balls)
    def _has_angular_velocity(ball):
        val = ball.get("angular_velocity", 0)
        try:
            val = parse_numeric_value(val, 0.0)
        except (ValueError, TypeError):
            return False
        return abs(val) > 1e-6

    inner_rotation = any(_has_angular_velocity(b) for b in balls)
    # Priority map to single axis expectation
    if inner_rotation: return "A"
    if outer_rotation: return "B"
    if outer_translation: return "C"
    if has_gravity: return "D"
    if irregular_inner: return "E"
    if multi_containers: return "F"
    if multi_polys: return "G"
    # Fallback: choose based on structure
    return "G" if multi_polys else "A"


def _format_container(index: int, box: Dict[str, Any], rotation_profile) -> str:
    sides = box.get("sides", 6)
    # Backward compatible: prefer explicit radius; fall back to diameter/2
    radius = box.get("radius")
    if radius is None:
        diameter_legacy = box.get("diameter", 0)
        radius = diameter_legacy / 2 if diameter_legacy else 0
    center = box.get("center", [0, 0])
    rot_deg = box.get("rotation", 0.0)
    rot_speed = box.get("rotation_speed", 0.0)
    # Changed: Use precise radius with 2 decimal places
    parts: List[str] = [
        f"Container {index}: regular polygon with {sides} sides, radius {radius:.2f}m, center at ({center[0]}, {center[1]})"]
    # changed: report radians instead of degrees
    rot_rad = rot_deg * math.pi / 180.0
    parts.append(f"initial orientation {rot_rad:.3f} rad")
    prof_txt = _summarize_rotation_profile(rotation_profile)
    # Parse rot_speed in case it's a string expression
    rot_speed_parsed = parse_numeric_value(rot_speed, 0.0)
    if abs(rot_speed_parsed) > 1e-6 and not prof_txt:
        parts.append(_summarize_rotation_speed(rot_speed))
    elif prof_txt:
        parts.append(prof_txt)
    path_desc = _summarize_translation_path(box.get("translation_path"))
    if path_desc:
        parts.append(path_desc)
    return "; ".join(parts)


def _format_ball(index: int, ball: Dict[str, Any], include_vertices: bool = False) -> str:
    pos = ball.get("position", [0, 0])
    vel = ball.get("velocity", [0, 0])
    # Keep velocity as-is for display (can be string expressions or numbers)
    vel_x = vel[0] if len(vel) > 0 else 0
    vel_y = vel[1] if len(vel) > 1 else 0
    radius = ball.get("radius", 0)
    sides = ball.get("sides", 0)
    angular_velocity = ball.get("angular_velocity", 0.0)

    # Check for E-type irregular design
    irregular_design = ball.get("irregular_design", False)
    if irregular_design:
        # E-type: irregular convex polygon
        shape = f"irregular convex polygon (approx {sides} sides)" if sides > 2 else "circle"
    else:
        # Standard behavior
        shape = "circle" if sides <= 2 else f"regular polygon ({sides} sides)"

    if irregular_design:
        base = (f"Ball {index}: {shape}, initial position ({pos[0]}, {pos[1]}), "
                f"initial velocity ({vel_x}, {vel_y}) m/s")
    else:
        base = (f"Ball {index}: {shape}, radius {radius}m, initial position ({pos[0]}, {pos[1]}), "
                f"initial velocity ({vel_x}, {vel_y}) m/s")

    # Add angular velocity information
    angular_vel_desc = _summarize_angular_velocity(angular_velocity)
    base += f", {angular_vel_desc}"

    # Add irregular design note for E-type
    if irregular_design:
        irregularity_factor = ball.get("irregularity_factor", 0.0)
        base += f" — irregular convex polygon with slight vertex perturbations (irregularity={irregularity_factor:.2f}), CCW vertices if provided; treat as rigid body with its actual outline"
        if include_vertices:
            verts = ball.get("vertices")  # expected as list[[x,y],...]
            if isinstance(verts, list) and len(verts) >= 3:
                # compute centroid, sort by CCW angle around centroid to standardize
                cx = sum(v[0] for v in verts) / len(verts)
                cy = sum(v[1] for v in verts) / len(verts)

                def ang(v):
                    return math.atan2(v[1] - cy, v[0] - cx)

                verts_sorted = sorted(verts, key=ang)
                # compress to polar about given center (initial position)
                pcx, pcy = pos[0], pos[1]
                polar = []
                for vx, vy in verts_sorted:
                    dx, dy = vx - pcx, vy - pcy
                    ang_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                    r = (dx * dx + dy * dy) ** 0.5
                    polar.append((int(round(ang_deg)), int(round(r))))
                # natural language summary
                polar_str = ", ".join(f"{ang}° at {rad} m" for ang, rad in polar)
                base += f"; CCW vertices in polar coordinates (degrees and meters): {polar_str}"
    return base


def generate_scene_prompt(scene_cfg: Dict[str, Any], timestamps: List[float], include_vertices: bool = False,
                          prompt_setting: str = "list") -> str:
    # Modified: accept multiple timestamps instead of single
    physics = scene_cfg.get("physics", {}) or {}
    meta = scene_cfg.get("meta", {}) or {}
    boxes = scene_cfg.get("boxes", []) or []
    balls = scene_cfg.get("balls", []) or []
    rotation_profile = meta.get("outer_rotation_profile")

    n_balls = len(balls)
    n_timestamps = len(timestamps)

    containers_desc = "\n".join(
        ["- " + _format_container(i + 1, b, rotation_profile) for i, b in enumerate(boxes)]
    ) if boxes else "- No containers."

    balls_desc = "\n".join(
        ["- " + _format_ball(i + 1, b, include_vertices) for i, b in enumerate(balls)]
    ) if balls else "- No dynamic objects."

    gravity_desc = _summarize_gravity(physics, meta)

    dynamics_parts = []
    if rotation_profile:
        dynamics_parts.append("- Outer container rotation varies with time.")
    if any(b.get("translation_path") for b in boxes):
        dynamics_parts.append("- A container executes a prescribed translation path.")
    if "gravity_profile" in meta:
        dynamics_parts.append("- Gravity varies over time.")
    if not dynamics_parts:
        dynamics_parts.append("- No additional time-varying mechanisms.")
    dynamics_desc = "\n".join(dynamics_parts)

    # Conventions block (scene-specific, previously in system)
    conventions_desc = "\n".join([
        "- Containers are convex regular polygons (parameters: 'sides', 'radius', 'center'), unless otherwise specified.",
        "- Angle baseline: By default, the initial orientation is 0.000 rad, pointing to the first vertex along +X (standard Cartesian axes); positive angles rotate CCW about the container center.",
        "- Polygon vertices (if provided) are CCW and form a simple convex polygon.",
        "- Container 'radius' denotes the circumradius (meters).",
        "- For balls: irregular convex polygons rely on provided vertices (no radius mentioned); regular polygons may be derived from 'sides/radius/center/rotation'.",
        "- Containers are kinematic (infinite mass, prescribed motion); impacts do not alter container motion.",
    ])

    # Task description (does not contain general physical laws and coordinates/units, leave to system)
    timestamps_str = ", ".join(f"{t:.2f}s" for t in timestamps)
    if prompt_setting == "function":
        task_lines = "\n".join([
            f"- Number of balls: {n_balls}",
            "- Your should think step by step and write python code.",
            "- The final output should be in the following format: \n[Your thinking steps here...]\n```python\n[Your Python code here]\n```",
            "- Define predict_position(t) returning a list of length n_balls; each element is [x_i, y_i] (rounded to 2 decimals) for Ball i at time t (seconds)",
        ])
        output_desc = "- Required format: function predict_position(t: float) -> [[x1,y1],[x2,y2],...]; coordinates as 2-decimal floats"
    else:
        task_lines = "\n".join([
            f"- Prediction target times: {timestamps_str} ({n_timestamps} timestamps)",
            f"- Number of balls: {n_balls}",
            "- Your should think step by step and write python code.",
            "- The final output should be in the following format: \n<think>[Your thinking steps here]</think>\n```python\n[Your Python code here]\n```",
            f"- Define predict_position() with NO parameters returning a list of length {n_timestamps}; each element is a list of length {n_balls} with [x_i, y_i] (rounded to 2 decimals) for Ball i at that timestamp",
            "- Structure: [timestamp_0_positions, timestamp_1_positions, ...] where each timestamp_X_positions is [[x1,y1], [x2,y2], ...]",
        ])
        output_desc = f"- Required format: nested list with {n_timestamps} timestamps × {n_balls} balls; coordinates as 2-decimal floats"

    return "\n".join([
        "### Scene description",
        "#### Containers",
        containers_desc,
        "\n#### Objects",
        balls_desc,
        "\n### Physics",
        f"- {gravity_desc}.",
        "\n### Dynamics",
        dynamics_desc,
        "\n### Conventions for this scene",
        conventions_desc,
        "\n### Task",
        task_lines,
        "\n### Output",
        output_desc
    ])


# SYSTEM_TEMPLATE = """You are a precise 2D physics reasoning code assistant.
# Goal: Write a program to predict positions of all balls in an ideal elastic 2D polygonal container scene at multiple specified future times.
# Assumptions:
# - Perfectly elastic collisions (restitution 1.0), no energy loss.
# - Ignore rotational inertia of balls unless explicitly needed for linear trajectory changes (only wall reflections matter).
# Instructions:
# 1. Parse the scene description.
# 2. Reconstruct initial linear motion and any active influences (gravity, container rotation/translation).
# 3. Estimate collision sequences; for each wall reflection invert the normal component of velocity.
# 4. Provide a function predict_position() that returns a nested list: [timestamp_0_results, timestamp_1_results, ...] where each timestamp_X_results is [[x1,y1], [x2,y2], ...] for all balls at that timestamp.
# 5. The function must handle multiple timestamps and return results for all of them in chronological order.
# 6. All coordinates should be returned as floats rounded to 2 decimal places using round(value, 2).
# """
old = """
### Debugging tips
- Inputs and units: verify required fields exist; check meters/seconds/radians; confirm radius semantics match the scene conventions.
- Collision geometry: confirm inward normals; validate a single reflection against an axis-aligned edge (e.g., vertical wall) recovers the expected mirrored velocity.
- Finite-size handling: test that shrinking the container by radius prevents center from penetrating walls; compare against a point-particle control.
- Timestamp independence: evaluate t directly from initial conditions, not from prior calls; guard against hidden state.
- Progressive validation: start from simplest scenes (one ball, static container), then enable one effect at a time (gravity, rotation, translation) to isolate failures.
"""
def generate_scene_messages(scene: Dict[str, Any], timestamps: List[float],
                            include_vertices: bool = False, prompt_setting: str = "list") -> List[Dict[str, str]]:
    """Generate chat messages for LLM based on scene configuration."""

    user_content = generate_scene_prompt(scene, timestamps, include_vertices=include_vertices,
                                         prompt_setting=prompt_setting)


    # Academic-style, generalized system prompt (mechanics categorized; scene-specific conventions live in user content)
    # NOTE: use user role rather than system since there are general system prompts
    if prompt_setting == "function":
        system_prompt = """## Polygon Dynamics Prediction
In this task, you will implement a single function predict_position(t) that computes the 2D positions of all balls at an arbitrary future time t under idealized mechanics. The function parses the scene configuration (containers, balls, and physics/meta), reconstructs the motions, detects and handles boundary collisions with finite-size treatment, and returns a list where each element is the [x, y] position (rounded to 2 decimals) of a ball at time t. Each evaluation of t must be computed directly from initial conditions and scene mechanics with no hidden state or accumulation across calls. Rendering, animation, and explanatory text are out of scope; prefer closed-form reasoning and avoid coarse time-stepping except where narrowly required for collision resolution.

### Mechanics (General)
- Kinematics: Use closed-form equations under constant acceleration: x(t)=x0+vx0*t+0.5*ax*t^2, y(t)=y0+vy0*t+0.5*ay*t^2.
- Collisions: Perfectly elastic. Reflect velocity using v' = v - 2·dot(v, n̂)·n̂, where n̂ is the inward unit normal at the contact.
- Finite size: Use polygon–polygon contact. Derive regular shapes from ('sides','radius','center','rotation'); irregular convex polygon balls use provided vertices.
- Geometry: Irregular convex polygons (if present) are simple (non self-intersecting). Ball finite size must be respected in all interactions.
- Units: Positions in meters; time in seconds; angles in radians; velocities in m/s; accelerations in m/s^2.
- Cartesian Axes: +X is right, +Y is up.

### Constraints
- Implement only predict_position(t); no other entry points will be called.
- No global variables; no variables defined outside the function.
- Do not import external libraries (except math); do not perform I/O; do not print; do not use randomness.
- Numerical output must be round(value, 2); normalize -0.0 to 0.0.

### Verification and output contract
- Return a list of positions per ball for the provided t: [[x1,y1],[x2,y2],...].
- Each call must be computed independently (no state carry-over between calls).
- You should assume that the ball will hit the wall and bounce back, which will be verified in test cases.
"""
    else:
        system_prompt = """## Polygon Dynamics Prediction
In this task, you will implement a single function predict_position() that computes the 2D positions of all balls at a set of specified future timestamps under idealized mechanics. The function parses the scene configuration (containers, balls, and physics/meta), reconstructs the motions, detects and handles boundary collisions with finite-size treatment, and returns a nested list where each outer element corresponds to a timestamp and each inner element is the [x, y] position (rounded to 2 decimals) of a ball. Each timestamp must be evaluated directly from initial conditions and scene mechanics without reusing or accumulating state across timestamps. Rendering, animation, and explanatory text are out of scope; prefer closed-form reasoning and avoid coarse time-stepping except where narrowly required for collision resolution.

### Mechanics (General)
- Kinematics: Use closed-form equations under constant acceleration: x(t)=x0+vx0*t+0.5*ax*t^2, y(t)=y0+vy0*t+0.5*ay*t^2.
- Collisions: Perfectly elastic. Reflect velocity using v' = v - 2·dot(v, n̂)·n̂, where n̂ is the inward unit normal at the contact.
- Finite size: Use polygon–polygon contact. Derive regular shapes from ('sides','radius','center','rotation'); irregular convex polygon balls use provided vertices.
- Geometry: Irregular convex polygons (if present) are simple (non self-intersecting). Ball finite size must be respected in all interactions.
- Units: Positions in meters; time in seconds; angles in radians; velocities in m/s; accelerations in m/s^2.
- Cartesian Axes: +X is right, +Y is up.

### Constraints
- Implement only predict_position(); no other entry points will be called.
- No global variables; no variables defined outside the function.
- Do not import external libraries (except math); do not perform I/O; do not print; do not use randomness.
- Numerical output must be round(value, 2); normalize -0.0 to 0.0.

### Verification and output contract
- Return a nested list: outer index over timestamps (chronological), inner index over balls per timestamp: [[[x1,y1],[x2,y2],...], ...].
- Each timestamp must be computed independently (no state carry-over from previous timestamps).
- You should assume that the ball will hit the wall and bounce back, which will be verified in test cases.
"""

    return [
        {"role": "user", "content": system_prompt + "\n\n" + user_content}
    ]


# Update example_minimal to show messages usage
def example_minimal_messages():
    """Print JSON messages for a minimal synthetic scene."""
    import json as _json
    minimal_scene = {
        "physics": {"gravity_enabled": False, "gravity_x": 0.0, "gravity_y": -9.8},
        "meta": {},
        "boxes": [
            {"center": [400, 300], "radius": 160, "sides": 8, "rotation": 0.0, "rotation_speed": 0.4}
        ],
        "balls": [
            {"position": [420, 300], "velocity": [60, 0], "radius": 18, "sides": 6}
        ]
    }
    msgs = generate_scene_messages(minimal_scene, [1.0])
    print(_json.dumps(msgs, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Convert a scene JSON into OpenAI messages (system+user).")
    parser.add_argument("scene_json", help="Path to scene JSON (without .json accepted if full path exists).")
    parser.add_argument("--timestamp", "-t", type=float, default=1.0, help="Prediction timestamp (seconds).")
    parser.add_argument("--timestamps", "-ts", nargs="+", type=float,
                        help="Multiple prediction timestamps (seconds).")  # Added
    parser.add_argument("--with-vertices", action="store_true",
                        help="Include compact polar vertex summary for irregular convex polygons if available.")
    parser.add_argument("--prompt-setting", choices=["list", "function"], default="list",
                        help="Prompt style: 'list' reveals timestamps and expects nested-list output; 'function' hides timestamps and requires predict_position(t).")
    parser.add_argument("--out", "-o", help="Optional output text file.")
    args = parser.parse_args()

    path = args.scene_json
    if not path.endswith(".json"):
        if not path.lower().endswith(".json"):
            path_json = path + ".json"
            try:
                with open(path_json, "r", encoding="utf-8"):
                    path = path_json
            except FileNotFoundError:
                pass

    with open(path, "r", encoding="utf-8") as f:
        scene_cfg = json.load(f)

    # Modified: support multiple timestamps
    if args.timestamps:
        timestamps = sorted(args.timestamps)
    else:
        timestamps = [args.timestamp]

    messages = generate_scene_messages(scene_cfg, timestamps, include_vertices=args.with_vertices,
                                       prompt_setting=args.prompt_setting)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    print(json.dumps(messages, indent=2))


if __name__ == "__main__":
    main()

