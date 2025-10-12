import os
import json
import math
import random
import argparse
from typing import Dict, Any, List, Tuple
from itertools import combinations  # Added

from .scene_config import SCENE_SPACE_CONFIG, GLOBAL_SCENE_CONSTANTS, DIFF_NAMES, INTERNAL_SIDES_MAX, AXIS_RANGES
from .scene_config import AXIS_COLORS, COLOR_RANDOM_RANGE


def to_int(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(round(v))
    return v


def to_decimal(v):
    """Round float to two decimal places"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return v


# Get current configuration
def get_current_config() -> dict:
    return SCENE_SPACE_CONFIG

def get_scene_center(difficulty: int) -> Tuple[float, float]:
    cfg = get_current_config()
    cx_frac, cy_frac = GLOBAL_SCENE_CONSTANTS["center_fraction"]
    return cfg["base_width"] * cx_frac, cfg["base_height"] * cy_frac

def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    """Calculate distance between two points"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def _get_project_root() -> str:
    # This file is located at scene_generation/scene_generator.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _resolve_dataset_name(out_root: str) -> str:
    """Resolve dataset name based on out_root (required to be under data/scenes/<dataset_name>)."""""
    project_root = _get_project_root()
    scenes_root = os.path.join(project_root, "data", "scenes")
    out_abs = os.path.abspath(out_root)
    scenes_root_abs = os.path.abspath(scenes_root)
    if not out_abs.startswith(scenes_root_abs + os.sep) and out_abs != scenes_root_abs:
        raise ValueError(f"Output directory must be located under data/scenes: given out_root={out_root}")
    rel = os.path.relpath(out_abs, scenes_root_abs)
    # Take the first level directory as dataset name
    parts = rel.split(os.sep)
    dataset_name = parts[0] if parts and parts[0] != "." else ""
    if not dataset_name:
        raise ValueError(f"Unable to resolve dataset name, out_root={out_root}")
    return dataset_name

def _predict_positions(scene_rel_without_ext: str, dataset_name: str, t: float, dt: float) -> List[List[float]]:
    # Delayed import to avoid differences in relative package execution
    try:
        from src.utils.predict import predict_ball_positions
    except Exception:
        # Fallback: try importing from utils (if src is already in PYTHONPATH)
        from utils.predict import predict_ball_positions
    return predict_ball_positions(scene_rel_without_ext, t, dataset_name=dataset_name, time_step=dt)

def _validate_scene_difference(scene_rel_without_ext: str,
                               dataset_name: str,
                               timestamps: List[float],
                               dt_default: float,
                               dt_target: float,
                               tolerance_px: float) -> Tuple[bool, float]:
    """Return (whether passed, maximum error)."""""
    max_err = 0.0
    for ts in timestamps:
        pos_def = _predict_positions(scene_rel_without_ext, dataset_name, ts, dt_default)
        pos_tgt = _predict_positions(scene_rel_without_ext, dataset_name, ts, dt_target)
        if len(pos_def) != len(pos_tgt):
            return False, float("inf")
        frame_err = 0.0
        for a, b in zip(pos_def, pos_tgt):
            d = calculate_distance(a, b)
            if d > frame_err:
                frame_err = d
        if frame_err > max_err:
            max_err = frame_err
        if max_err > tolerance_px:
            return False, max_err
    return True, max_err

def get_scene_dimensions(difficulty: int) -> Tuple[float, float]:
    """Get scene space dimensions based on difficulty."""
    config = get_current_config()
    return config["base_width"], config["base_height"]

def get_outer_container_diameter(difficulty: int) -> float:
    """Get outer container diameter (strictly from configured box_base_diameter)."""""
    cfg = get_current_config()
    if "box_base_diameter" not in cfg:
        raise KeyError("SCENE_SPACE_CONFIG missing 'box_base_diameter'")
    return float(cfg["box_base_diameter"])

def get_container_spacing(difficulty: int) -> float:
    """Get container spacing parameters (from configuration)."""""
    config = get_current_config()
    base_min_gap = config["multi_container"]["min_gap"]
    return base_min_gap

def clamp_sides(n: int, is_outer=False) -> int:
    if is_outer:
        return max(3, n)  # Still allow >32 if it appears after combination; performance recommendation not to hard truncate
    return min(INTERNAL_SIDES_MAX, max(3, n))

def sample_int(low: int, high: int) -> int:  # Modified: return integer
    return random.randint(low, high)

def sample_float(low: float, high: float) -> float:
    return to_decimal(random.uniform(low, high))

def sample_velocity(speed_range: Tuple[int, int]) -> Tuple[float, float]:
    lo, hi = speed_range
    if hi <= 0:
        return 0.0, 0.0
    v = sample_float(lo, hi)
    theta = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
    return v * math.cos(theta), v * math.sin(theta)

def gravity_vector(mode: str) -> Tuple[float, float, bool, Dict[str, float]]:
    # Return (gx, gy, enabled, extra_meta)
    gcfg = GLOBAL_SCENE_CONSTANTS["gravity"]
    if mode == "tiny":
        return (0.0, -gcfg["tiny"], True, {})
    if mode == "small":
        return (0.0, -gcfg["small"], True, {})
    if mode == "large":
        return (0.0, -gcfg["large"], True, {})
    if mode == "tilted":
        angle = random.uniform(-math.radians(gcfg["tilt_max_deg"]), math.radians(gcfg["tilt_max_deg"]))
        g = gcfg["large"]
        return (g * math.sin(angle), -g * math.cos(angle), True, {"tilt_angle_deg": math.degrees(angle)})
    if mode == "time_var":
        tv = gcfg["time_var_default"]
        return (0.0, -gcfg["g"], True, {"time_variation": {"amp": tv["amp_ratio"] * gcfg["g"], "freq": tv["freq"]}})
    if mode == "chaotic":
        # Random direction + random strength + time variation
        angle = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
        lo, hi = gcfg["chaotic_mag"]
        magnitude = random.uniform(lo, hi)
        gx = magnitude * math.cos(angle)
        gy = magnitude * math.sin(angle)
        return (gx, gy, True, {
            "chaotic_gravity": True,
            "base_angle": math.degrees(angle),
            "magnitude": magnitude,
            "time_variation": {"amp": gcfg["time_variation"]["chaotic_amp_ratio"] * magnitude, "freq": random.uniform(*gcfg["time_variation"]["chaotic_freq_range"])}
        })
    if mode == "extreme_chaotic":
        # Extreme chaotic gravity: higher strength, completely random direction
        angle = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
        lo, hi = gcfg["extreme_chaotic_mag"]
        magnitude = random.uniform(lo, hi)
        gx = magnitude * math.cos(angle)
        gy = magnitude * math.sin(angle)
        return (gx, gy, True, {
            "extreme_chaotic_gravity": True,
            "base_angle": math.degrees(angle),
            "magnitude": magnitude,
            "time_variation": {"amp": gcfg["time_variation"]["extreme_amp_ratio"] * magnitude, "freq": random.uniform(*gcfg["time_variation"]["extreme_freq_range"])},
            "random_jumps": True
        })
    return (0.0, 0.0, False, {})

def place_inside(center, diameter, count: int, radius: int, difficulty: int, sides: int) -> List[Tuple[float, float]]:
    """Place objects inside regular polygon containers, based on incircle constraints on sphere centers, avoid touching edges/penetration"""
    cx, cy = center
    # Use regular polygon incircle radius, subtract ball radius and safety clearance
    clearance = GLOBAL_SCENE_CONSTANTS["placement"]["container_clearance"]
    overlap_margin = GLOBAL_SCENE_CONSTANTS["overlap_check_margin"]
    R_in = (diameter / 2.0) * math.cos(math.pi / max(3, int(sides)))
    R_allow = R_in - radius - max(clearance, overlap_margin)
    if R_allow <= 0:
        raise RuntimeError(f"Container too small for ball radius (inradius={R_in:.2f}, radius={radius}, margin={max(clearance, overlap_margin)})")
    
    pts: List[Tuple[float, float]] = []
    for _ in range(count):
        for _try in range(GLOBAL_SCENE_CONSTANTS["placement"]["tries_per_point"]):
            r = random.uniform(0, R_allow)
            ang = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            # Ball-ball minimum distance during placement (loose threshold)
            if all((x - px) ** 2 + (y - py) ** 2 >= (2 * radius + GLOBAL_SCENE_CONSTANTS["placement"]["extra_radius_margin"]) ** 2 for px, py in pts):
                pts.append((x, y))
                break
        else:
            # Strict strategy: if unable to place, raise error for upper layer retry [[memory:7333974]]
            raise RuntimeError("Failed to place non-overlapping ball within polygon incircle after many attempts")
    
    return pts

def generate_irregular_vertices(sides: int, radius: float, irregularity: float) -> List[Tuple[float, float]]:
    """Generate irregular convex polygon vertices"""
    vertices = []
    angle_step = GLOBAL_SCENE_CONSTANTS["math"]["tau"] / sides
    for i in range(sides):
        angle = i * angle_step
        # 对半径施加随机扰动，保持凸性
        r_variation = 1.0 + random.uniform(-irregularity, irregularity)
        r = radius * max(GLOBAL_SCENE_CONSTANTS["geometry"]["irregularity_min_scale"], r_variation)  # 防止过度收缩
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        vertices.append((x, y))
    return vertices

def place_non_overlapping_containers(scene_width: float, scene_height: float, count: int, 
                                   diameter_range: Tuple[float, float],
                                   difficulty: int,
                                   sides_range: Tuple[int, int],
                                   existing: List[Tuple[float, float, float]],
                                   min_gap: float) -> List[Dict]:
    """在场景空间内放置不重叠的多个容器，考虑难度相关的间距调整"""
    if sides_range is None:
        raise RuntimeError("box_sides_range is required for multi-container placement")
    
    containers = []
    existing = existing or []
    
    for i in range(count):
        diameter = float(random.randint(*diameter_range))
        radius = diameter / 2.0
        
        for attempt in range(GLOBAL_SCENE_CONSTANTS["placement"]["container_max_attempts"]):
            # 在场景空间内随机放置，考虑边距
            x = float(random.uniform(min_gap + radius, scene_width - min_gap - radius))
            y = float(random.uniform(min_gap + radius, scene_height - min_gap - radius))
            
            valid = True
            # 与本函数已放容器检测
            for e in containers:
                ex, ey = e["center"]
                er = e["diameter"] / 2.0
                if ((x - ex)**2 + (y - ey)**2)**0.5 < (radius + er + min_gap):
                    valid = False
                    break
            if not valid:
                continue
                
            # 与传入 existing 检测
            if valid:
                for (ex, ey, er) in existing:
                    if ((x - ex)**2 + (y - ey)**2)**0.5 < (radius + er + min_gap):
                        valid = False
                        break
            if valid:
                sides = clamp_sides(sample_int(*sides_range), is_outer=True)
                containers.append({
                    "center": [x, y],
                    "diameter": diameter,
                    "sides": sides,
                    "rotation": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
                    "friction": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
                    "restitution": GLOBAL_SCENE_CONSTANTS["restitution"],
                    "rotation_speed": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
                })
                break
        else:
            # 无退化放置：严格失败
            raise RuntimeError(f"Failed to place container {i} without overlap after {GLOBAL_SCENE_CONSTANTS['placement']['container_max_attempts']} attempts (min_gap={min_gap}, range={diameter_range})")
    return containers

def place_container_with_distance_range(scene_width: float, scene_height: float,
                                      diameter: float,
                                      sides: int,
                                      center_distance_range: Tuple[float, float],
                                      reference_center: Tuple[float, float],
                                      existing: List[Tuple[float, float, float]] = None,
                                      min_gap: float = 0.0) -> Dict:
    """
    Spawn a single container at a position determined by a preset center distance range
    from a reference center point.
    
    Args:
        scene_width: Width of the scene
        scene_height: Height of the scene
        diameter: Diameter of the container to place
        sides: Number of sides for the container
        center_distance_range: (min_distance, max_distance) from reference center
        reference_center: (x, y) coordinates of the reference point
        existing: List of existing containers as (x, y, radius) tuples
        min_gap: Minimum gap between containers
        
    Returns:
        Dictionary representing the placed container
        
    Raises:
        RuntimeError: If unable to place container after maximum attempts
    """
    existing = existing or []
    radius = diameter / 2.0
    ref_x, ref_y = reference_center
    min_distance, max_distance = center_distance_range
    
    max_attempts = GLOBAL_SCENE_CONSTANTS["placement"]["container_max_attempts"]
    
    # Sample distance within the specified range
    distance = random.uniform(min_distance, max_distance)

    # Sample random angle
    angle = random.uniform(0, 2 * math.pi)

    # Calculate position based on distance and angle from reference
    x = ref_x + distance * math.cos(angle)
    y = ref_y + distance * math.sin(angle)

    return {
        "center": [x, y],
        "diameter": diameter,
        "sides": clamp_sides(sides, is_outer=True),
        "rotation": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
        "friction": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
        "restitution": GLOBAL_SCENE_CONSTANTS["restitution"],
        "rotation_speed": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
    }

def _approx_boxes(boxes: List[Dict]) -> List[Tuple[float,float,float]]:
    """近似: 使用中心+半径(=diameter/2)"""
    out = []
    for b in boxes:
        c = b.get("center")
        d = b.get("diameter", 0)
        out.append((c[0], c[1], d/2))
    return out

def _assert_no_overlap(boxes: List[Dict], margin: float = 0.0):
    """严格断言: 近似圆不相交，若相交则抛异常"""
    approx = _approx_boxes(boxes)
    n = len(approx)
    for i in range(n):
        x1,y1,r1 = approx[i]
        for j in range(i+1,n):
            x2,y2,r2 = approx[j]
            dist2 = (x1-x2)**2 + (y1-y2)**2
            limit = (r1 + r2 + margin)
            if dist2 < limit*limit:
                raise RuntimeError(f"Box overlap detected (approx circles) between {i} and {j}")

def check_periodic_motion(velocity: List[float], period: float, container_diameter: float) -> bool:
    """
    检查运动是否具有周期性
    
    Args:
        velocity: 球的初始速度 [vx, vy]
        period: 运动周期
        container_diameter: 容器直径
        
    Returns:
        bool: 是否能确保周期性运动
    """
    vx, vy = velocity
    speed = math.sqrt(vx**2 + vy**2)
    
    # 计算一个周期内球的位移
    displacement_per_period = speed * period
    
    cfg = GLOBAL_SCENE_CONSTANTS["period_check"]
    if speed < cfg["min_speed"]:
        return False
    if period < cfg["min_period"] or period > cfg["max_period"]:
        return False
    if displacement_per_period < container_diameter * cfg["displacement_ratio"]: # FIXME

        return False
    
    # 对于H场景，我们更关注的是运动的基本可行性，而不是严格的周期性
    # 因为即使位移较小，球在容器内的运动仍然可以形成某种周期性模式
    
    return True

def calculate_optimal_period(velocity: List[float], container_diameter: float) -> float:
    """
    计算最优运动周期
    
    Args:
        velocity: 球的初始速度 [vx, vy]
        container_diameter: 容器直径
        
    Returns:
        float: 最优周期
    """
    vx, vy = velocity
    speed = math.sqrt(vx**2 + vy**2)
    
    div = GLOBAL_SCENE_CONSTANTS["period"]["speed_divisor"]
    optimal_period = container_diameter / (speed * div) if speed > 0 else GLOBAL_SCENE_CONSTANTS["period"]["min_period"]
    # 限制在合理范围内
    return max(GLOBAL_SCENE_CONSTANTS["period"]["min_period"], min(GLOBAL_SCENE_CONSTANTS["period"]["max_period"], optimal_period))

def gen_single_axis_scene(letter: str, diff: int, seed: int) -> Dict[str, Any]:
    """
    生成单轴场景
    
    Args:
        letter: 场景类型 (A-G)
        diff: 难度等级 (0-3)
        seed: 随机种子 - 用于确保场景的可重现性。相同的(letter, diff, seed)组合
              会生成完全相同的场景参数，包括：物体位置、速度、边数、角速度等。
              这对测试、调试和基准评估至关重要，确保LLM在相同条件下的表现可比较。
    """
    random.seed(seed)
    meta = generate_scene_metadata(diff, letter, seed)

    # 获取场景空间参数
    scene_width, scene_height = get_scene_dimensions(diff)
    outer_center = list(get_scene_center(diff))
    base_outer_diameter = get_outer_container_diameter(diff)
    physics = {"gravity_enabled": False, "gravity_x": 0.0, "gravity_y": 0.0}

    boxes = []
    balls = []

    if letter == "A":
        r = AXIS_RANGES["A"][diff]
        inner_sides = clamp_sides(sample_int(*r["inner_sides"]))
        ang_low, ang_high = r["ang"]
        rotation_speed = sample_float(ang_low, ang_high)
        vx, vy = sample_velocity(r["lin_speed"])
        
        # 使用新的box参数
        box_diameter = base_outer_diameter
        factor = sample_float(*r["box_diameter_factor"])
        box_diameter = base_outer_diameter * factor
        
        box_sides = clamp_sides(sample_int(*r["box_sides"]), is_outer=True)
        
        # 明确: 球半径需在 AXIS_RANGES 中提供
        ball_radius = float(r["ball_radius"])
        
        ox, oy = random_spawn_offset(ball_radius)
        boxes.append(dict(
            center=outer_center,
            diameter=box_diameter,
            sides=box_sides,
            rotation=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
            friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            rotation_speed=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
        ))
        balls.append(dict(
            position=[outer_center[0] + ox, outer_center[1] + oy],
            velocity=[vx, vy],
            radius=ball_radius,
            density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            color=AXIS_COLORS["A"],
            sides=inner_sides,
            rotation_affected=True,
            inner_rotation_enabled=True,
            angular_velocity=rotation_speed,
            angular_time_variation=False
        ))

    elif letter == "B":
        r = AXIS_RANGES["B"][diff]
        inner_sides = clamp_sides(sample_int(*r["inner_sides"]))
        box_sides = clamp_sides(sample_int(*r["box_sides"]), is_outer=True)
        ang_low, ang_high = r["ang"]
        rotation_speed = sample_float(ang_low, ang_high)
        time_var = r.get("time_var", False)
        if time_var:
            ratio = GLOBAL_SCENE_CONSTANTS["outer_rotation_profile"]["omega_amp_ratio"]
            meta["outer_rotation_profile"] = {"omega_base": rotation_speed, "omega_amp": to_decimal(rotation_speed * ratio)}
        vx, vy = sample_velocity(r["lin_speed"])
        
        # 使用新的box参数
        box_diameter = base_outer_diameter
        factor = sample_float(*r["box_diameter_factor"])
        box_diameter = base_outer_diameter * factor
        
        # 明确: 球半径需在 AXIS_RANGES 中提供
        ball_radius = float(r["ball_radius"])
        
        ox, oy = random_spawn_offset(ball_radius)
        boxes.append(dict(
            center=outer_center,
            diameter=box_diameter,
            sides=box_sides,
            rotation=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
            friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            rotation_speed=rotation_speed
        ))
        balls.append(dict(
            position=[outer_center[0] + ox, outer_center[1] + oy],
            velocity=[vx, vy],
            radius=ball_radius,
            density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            color=AXIS_COLORS["B"],
            sides=inner_sides,
            rotation_affected=False
        ))

    elif letter == "C":
        r = AXIS_RANGES["C"][diff]
        inner_sides = clamp_sides(sample_int(*r["inner_sides"]))
        amp = sample_int(*r["amp"])
        complex_path = r.get("complex_path", False)
        vx, vy = sample_velocity(r["lin_speed"])
        
        # 明确: 球半径
        ball_radius = float(r["ball_radius"])
        
        # 修改: Basic难度也给外框轻微平移
        path = None
        if diff == 0:  # Basic: 微幅静态或极小振荡
            if amp > 0:
                path = {"type": "sin1d", "amplitude": amp, "axis": "x", "freq": GLOBAL_SCENE_CONSTANTS["translation_profiles"]["C"]["basic_freq"]}
        elif diff == 1:  # Easy: 单轴往返
            path = {"type": "sin1d", "amplitude": amp, "axis": "x", "freq": GLOBAL_SCENE_CONSTANTS["translation_profiles"]["C"]["easy_freq"]}
        elif diff == 2:
            path = {"type": "sin1d", "amplitude": amp, "axis": "x", "freq": GLOBAL_SCENE_CONSTANTS["translation_profiles"]["C"]["medium_freq"]}
        elif diff >= 3:  # Medium+: 复杂轨迹
            path = {"type": "lissajous", "ax": amp, "ay": amp, "fx": GLOBAL_SCENE_CONSTANTS["translation_profiles"]["C"]["lissajous_fx"], "fy": GLOBAL_SCENE_CONSTANTS["translation_profiles"]["C"]["lissajous_fy"]}
        if complex_path:
            path = {"type": "lissajous", "ax": amp, "ay": amp, "fx": GLOBAL_SCENE_CONSTANTS["translation_profiles"]["C"]["lissajous_fx"], "fy": GLOBAL_SCENE_CONSTANTS["translation_profiles"]["C"]["lissajous_fy"]}
            
        # 使用新的box参数
        box_diameter = base_outer_diameter
        factor = sample_float(*r["box_diameter_factor"])
        box_diameter = base_outer_diameter * factor
        
        box_sides = clamp_sides(sample_int(*r["box_sides"]), is_outer=True)
        
        ox, oy = random_spawn_offset(ball_radius)
        boxes.append(dict(
            center=outer_center,
            diameter=box_diameter,
            sides=box_sides,
            rotation=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
            friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            rotation_speed=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"],
            translation_path=path
        ))
        balls.append(dict(
            position=[outer_center[0] + ox, outer_center[1] + oy],
            velocity=[vx, vy],
            radius=ball_radius,
            density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            color=AXIS_COLORS["C"],
            sides=inner_sides,
            rotation_affected=False  # 修改: 禁用旋转，确保轴纯度
        ))

    elif letter == "D":
        r = AXIS_RANGES["D"][diff]
        inner_sides = clamp_sides(sample_int(*r["inner_sides"]))
        vx, vy = sample_velocity(r["lin_speed"])
        gx, gy, enabled, gmeta = gravity_vector(r["g_mode"])
        physics.update(dict(gravity_enabled=enabled, gravity_x=gx, gravity_y=gy))
        if gmeta:
            meta["gravity_profile"] = gmeta
        
        # 明确: 球半径
        ball_radius = float(r["ball_radius"])
        
        # 使用新的box参数
        box_diameter = base_outer_diameter
        factor = sample_float(*r["box_diameter_factor"])
        box_diameter = base_outer_diameter * factor
        
        box_sides = clamp_sides(sample_int(*r["box_sides"]), is_outer=True)
        
        ox, oy = random_spawn_offset(ball_radius)
        boxes.append(dict(
            center=outer_center, diameter=box_diameter, sides=box_sides,
            rotation=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"], friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"], restitution=GLOBAL_SCENE_CONSTANTS["restitution"], rotation_speed=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
        ))
        balls.append(dict(
            position=[outer_center[0] + ox, outer_center[1] + oy],
            velocity=[vx, vy],
            radius=ball_radius,
            density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            color=AXIS_COLORS["D"],
            sides=inner_sides,
            rotation_affected=False  
        ))

    elif letter == "E":
        r = AXIS_RANGES["E"][diff]
        inner_sides = clamp_sides(sample_int(*r["inner_sides"]))
        irregularity = r["irregularity"]
        vx, vy = sample_velocity(r["lin_speed"])
        
        # 明确: 球半径
        ball_radius = float(r["ball_radius"])
        
        # 使用新的box参数
        box_diameter = base_outer_diameter
        factor = sample_float(*r["box_diameter_factor"])
        box_diameter = base_outer_diameter * factor
        
        box_sides = clamp_sides(sample_int(*r["box_sides"]), is_outer=True)
        
        ox, oy = random_spawn_offset(ball_radius)
        boxes.append(dict(
            center=outer_center, diameter=box_diameter, sides=box_sides,
            rotation=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"], friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"], restitution=GLOBAL_SCENE_CONSTANTS["restitution"], rotation_speed=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
        ))
        
        # 生成不规则内部多边形顶点（用于记录，引擎仍用规则多边形近似） FIXME
        # 半径采用球半径范围中值
        irregular_vertices = generate_irregular_vertices(inner_sides, ball_radius, irregularity)
        
        ox, oy = random_spawn_offset(ball_radius)
        balls.append(dict(
            position=[outer_center[0] + ox, outer_center[1] + oy],
            velocity=[vx, vy],
            radius=ball_radius,
            density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            color=AXIS_COLORS["E"],
            sides=inner_sides,  # 引擎用规则多边形
            # 记录设计意图
            irregular_design=True,
            target_vertices=irregular_vertices,
            irregularity_factor=irregularity,
            rotation_affected=False  # 修改: 禁用旋转，确保轴纯度
        ))
        meta["irregular_inner"] = {
            "implementation": "regular_polygon_approximation",
            "target_design": "irregular_convex_polygon",
            "irregularity_factor": irregularity,
            "target_vertices": irregular_vertices
        }

    elif letter == "F":
        r = AXIS_RANGES["F"][diff]
        inner_sides = clamp_sides(sample_int(*r["inner_sides"]))
        vx, vy = sample_velocity(r["lin_speed"])
        ball_radius = float(r["ball_radius"])  # 明确半径，避免未定义
        
        # 生成多个容器 - 使用距离范围放置
        container_count = r["container_count"] if isinstance(r["container_count"], int) else sample_int(*r["container_count"])

        min_factor, max_factor = r["box_diameter_factor"]
        container_diameter_range = (int(base_outer_diameter * min_factor), int(base_outer_diameter * max_factor))

        # 使用场景中心作为参考点
        reference_center = (scene_width / 2.0, scene_height / 2.0)

        containers = []
        existing_containers = []  # Track placed containers for collision detection
        min_gap = get_container_spacing(diff)

        for i in range(container_count):
            # Sample diameter for this container
            diameter = float(random.randint(*container_diameter_range))
            sides = clamp_sides(sample_int(*r["box_sides"]), is_outer=True)

            if i == 0:
                # First container placed exactly at center
                container = {
                    "center": list(reference_center),
                    "diameter": diameter,
                    "sides": sides,
                    "rotation": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
                    "friction": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
                    "restitution": GLOBAL_SCENE_CONSTANTS["restitution"],
                    "rotation_speed": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
                }
            else:
                # Subsequent containers use distance range
                base_distance = 250.0  # Base distance from center
                distance_increment = 0.0  # Distance increment per container
                min_distance = base_distance + i * distance_increment
                max_distance = min_distance + 30.0

                # Ensure max_distance doesn't exceed scene bounds (rough approximation)
                max_possible_distance = min(scene_width, scene_height) / 2.0 - diameter / 2.0 - min_gap
                max_distance = min(max_distance, max_possible_distance)

                if min_distance >= max_distance:
                    min_distance = max_distance - 10.0  # Ensure valid range

                container = place_container_with_distance_range(
                    scene_width, scene_height, diameter, sides,
                    (min_distance, max_distance), reference_center,
                    existing_containers, min_gap)

            containers.append(container)
            # Add to existing list for next container placement
            existing_containers.append((container["center"][0], container["center"][1], container["diameter"] / 2.0))
        
        # 添加所有容器
        boxes.extend(containers)
        
        # 在第一个容器中放置球
        if containers:
            first_container = containers[0]
            cx, cy = first_container["center"]
            
            ox, oy = random_spawn_offset(ball_radius)
            balls.append(dict(
                position=[cx + ox, cy + oy],
                velocity=[vx, vy],
                radius=ball_radius,
                density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
                restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
                color=AXIS_COLORS["F"],
                sides=inner_sides,
                rotation_affected=False  # 修改: 禁用旋转，确保轴纯度
            ))
        
        meta["multiple_containers"] = {
            "container_count": len(containers),
            "container_layout": "non_overlapping",
            "interaction_rule": "no_container_collisions"
        }

    elif letter == "G":
        r = AXIS_RANGES["G"][diff]
        count = sample_int(*r["count"])
        sides_lo, sides_hi = r["inner_sides"]
        speed_range = r["lin_speed"]
        
        # 使用新的box参数
        box_diameter = base_outer_diameter
        factor = sample_float(*r["box_diameter_factor"])
        box_diameter = base_outer_diameter * factor
        
        box_sides = clamp_sides(sample_int(*r["box_sides"]), is_outer=True)
        
        ball_radius = float(r["ball_radius"])
        
        ox, oy = random_spawn_offset(ball_radius)
        boxes.append(dict(
            center=outer_center,
            diameter=box_diameter,
            sides=box_sides,
            rotation=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
            friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            rotation_speed=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
        ))
        centers = place_inside(outer_center, box_diameter, int(count), radius=ball_radius, difficulty=diff, sides=box_sides)
        for i, (x, y) in enumerate(centers):
            v = sample_float(*speed_range) if speed_range[1] > 0 else 0.0
            theta = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
            ox, oy = random_spawn_offset(ball_radius)
            balls.append(dict(
                position=[x + ox, y + oy],
                velocity=[v * math.cos(theta), v * math.sin(theta)],
                radius=ball_radius,
                density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
                restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
                color=[float(random.randint(*COLOR_RANDOM_RANGE)), float(random.randint(*COLOR_RANDOM_RANGE)), float(random.randint(*COLOR_RANDOM_RANGE))],
                sides=clamp_sides(sample_int(sides_lo, sides_hi)),
                rotation_affected=False
            ))
        meta["multi_object_count"] = count

    elif letter == "H":
        # 导入可预测配置模块
        from .predictable_configs import calculate_theoretical_period
        
        # 统一采用偶数边，确保直线往返
        hcfg = GLOBAL_SCENE_CONSTANTS["H"][diff]
        N = hcfg["N"]
        if N % 2 == 1:
            N += 1
        box_sides = N
        inner_sides = N
        
        # 同中心
        box_diameter = base_outer_diameter
        initial_pos = [outer_center[0], outer_center[1]]
        
        # 初速度沿未旋转容器的一对平行边法向：法向角为 pi/N
        speed = hcfg["speed"]
        theta = math.pi / box_sides
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
        
        # 周期（精确一维公式，中心到中心）
        period = calculate_theoretical_period([vx, vy], box_diameter, box_sides, ball_radius=hcfg["ball_radius"])
        
        boxes.append(dict(
            center=outer_center,
            diameter=box_diameter,
            sides=box_sides,
            rotation=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
            friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            rotation_speed=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
        ))
        
        ox, oy = random_spawn_offset(hcfg["ball_radius"])
        balls.append(dict(
            position=[outer_center[0] + ox, outer_center[1] + oy],
            velocity=[vx, vy],
            radius=hcfg["ball_radius"],
            density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            color=AXIS_COLORS["H"],
            sides=inner_sides,
            rotation_affected=False,
            inner_rotation_enabled=False,
            angular_velocity=0.0,
            angular_time_variation=False
        ))
        
        meta["periodic_motion"] = {
            "period": period,
            "period_type": "linear_1d",
            "chaotic": False,
            "velocity": [round(vx, 2), round(vy, 2)],
            "speed": speed,
            "config_name": f"regular_{N}_same_center_normal",
            "theoretical_period": period
        }
        key_div = GLOBAL_SCENE_CONSTANTS["period"]["key_divisions"]
        meta["key_timestamps"] = [round(i * period / key_div, 3) for i in range(key_div+1)]

    elif letter == "I":
        # I 场景：偶数边，存在水平边；速度竖直（垂直于水平边），面-面往返
        from .predictable_configs import get_predictable_i_config_by_period

        # 选择偶数边数与速度（与 H 保持一致的等级）
        icfg = GLOBAL_SCENE_CONSTANTS["I"][diff]
        N = icfg["N"]
        speed = icfg["speed"]

        # 选择两位小数的目标周期（示例取值，保证可读）
        target_period = icfg["target_period"]

        # 由周期反推直径（允许多位小数），速度取竖直向下（y 正方向）
        cfg = get_predictable_i_config_by_period(
            container_sides=N,
            speed=speed,
            period=target_period,
            ball_radius=icfg["ball_radius"],
            upward=icfg["upward"],
        )

        ox, oy = random_spawn_offset(icfg["ball_radius"])
        boxes.append(dict(
            center=outer_center,
            diameter=cfg["container_diameter"],
            sides=N,
            rotation=cfg["rotation"],
            friction=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            rotation_speed=GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
        ))

        ox, oy = random_spawn_offset(icfg["ball_radius"])
        balls.append(dict(
            position=[outer_center[0] + ox, outer_center[1] + oy],
            velocity=cfg["velocity"],
            radius=cfg["ball_radius"],
            density=GLOBAL_SCENE_CONSTANTS["defaults"]["ball"]["density"],
            restitution=GLOBAL_SCENE_CONSTANTS["restitution"],
            color=AXIS_COLORS["I"],
            sides=N,
            rotation_affected=False,
            inner_rotation_enabled=False,
            angular_velocity=0.0,
            angular_time_variation=False
        ))

        meta["periodic_motion"] = {
            "period": target_period,
            "period_type": "linear_1d",
            "chaotic": False,
            "velocity": cfg["velocity"],
            "speed": speed,
            "config_name": f"regular_{N}_I_vertical_face_bounce",
            "theoretical_period": cfg["period"]
        }
        key_div = GLOBAL_SCENE_CONSTANTS["period"]["key_divisions"]
        meta["key_timestamps"] = [round(i * target_period / key_div, 3) for i in range(key_div+1)]

    scene = {
        "physics": physics,
        "boxes": boxes,
        "balls": balls,
        "meta": meta
    }
    # 最终写出前进行球重叠断言（复用全局 overlap 参数）
    _assert_no_ball_overlap(balls, margin=GLOBAL_SCENE_CONSTANTS["overlap_check_margin"])
    _assert_balls_inside_containers(balls, boxes)
    scene = normalize_scene_integers(scene)
    return scene

# ---------- Envelope 组合支持 新增 开始 ----------

def merge_range(a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[int,int]:
    return (min(a[0], b[0]), max(a[1], b[1]))

def pick_gravity_mode(modes: List[str]) -> str:
    if not modes: return "none"
    priority = GLOBAL_SCENE_CONSTANTS["gravity_modes_priority"]
    for p in priority:
        if p in modes: return p
    return modes[0]

def build_envelope(features: List[str], diff: int) -> Dict[str, Any]:
    feats = sorted(set(features))
    env = {
        "axes": feats,
        "inner_sides_range": None,
        "box_sides_range": None,
        "inner_rot_ang_range": None,
        "outer_rot_ang_range": None,
        "translation_amp_range": None,
        "irregularity_range": None,
        "container_count_range": None,
        "box_sides_range": None,
        "multi_object_count_range": None,
        "gravity_modes": [],
        "box_diameter_factor_range": None,
        "flags": set()
    }
    for ax in feats:
        r = AXIS_RANGES[ax][diff]
        # inner_sides
        if "inner_sides" in r:
            rng = _as_pair(r["inner_sides"])
            env["inner_sides_range"] = rng if env["inner_sides_range"] is None else merge_range(env["inner_sides_range"], rng)
        # box_sides (outer frame sides)
        if "box_sides" in r:
            rng = _as_pair(r["box_sides"])
            env["box_sides_range"] = rng if env["box_sides_range"] is None else merge_range(env["box_sides_range"], rng)
        # rotation (A inner / B outer)
        if ax == "A":
            rng = _as_pair(r["ang"]) if "ang" in r else None
            if rng is not None:
                env["inner_rot_ang_range"] = rng if env["inner_rot_ang_range"] is None else merge_range(env["inner_rot_ang_range"], rng)
            if r.get("time_var"): env["flags"].add("inner_time_var")
        if ax == "B":
            rng = _as_pair(r["ang"]) if "ang" in r else None
            if rng is not None:
                env["outer_rot_ang_range"] = rng if env["outer_rot_ang_range"] is None else merge_range(env["outer_rot_ang_range"], rng)
            if r.get("time_var"): env["flags"].add("outer_time_var")
            if r.get("random_jumps"): env["flags"].add("outer_random_jumps")
        # translation
        if ax == "C":
            if "amp" in r:
                rng = _as_pair(r["amp"])
                env["translation_amp_range"] = rng if env["translation_amp_range"] is None else merge_range(env["translation_amp_range"], rng)
            if r.get("complex_path"): env["flags"].add("complex_path")
            if r.get("chaotic_path"): env["flags"].add("chaotic_path")
            if r.get("multi_freq"): env["flags"].add("multi_freq_path")
        # gravity
        if ax == "D":
            env["gravity_modes"].append(r["g_mode"])
        # irregular
        if ax == "E":
            irr = float(r["irregularity"]) if "irregularity" in r else None
            if irr is not None:
                if env["irregularity_range"] is None:
                    env["irregularity_range"] = (irr, irr)
                else:
                    lo, hi = env["irregularity_range"]
                    env["irregularity_range"] = (min(lo, irr), max(hi, irr))
        # multiple containers
        if ax == "F":
            if "container_count" in r:
                cc_rng = _as_pair(r["container_count"])
                env["container_count_range"] = cc_rng if env["container_count_range"] is None else merge_range(env["container_count_range"], cc_rng)
            if "box_sides" in r:
                cs_rng = _as_pair(r["box_sides"])
                env["box_sides_range"] = cs_rng if env["box_sides_range"] is None else merge_range(env["box_sides_range"], cs_rng)
            # 并入 box_diameter_factor 区间
            if "box_diameter_factor" in r:
                bf_rng = _as_pair(r["box_diameter_factor"])
                env["box_diameter_factor_range"] = bf_rng if env["box_diameter_factor_range"] is None else merge_range(env["box_diameter_factor_range"], bf_rng)
                # 新增：为F轴保留独立的尺寸区间，供含F组合时专用
                env["box_diameter_factor_range_F"] = bf_rng
        # multi objects
        if ax == "G":
            mo_rng = _as_pair(r["count"]) if "count" in r else None
            if mo_rng is not None:
                env["multi_object_count_range"] = mo_rng if env["multi_object_count_range"] is None else merge_range(env["multi_object_count_range"], mo_rng)
        # diameter factor (new unified sizing control)
        if "box_diameter_factor" in r:
            bf_rng = _as_pair(r["box_diameter_factor"])
            env["box_diameter_factor_range"] = bf_rng if env["box_diameter_factor_range"] is None else merge_range(env["box_diameter_factor_range"], bf_rng)
        # periodic motion
        if ax == "H":
            env["has_H"] = True
            env["H_difficulty"] = diff
    env["gravity_mode_selected"] = pick_gravity_mode(env["gravity_modes"])
    return env

def _sample_outer_translation(env: Dict[str,Any], diff: int):
    amp_range = env.get("translation_amp_range")
    if not amp_range:
        return None
    lo, hi = amp_range
    amp = sample_float(lo, hi)
    # 选择路径类型 (优先复杂标志)
    if "chaotic_path" in env["flags"]:
        return {"type":"piecewise","segments":6,"amplitude":amp}
    if "complex_path" in env["flags"]:
        return {"type":"lissajous","ax":amp,"ay":amp,
                "fx":0.6,"fy":0.9}
    # 基础
    low_max = GLOBAL_SCENE_CONSTANTS["outer_translation"]["low_diff_max"]
    freq = GLOBAL_SCENE_CONSTANTS["outer_translation"]["freq_low_diff"] if diff <= low_max else GLOBAL_SCENE_CONSTANTS["outer_translation"]["freq_high_diff"]
    return {"type":"sin1d","amplitude":amp,"axis":"x","freq":freq}

def _apply_gravity(physics: Dict[str,Any], mode: str, meta: Dict[str,Any]):
    gx=0.0; gy=-GLOBAL_SCENE_CONSTANTS["gravity"]["g"]; enabled=False
    if mode == "none": 
        physics.update(dict(gravity_enabled=False, gravity_x=0.0, gravity_y=0.0))
        return
    enabled=True
    if mode == "tiny":
        gy = -GLOBAL_SCENE_CONSTANTS["gravity"]["tiny"]
    elif mode == "small":
        gy = -GLOBAL_SCENE_CONSTANTS["gravity"]["small"]
    elif mode == "large":
        gy = -GLOBAL_SCENE_CONSTANTS["gravity"]["large"]
    elif mode == "tilted":
        angle = random.uniform(-math.radians(GLOBAL_SCENE_CONSTANTS["gravity"]["tilt_max_deg"]), math.radians(GLOBAL_SCENE_CONSTANTS["gravity"]["tilt_max_deg"]))
        g=GLOBAL_SCENE_CONSTANTS["gravity"]["large"]
        gx, gy = g*math.sin(angle), -g*math.cos(angle)
        meta["gravity_profile"]={"tilt_angle_deg":round(math.degrees(angle),2)}
    elif mode == "chaotic":
        ang = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
        mag = random.uniform(*GLOBAL_SCENE_CONSTANTS["gravity"]["chaotic_mag"])
        gx, gy = mag*math.cos(ang), mag*math.sin(ang)
        meta["gravity_profile"]={"chaotic_gravity":True,"magnitude":round(mag,2),"base_angle":round(math.degrees(ang),2),
                                 "time_variation":{"amp":round(GLOBAL_SCENE_CONSTANTS["gravity"]["time_variation"]["chaotic_amp_ratio"]*mag,2),"freq":random.uniform(*GLOBAL_SCENE_CONSTANTS["gravity"]["time_variation"]["chaotic_freq_range"])}}
    elif mode == "extreme_chaotic":
        ang = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
        mag = random.uniform(*GLOBAL_SCENE_CONSTANTS["gravity"]["extreme_chaotic_mag"])
        gx, gy = mag*math.cos(ang), mag*math.sin(ang)
        meta["gravity_profile"]={"extreme_chaotic_gravity":True,"magnitude":round(mag,2),"base_angle":round(math.degrees(ang),2),
                                 "time_variation":{"amp":round(GLOBAL_SCENE_CONSTANTS["gravity"]["time_variation"]["extreme_amp_ratio"]*mag,2),"freq":random.uniform(*GLOBAL_SCENE_CONSTANTS["gravity"]["time_variation"]["extreme_freq_range"])}}
    physics.update(dict(gravity_enabled=enabled, gravity_x=round(gx,2), gravity_y=round(gy,2)))

def generate_from_envelope(features: List[str], diff: int, seed: int) -> Dict[str,Any]:
    random.seed(seed)
    env = build_envelope(features, diff)
    base_meta = generate_scene_metadata(diff, "".join(sorted(set(features))), seed)
    meta = base_meta.copy()
    meta.update({
        "envelope": True,
        "envelope_axes": env["axes"],
        "order_invariant": True,
        "envelope_ranges": {}
    })
    physics = {"gravity_enabled":False,"gravity_x":0.0,"gravity_y":0.0}
    # 记录主要区间
    for k in ("inner_sides_range","box_sides_range","inner_rot_ang_range","outer_rot_ang_range",
              "translation_amp_range","irregularity_range","container_count_range",
              "multi_object_count_range","box_diameter_factor_range"):
        if env.get(k):
            meta["envelope_ranges"][k] = env[k]
    # 获取场景空间参数
    scene_width, scene_height = get_scene_dimensions(diff)
    outer_center = list(get_scene_center(diff))
    outer_diameter_base = get_outer_container_diameter(diff)  # simplified baseline = min(width,height)
    boxes=[]; balls=[]
    has_F = "F" in env["axes"]
    has_G = "G" in env["axes"]
    has_H = "H" in env["axes"]
    has_E = "E" in env["axes"]
    has_A = "A" in env["axes"]
    has_B = "B" in env["axes"]
    has_C = "C" in env["axes"]

    # 结构: F 优先
    if has_F:
        cc_range = env.get("container_count_range")
        if isinstance(cc_range, tuple):
            cc_lo, cc_hi = cc_range
            container_count = sample_int(cc_lo, cc_hi)
        else:
            container_count = int(cc_range) if isinstance(cc_range, int) else 1

        # 使用F轴专属的尺寸因子（若存在），避免被其它轴放大导致无法放置
        factor_range = env.get("box_diameter_factor_range_F") or env.get("box_diameter_factor_range")
        f_lo, f_hi = factor_range
        base_d = int(outer_diameter_base * f_lo)
        max_d = int(outer_diameter_base * f_hi)
        # 间距参数
        min_gap = get_container_spacing(diff)

        # 使用距离范围放置多个容器
        reference_center = (scene_width / 2.0, scene_height / 2.0)
        containers = []
        existing_containers = []  # Track placed containers for collision detection

        for i in range(container_count):
            # Sample diameter for this container
            diameter = float(random.randint(base_d, max_d))
            sides_range = env.get("box_sides_range")
            if sides_range:
                sides = clamp_sides(sample_int(*sides_range), is_outer=True)
            else:
                sides = clamp_sides(sample_int(3, 8), is_outer=True)  # Default range

            if i == 0:
                # First container placed exactly at center
                container = {
                    "center": list(reference_center),
                    "diameter": diameter,
                    "sides": sides,
                    "rotation": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation"],
                    "friction": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["friction"],
                    "restitution": GLOBAL_SCENE_CONSTANTS["restitution"],
                    "rotation_speed": GLOBAL_SCENE_CONSTANTS["defaults"]["box"]["rotation_speed"]
                }
            else:
                # Subsequent containers use distance range
                base_distance = 250.0  # Base distance from center
                distance_increment = 0.0  # Distance increment per container
                min_distance = base_distance + i * distance_increment
                max_distance = min_distance + 30.0

                # Ensure max_distance doesn't exceed scene bounds (rough approximation)
                max_possible_distance = min(scene_width, scene_height) / 2.0 - diameter / 2.0 - min_gap
                max_distance = min(max_distance, max_possible_distance)

                if min_distance >= max_distance:
                    min_distance = max_distance - 10.0  # Ensure valid range

                container = place_container_with_distance_range(
                    scene_width, scene_height, diameter, sides,
                    (min_distance, max_distance), reference_center,
                    existing_containers, min_gap)

            containers.append(container)
            # Add to existing list for next container placement
            existing_containers.append((container["center"][0], container["center"][1], container["diameter"] / 2.0))
        boxes.extend(containers)
        meta["multiple_containers"]={"container_count":len(containers),"layout":"envelope_non_overlap"}
    else:
        # 单外框：使用统一 factor 控制尺寸（去除默认，严格校验）
        sides_lo, sides_hi = _require_pair(env.get("box_sides_range"), "box_sides_range")
        sides = clamp_sides(sample_int(sides_lo, sides_hi), is_outer=True)
        box_sides = sides

        f_lo, f_hi = _require_pair(env.get("box_diameter_factor_range"), "box_diameter_factor_range")
        factor = sample_float(f_lo, f_hi)
        box_diameter = outer_diameter_base * factor

        boxes.append(dict(center=outer_center,diameter=box_diameter,sides=box_sides,rotation=0.0,
                          friction=0.0,restitution=GLOBAL_SCENE_CONSTANTS["restitution"],rotation_speed=0.0))

    # 外旋 (B)
    if has_B and env.get("outer_rot_ang_range"):
        ang_lo, ang_hi = _require_pair(env.get("outer_rot_ang_range"), "outer_rot_ang_range")
        omega = sample_float(ang_lo, ang_hi)
        if has_F:
            for c in boxes:
                c["rotation_speed"]=omega
        else:
            boxes[0]["rotation_speed"]=omega
        if "outer_time_var" in env["flags"]:
            ratio = GLOBAL_SCENE_CONSTANTS["outer_rotation_profile"]["omega_amp_ratio"]
            meta["outer_rotation_profile"]={"omega_base":omega,"omega_amp":round(ratio*omega,2)}

    # 外平移 (C)
    translation_path=_sample_outer_translation(env, diff) if has_C else None
    if translation_path:
        if has_F:
            for c in boxes: c["translation_path"]=translation_path
            meta["rigid_group_motion"]=True
        else:
            boxes[0]["translation_path"]=translation_path

    # 重力 (D)
    if env["gravity_mode_selected"] != "none":
        _apply_gravity(physics, env["gravity_mode_selected"], meta)
        meta["gravity_mode_selected"]=env["gravity_mode_selected"]

    # 球数量 (G / else) - 去除默认，若声明了 G 必须提供
    if has_G:
        mo_lo, mo_hi = _require_pair(env.get("multi_object_count_range"), "multi_object_count_range")
        cnt = sample_int(mo_lo, mo_hi)
    else:
        cnt = 1
    
    # 生成球 - 修改：为每个球分配不同的初始位置
    inner_sides_range = env.get("inner_sides_range")
    if inner_sides_range is None:
        raise ValueError("Missing required envelope range: inner_sides_range")
    inner_lo, inner_hi = _require_pair(inner_sides_range, "inner_sides_range")
    
    # 计算球的初始位置分布
    ball_positions = []
    if has_F:
        # 多容器情况：将球分散到不同容器中
        for i in range(cnt):
            container_idx = i % len(boxes)
            container = boxes[container_idx]
            container_center = container["center"]
            container_radius = container["diameter"] / 2
            
            # 在容器内随机选择一个位置，避免重叠
            if i == 0:
                # 第一个球放在容器中心
                pos = [container_center[0], container_center[1]]
            else:
                # 后续球在容器内随机分布，但避免重叠
                attempts = 0
                max_attempts = GLOBAL_SCENE_CONSTANTS["placement"]["ball_max_attempts"]
                while attempts < max_attempts:
                    # 在容器内随机选择位置
                    angle = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
                    # 距离中心的距离：使用"多球径向分布比例范围"
                    radial_lo, radial_hi = GLOBAL_SCENE_CONSTANTS["spawn_random"]["multi_object_radial_fraction_range"]
                    distance = random.uniform(container_radius * radial_lo, container_radius * radial_hi)
                    pos = [
                        container_center[0] + distance * math.cos(angle),
                        container_center[1] + distance * math.sin(angle)
                    ]
                    
                    # 检查是否与之前的球重叠
                    overlap = False
                    threshold = GLOBAL_SCENE_CONSTANTS["placement"]["overlap_distance_threshold"]
                    for prev_pos in ball_positions:
                        if calculate_distance(pos, prev_pos) < threshold:
                            overlap = True
                            break
                    
                    if not overlap:
                        break
                    attempts += 1
                
                if attempts >= max_attempts:
                    # 严格模式：找不到不重叠位置则抛错，交给上层重试
                    raise RuntimeError("Failed to place non-overlapping ball inside container after many attempts")
            
            ball_positions.append(pos)
    else:
        # 单容器情况：在容器内分散分布
        container_center = outer_center
        container_radius = outer_diameter_base / 2
        
        for i in range(cnt):
            if i == 0:
                # 第一个球放在容器中心
                pos = [container_center[0], container_center[1]]
            else:
                # 后续球在容器内随机分布，但避免重叠
                attempts = 0
                max_attempts = GLOBAL_SCENE_CONSTANTS["placement"]["ball_max_attempts"]
                while attempts < max_attempts:
                    # 在容器内随机选择位置
                    angle = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
                    # 距离中心的距离：使用"多球径向分布比例范围"
                    radial_lo, radial_hi = GLOBAL_SCENE_CONSTANTS["spawn_random"]["multi_object_radial_fraction_range"]
                    distance = random.uniform(container_radius * radial_lo, container_radius * radial_hi)
                    pos = [
                        container_center[0] + distance * math.cos(angle),
                        container_center[1] + distance * math.sin(angle)
                    ]
                    
                    # 检查是否与之前的球重叠
                    overlap = False
                    threshold = GLOBAL_SCENE_CONSTANTS["placement"]["overlap_distance_threshold"]
                    for prev_pos in ball_positions:
                        if calculate_distance(pos, prev_pos) < threshold:
                            overlap = True
                            break
                    
                    if not overlap:
                        break
                    attempts += 1
                
                if attempts >= max_attempts:
                    # 严格模式：找不到不重叠位置则抛错，交给上层重试
                    raise RuntimeError("Failed to place non-overlapping ball in single container after many attempts")
            
            ball_positions.append(pos)
    
    # 创建球对象
    for i in range(cnt):
        inner_sides = clamp_sides(sample_int(inner_lo, inner_hi))
        if env.get("axes"):
            candidate_min_speeds = [AXIS_RANGES[ax][diff]["lin_speed"][0] for ax in env["axes"] if "lin_speed" in AXIS_RANGES[ax][diff]]
            candidate_max_speeds = [AXIS_RANGES[ax][diff]["lin_speed"][1] for ax in env["axes"] if "lin_speed" in AXIS_RANGES[ax][diff]]
            if not candidate_max_speeds or not candidate_min_speeds:
                raise ValueError("No available lin_speed from env axes to determine velocity range")
            vmin = min(candidate_min_speeds)
            vmax = max(candidate_max_speeds)
        else:
            raise ValueError("env.axes is required to determine vmax")
        vx, vy = sample_velocity((vmin, vmax))
        
        # 使用计算好的位置
        pos = ball_positions[i]
        
        # 使用统一半径：取 env 轴中可用 ball_radius 的最小值；若无则报错
        if not env.get("axes"):
            raise ValueError("env.axes is required to determine ball radius")
        candidate_radii = [AXIS_RANGES[ax][diff]["ball_radius"] for ax in env["axes"] if "ball_radius" in AXIS_RANGES[ax][diff]]
        if not candidate_radii:
            raise ValueError("No available ball_radius from env axes to determine radius")
        env_ball_radius = min(candidate_radii)
        ball = dict(position=pos, velocity=[vx, vy], radius=env_ball_radius, density=1.0,
                    restitution=GLOBAL_SCENE_CONSTANTS["restitution"], color=[random.randint(*COLOR_RANDOM_RANGE),random.randint(*COLOR_RANDOM_RANGE),random.randint(*COLOR_RANDOM_RANGE)],
                    sides=inner_sides, rotation_affected=False)
        balls.append(ball)

    # 内旋 (A) 给所有球或指定球 - 若有G轴则所有球都旋转，否则只给第一个球
    if has_A:
        lo, hi = _require_pair(env.get("inner_rot_ang_range"), "inner_rot_ang_range")
        omega = sample_float(lo, hi)

        # 如果有G轴（多球），所有球都旋转；否则只第一个球旋转
        rotation_balls = balls if has_G else [balls[0]]

        for i, ball in enumerate(rotation_balls):
            ball.update(dict(
                rotation_affected=True,
                inner_rotation_enabled=True,
                angular_velocity=omega,
                angular_time_variation=False
            ))

        # 更新元数据以反映哪些球获得了旋转
        if has_G:
            meta["inner_rotation_assigned_balls"] = list(range(len(balls)))
        else:
            meta["inner_rotation_assigned_ball"] = 0

    # 不规则 (E) 给一个球 - 去除默认，若 has_E 则必须提供 irregularity_range
    if has_E:
        irr_lo, irr_hi = _require_pair(env.get("irregularity_range"), "irregularity_range")
        irr = round(random.uniform(irr_lo, irr_hi),2)
        b = balls[-1]
        # 仅元数据记录（实际仍规则）
        b.update(dict(irregular_design=True, irregularity_factor=irr,
                      target_vertices=generate_irregular_vertices(b["sides"], b["radius"], irr)))
        meta["irregular_inner"]={"assigned_ball":len(balls)-1,"irregularity_factor":irr}

    # 多容器旋转/平移标志
    if has_F:
        meta.setdefault("multi_container_base", True)
        if translation_path: meta["rigid_group_motion"]=True

    # 周期性运动 (H) - 在组合场景中应用H的特性
    if has_H and env.get("has_H"):
        from .predictable_configs import calculate_theoretical_period
        
        # 仅支持同中心、同角度、偶数边
        N = 6 if diff == 0 else 8 if diff == 1 else 10
        if N % 2 == 1:
            N += 1
        
        # 统一设置单外框
        for box in boxes[:1]:
            box["sides"] = N
            box["center"] = outer_center
            box["rotation"] = 0.0
        if not boxes:
            boxes.append(dict(center=outer_center, diameter=outer_diameter_base, sides=N, rotation=0.0,
                              friction=0.0, restitution=GLOBAL_SCENE_CONSTANTS["restitution"], rotation_speed=0.0))
        
        # 只保留一个球，并设置同中心
        balls[:] = balls[:1] or [dict(position=outer_center, velocity=[0.0,0.0], radius=25, density=1.0,
                                      restitution=GLOBAL_SCENE_CONSTANTS["restitution"], color=AXIS_COLORS.get("H"), sides=N, rotation_affected=False)]
        balls[0]["position"] = list(outer_center)
        balls[0]["sides"] = N
        
        # 速度沿未旋转容器一对平行边的法向（角度 pi/N） FIXME
        speed = 200.0 if diff == 0 else 300.0 if diff == 1 else 400.0
        theta = math.pi / N
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
        balls[0]["velocity"] = [vx, vy]
        
        # 周期
        container_diameter = boxes[0]["diameter"]
        period = calculate_theoretical_period([vx, vy], container_diameter, N, ball_radius=25.0)
        
        # ABH 同步旋转：若包含 A 或 B，则设置 omega 使 omega*T = 2π
        sync = False
        omega = 0.0
        if (has_A or has_B) and period > 0:
            omega = GLOBAL_SCENE_CONSTANTS["math"]["tau"] / period  # k=1
            sync = True
            # A: 给球内旋同步
            if has_A:
                balls[0].update(dict(rotation_affected=True, inner_rotation_enabled=True, angular_velocity=omega))
            # B: 给外框同步旋转
            if has_B:
                for box in boxes:
                    box["rotation_speed"] = omega
        
        meta["periodic_motion"] = {
            "period": period,
            "period_type": "linear_1d",
            "chaotic": False,
            "velocity": [round(vx, 2), round(vy, 2)],
            "speed": speed,
            "config_name": f"regular_{N}_same_center_normal",
            "theoretical_period": period,
            "synchronized_rotation": sync,
            "omega": omega
        }
        key_div = GLOBAL_SCENE_CONSTANTS["period"]["key_divisions"]
        meta["key_timestamps"] = [round(i * period / key_div, 3) for i in range(key_div+1)]

    # 重叠检测
    # _assert_no_overlap(boxes, margin=GLOBAL_SCENE_CONSTANTS["overlap_check_margin"])
    _assert_no_ball_overlap(balls, margin=GLOBAL_SCENE_CONSTANTS["overlap_check_margin"])
    _assert_balls_inside_containers(balls, boxes)

    scene = {"physics": physics, "boxes": boxes, "balls": balls, "meta": meta}
    scene = normalize_scene_integers(scene)
    return scene

def generate_scene_metadata(difficulty: int, scene_type: str, seed: int) -> Dict[str, Any]:
    """生成场景的元数据，包括场景空间信息"""
    scene_width, scene_height = get_scene_dimensions(difficulty)
    
    return {
        "scene_type": scene_type,
        "difficulty": DIFF_NAMES[difficulty],
        "seed": seed,
        "scene_space": {
            "width": scene_width,
            "height": scene_height,
            "center": get_scene_center(difficulty),
            "units": SCENE_SPACE_CONFIG["units"]
        },
        "difficulty_config": {
            "description": get_difficulty_description(difficulty) if 'get_difficulty_description' in globals() else f"Difficulty level {difficulty}"
        },
        "generation_config": {
            "multi_container_config": SCENE_SPACE_CONFIG["multi_container"]
        }
    }

def get_difficulty_description(difficulty: int) -> str:
    """获取难度级别的描述"""
    descriptions = {
        0: "Basic: 大空间，简单布局，标准密度",
        1: "Easy: 稍大空间，稍复杂布局，稍高密度", 
        2: "Medium: 标准空间，复杂布局，高密度",
        3: "Hard: 小空间，复杂布局，高密度，频繁碰撞",
        4: "Extreme: 极小空间，极复杂布局，极高密度，极高碰撞频率"
    }
    return descriptions.get(difficulty, "Unknown difficulty")

def generate(types: str, difficulty: int, num_samples: int, out_root: str, skip_existing: bool, max_retries: int):
    types_sorted = "".join(sorted(types))
    diff_name = DIFF_NAMES[difficulty]  # 新增: 获取难度名称
    # 新路径: scenes_<types>/<difficulty_name>/
    folder = os.path.join(out_root, f"scenes_{types_sorted}", diff_name)
    os.makedirs(folder, exist_ok=True)
    feature_list = list(types_sorted)

    generated_count = 0
    skipped_count = 0

    # 校验参数（默认启用）：
    VALIDATE = GLOBAL_SCENE_CONSTANTS["validation"]["enable"]
    TOLERANCE_PX = GLOBAL_SCENE_CONSTANTS["validation"]["tolerance_px"]
    TIMESTAMPS = GLOBAL_SCENE_CONSTANTS["validation"]["timestamps"]
    try:
        from src.utils.config import VALIDATION_BASELINE_TIME_STEP as DT_DEFAULT
    except Exception:
        from utils.config import VALIDATION_BASELINE_TIME_STEP as DT_DEFAULT
    # 目标步长使用配置中的 GROUND_TRUTH_TIME_STEP
    try:
        from src.utils.config import GROUND_TRUTH_TIME_STEP as DT_TARGET
    except Exception:
        from utils.config import GROUND_TRUTH_TIME_STEP as DT_TARGET

    # 解析数据集名（用于 predict 读取刚写入的场景）
    dataset_name = _resolve_dataset_name(out_root)

    for i in range(1, num_samples + 1):
        # 新文件名
        scene_name = f"scene_{types_sorted}_{diff_name}_{i}.json"
        path = os.path.join(folder, scene_name)
        
        # 检查文件是否已存在
        if skip_existing and os.path.exists(path):
            print(f"[SKIP] {path} already exists")
            skipped_count += 1
            continue

        attempt = 0
        while True:
            attempt += 1
            seed = random.randint(0, 10_000_000)
            try:
                if len(feature_list) == 1:
                    scene = gen_single_axis_scene(feature_list[0], difficulty, seed)
                else:
                    scene = generate_from_envelope(feature_list, difficulty, seed)
                scene["meta"]["index"] = i
                scene["meta"]["difficulty_level"] = difficulty

                # 直接写入最终路径用于校验
                with open(path, "w", encoding="utf-8") as f:
                    tmp_scene = normalize_scene_integers(scene)
                    json.dump(tmp_scene, f, indent=GLOBAL_SCENE_CONSTANTS["format"]["json_indent"])

                # 计算用于 predict 的相对场景名（不带 .json）
                scene_rel_wo_ext = os.path.join(f"scenes_{types_sorted}", diff_name, os.path.splitext(scene_name)[0]).replace("\\", "/")

                passed = True
                max_err = 0.0
                if VALIDATE:
                    try:
                        passed, max_err = _validate_scene_difference(scene_rel_wo_ext, dataset_name, TIMESTAMPS, DT_DEFAULT, DT_TARGET, TOLERANCE_PX)
                    except Exception as e:
                        # 读取失败或仿真失败也视为不通过
                        passed = False
                        max_err = float("inf")
                        print(f"[WARN] Validation failed due to error: {e}")

                if passed:
                    print(f"[OK] Generated {path} (max_err={max_err:.3f}px)")
                    generated_count += 1
                    break
                else:
                    # 删除写入的文件并重试
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
                    if attempt >= max_retries:
                        raise RuntimeError(f"生成场景多次校验失败（>{TOLERANCE_PX}px），类型={types_sorted} 难度={diff_name} 序号={i}，最后一次误差={max_err:.3f}px")
                    print(f"[RETRY] Validation failed (max_err={max_err:.3f}px) -> regenerate (attempt {attempt}/{max_retries})")
            except Exception as e:
                # 生成阶段失败（如重叠放置失败），也进行重试
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                if attempt >= max_retries:
                    raise RuntimeError(f"生成场景多次失败（生成阶段异常）：类型={types_sorted} 难度={diff_name} 序号={i}，错误={e}")
                print(f"[RETRY] Generation error: {e} -> regenerate (attempt {attempt}/{max_retries})")

    if skip_existing and skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} existing scenes, generated {generated_count} new scenes")

def parse_args():
    ap = argparse.ArgumentParser(description="Scene Generator (A-G axes + combinations)")
    # 修改: 支持列表输入和"all"
    cli = GLOBAL_SCENE_CONSTANTS["cli_defaults"]
    ap.add_argument("--types", nargs="*", default=cli["types"], 
                    help="Scene types to generate: single letters (A B), combinations (ABC DEF), or 'all' for all combinations")
    ap.add_argument("--difficulty", type=int, default=cli["difficulty"], choices=[0, 1, 2, 3, 4], help="Difficulty level: 0=Basic, 1=Easy, 2=Medium, 3=Hard, 4=Extreme")
    ap.add_argument("--scenes-per-config", type=int, default=cli["scenes_per_config"], help="Number of scenes to generate")
    ap.add_argument("--seed", type=int, default=cli["seed"], help="Random seed (optional)")
    ap.add_argument("--output-dir", type=str, default=cli["output_dir"], help="Output root directory")
    return ap.parse_args()

def parse_types_parameter(types_input):
    """
    解析types参数，支持多种格式：
    - ["all"] -> 生成所有组合
    - ["A", "B", "ABC"] -> 生成指定的单轴和组合
    - ["A"] -> 仅生成A轴场景
    
    注意：H场景只支持有限的组合 (H, HE)
    """
    if not types_input:
        return ["A"]  # 默认
    
    # 检查是否包含"all"
    if "all" in [t.lower() for t in types_input]:
        print("[INFO] Found 'all' in types - generating ALL scene type combinations...")
        all_types = []
        # 单轴场景
        for letter in "ABCDEFG":  # 移除H，因为H需要特殊处理
            all_types.append(letter)
        # H场景的支持组合
        all_types.extend(["H", "HE"])  # 只添加支持的H组合
        # 其他组合场景 (不包含H的组合)
        for combo_size in range(2, 8):
            for combo in combinations("ABCDEFG", combo_size):
                all_types.append("".join(combo))
        return all_types
    
    # 处理显式指定的类型
    valid_letters = set("ABCDEFGHI")
    valid_h_combinations = {"H", "EH", "AH", "BH", "ABH", "ABEH"}  # 支持同步旋转的H组合（注意字母顺序）
    type_combinations = []
    
    for type_str in types_input:
        # 清理和验证输入
        clean_type = "".join(sorted(set(type_str.upper())))
        
        # 验证所有字符都是有效字母
        if not set(clean_type).issubset(valid_letters):
            invalid = set(clean_type) - valid_letters
            raise ValueError(f"Invalid letters in type '{type_str}': {invalid}. Use A-H letters.")
        
        # 特殊验证H组合
        if "H" in clean_type and clean_type not in valid_h_combinations:
            raise ValueError(f"不支持的H组合: {clean_type}. H只支持: {valid_h_combinations}")
        
        if clean_type and clean_type not in type_combinations:
            type_combinations.append(clean_type)
    
    if not type_combinations:
        return ["A"]  # 默认
    
    return type_combinations

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    
    try:
        type_combinations = parse_types_parameter(args.types)
        
        print(f"[INFO] Generating scenes for {len(type_combinations)} type(s): {type_combinations}")
        print(f"[INFO] Scene space configuration: {SCENE_SPACE_CONFIG['base_width']}x{SCENE_SPACE_CONFIG['base_height']} meters base")
        
        for types_str in type_combinations:
            print(f"[INFO] Generating scenes for type: {types_str}")
            cli = GLOBAL_SCENE_CONSTANTS["cli_defaults"]
            generate(types_str, args.difficulty, args.scenes_per_config, args.output_dir, cli["skip_existing"], cli["max_retries"])
        
        total_scenes = len(type_combinations) * args.scenes_per_config
        print(f"[INFO] All scenes generated successfully! Total: {total_scenes} scenes")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

def normalize_scene_integers(scene: Dict[str, Any]) -> Dict[str, Any]:


    out = dict(scene)

    # physics
    if "physics" in out and isinstance(out["physics"], dict):
        phys = dict(out["physics"])
        for k in ("gravity_x", "gravity_y"):
            if k in phys:
                phys[k] = to_decimal(phys[k])
        out["physics"] = phys

    # boxes
    if "boxes" in out and isinstance(out["boxes"], list):
        new_boxes = []
        for b in out["boxes"]:
            if not isinstance(b, dict):
                new_boxes.append(b)
                continue
            nb = dict(b)
            if "center" in nb and isinstance(nb["center"], (list, tuple)):
                nb["center"] = [to_int(nb["center"][0]), to_int(nb["center"][1])]
            for k in ("diameter", "sides"):
                if k in nb:
                    nb[k] = to_int(nb[k])
            if "rotation_speed" in nb:
                nb["rotation_speed"] = to_decimal(nb["rotation_speed"])
            # translation_path: list of [x,y]
            if "translation_path" in nb and isinstance(nb["translation_path"], list):
                nb["translation_path"] = [
                    [to_int(p[0]), to_int(p[1])] if isinstance(p, (list, tuple)) and len(p) >= 2 else p
                    for p in nb["translation_path"]
                ]
            new_boxes.append(nb)
        out["boxes"] = new_boxes

    # balls
    if "balls" in out and isinstance(out["balls"], list):
        new_balls = []
        for ball in out["balls"]:
            if not isinstance(ball, dict):
                new_balls.append(ball)
                continue
            nb = dict(ball)
            if "position" in nb and isinstance(nb["position"], (list, tuple)):
                nb["position"] = [to_int(nb["position"][0]), to_int(nb["position"][1])]
            if "velocity" in nb and isinstance(nb["velocity"], (list, tuple)):
                nb["velocity"] = [to_decimal(nb["velocity"][0]), to_decimal(nb["velocity"][1])]
            # Use decimal precision for floating point properties
            for k in ("radius", "density", "restitution", "angular_velocity"):
                if k in nb:
                    nb[k] = to_decimal(nb[k])
            # Keep sides as integer
            for k in ("sides",):
                if k in nb:
                    nb[k] = to_int(nb[k])
            if "color" in nb and isinstance(nb["color"], (list, tuple)) and len(nb["color"]) >= 3:
                nb["color"] = [to_int(nb["color"][0]), to_int(nb["color"][1]), to_int(nb["color"][2])]
            new_balls.append(nb)
        out["balls"] = new_balls

    return out

def random_spawn_offset(radius: float) -> Tuple[float, float]:
    """按统一上界生成基于半径的随机偏移 (dx, dy)。"""
    m = GLOBAL_SCENE_CONSTANTS["spawn_random"]["radius_multiplier_max"]
    return radius * random.uniform(-m, m), radius * random.uniform(-m, m)

# 新增：球重叠近似与断言
def _approx_balls(balls: List[Dict]) -> List[Tuple[float, float, float]]:
    """近似：使用球心与半径"""
    out: List[Tuple[float, float, float]] = []
    for b in balls:
        pos = b.get("position")
        r = b.get("radius", 0)
        if not pos:
            continue
        out.append((pos[0], pos[1], r))
    return out

def _assert_no_ball_overlap(balls: List[Dict], margin: float = 0.0):
    """严格断言：任意两球的近似圆不相交（半径相加 + 可选边距）"""
    approx = _approx_balls(balls)
    n = len(approx)
    for i in range(n):
        x1, y1, r1 = approx[i]
        for j in range(i+1, n):
            x2, y2, r2 = approx[j]
            dist2 = (x1 - x2)**2 + (y1 - y2)**2
            limit = (r1 + r2 + margin)
            if dist2 < limit * limit:
                raise RuntimeError(f"Ball overlap detected between {i} and {j}")

# 新增：统一把标量或单值转为二元区间(tuple)
def _as_pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return (value[0], value[1])
        if len(value) == 1:
            return (value[0], value[0])
        raise ValueError("Empty range is not allowed")
    if isinstance(value, (int, float)):
        return (value, value)
    raise TypeError(f"Unsupported range type: {type(value)}")

# 新增：严格要求区间存在且为二元
def _require_pair(value, name: str) -> Tuple[float, float]:
    if value is None:
        raise ValueError(f"Missing required envelope range: {name}")
    pair = _as_pair(value) if not (isinstance(value, (list, tuple)) and len(value) == 2) else (value[0], value[1])
    return pair

# 基于内切圆的工具与最终断言

def compute_inradius(diameter: float, sides: int) -> float:
    sides = max(3, int(sides))
    return (diameter / 2.0) * math.cos(math.pi / sides)


def _allowed_center_radius(diameter: float, sides: int, ball_radius: float) -> float:
    clearance = GLOBAL_SCENE_CONSTANTS["placement"]["container_clearance"]
    overlap_margin = GLOBAL_SCENE_CONSTANTS["overlap_check_margin"]
    return compute_inradius(diameter, sides) - ball_radius - max(clearance, overlap_margin)


def random_offset_in_incircle(ball_radius: float, diameter: float, sides: int) -> Tuple[float, float]:
    R_allow = _allowed_center_radius(diameter, sides, ball_radius)
    if R_allow <= 0:
        raise RuntimeError("Insufficient incircle radius for ball spawn")
    ang = random.uniform(0, GLOBAL_SCENE_CONSTANTS["math"]["tau"])
    # 均匀采样圆盘：半径需取 sqrt(u)
    r = R_allow * math.sqrt(random.uniform(0.0, 1.0))
    return r * math.cos(ang), r * math.sin(ang)


def _assert_balls_inside_containers(balls: List[Dict], boxes: List[Dict]):
    """严格断言：每个球均位于某个容器的内切圆允许范围内（球心距离≤R_allow）。
    多容器：按最近容器中心匹配。
    """
    if not boxes or not balls:
        return
    for bi, b in enumerate(balls):
        bx, by = b.get("position", [None, None])
        br = float(b.get("radius", 0.0))
        if bx is None or by is None:
            raise RuntimeError(f"Ball {bi} has invalid position")
        # 选择最近容器
        min_d2 = float("inf")
        chosen = None
        for ci, c in enumerate(boxes):
            cx, cy = c.get("center", [0.0, 0.0])
            dx = bx - cx
            dy = by - cy
            d2 = dx*dx + dy*dy
            if d2 < min_d2:
                min_d2 = d2
                chosen = c
        if chosen is None:
            raise RuntimeError("No container available for ball containment check")
        cx, cy = chosen["center"]
        sides = int(chosen["sides"]) if "sides" in chosen else 3
        diameter = float(chosen["diameter"]) if "diameter" in chosen else 0.0
        R_allow = _allowed_center_radius(diameter, sides, br)
        if R_allow <= 0:
            raise RuntimeError(f"Container incircle too small (ball={bi})")
        d = math.hypot(bx - cx, by - cy)
        if d > R_allow:
            raise RuntimeError(f"Ball {bi} outside safe incircle bound of its container (d={d:.2f} > {R_allow:.2f})")


if __name__ == "__main__":
    main()
