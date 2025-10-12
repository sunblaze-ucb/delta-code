"""
预先验证的可预测H场景配置

这个模块包含经过数学验证的周期性运动配置，确保H场景的真正可预测性。
"""

import math
from typing import Dict, List, Tuple, Any

# 同步旋转配置（用于H+A, H+B, H+AB组合）
SYNCHRONIZED_ROTATION_CONFIGS = {
    0: {
        "angular_velocity": 0.3,  # 较慢的同步旋转
        "initial_angle_alignment": 0.0,  # 初始角度对齐
        "description": "慢速同步旋转",
        "ball_container_sync": True  # 球体和容器同步
    },
    1: {
        "angular_velocity": 0.5,
        "initial_angle_alignment": 0.0,
        "description": "中速同步旋转",
        "ball_container_sync": True
    },
    2: {
        "angular_velocity": 0.8,
        "initial_angle_alignment": 0.0,
        "description": "快速同步旋转",
        "ball_container_sync": True
    }
}

def get_sync_rotation_config(difficulty: int) -> Dict[str, Any]:
    """
    获取同步旋转配置
    
    Args:
        difficulty: 难度级别
        
    Returns:
        同步旋转配置
    """
    return SYNCHRONIZED_ROTATION_CONFIGS.get(difficulty, SYNCHRONIZED_ROTATION_CONFIGS[0]).copy()

# 预先验证的周期性配置库
VERIFIED_PERIODIC_CONFIGS = {
    # Basic难度：简单几何，长周期，低速度
    0: [
        {
            "name": "triangle_45deg",
            "ball_sides": 3,
            "container_sides": 8,
            "container_diameter": 300.0,
            "velocity_angle": math.pi / 4,  # 45度
            "speed": 25.0,
            "initial_offset": (-50.0, 0.0),  # 相对容器中心
            "theoretical_period": 4.24,  # 预计算的理论周期
            "description": "三角形球在八边形容器中45度运动"
        },
        {
            "name": "triangle_30deg", 
            "ball_sides": 3,
            "container_sides": 8,
            "container_diameter": 300.0,
            "velocity_angle": math.pi / 6,  # 30度
            "speed": 20.0,
            "initial_offset": (-60.0, 0.0),
            "theoretical_period": 5.20,
            "description": "三角形球在八边形容器中30度运动"
        }
    ],
    
    # Easy难度：稍微复杂的几何
    1: [
        {
            "name": "square_45deg",
            "ball_sides": 4,
            "container_sides": 8, 
            "container_diameter": 300.0,
            "velocity_angle": math.pi / 4,
            "speed": 35.0,
            "initial_offset": (-50.0, 0.0),
            "theoretical_period": 3.03,
            "description": "正方形球在八边形容器中45度运动"
        },
        {
            "name": "square_60deg",
            "ball_sides": 4,
            "container_sides": 8,
            "container_diameter": 300.0, 
            "velocity_angle": math.pi / 3,  # 60度
            "speed": 30.0,
            "initial_offset": (-45.0, 0.0),
            "theoretical_period": 3.46,
            "description": "正方形球在八边形容器中60度运动"
        }
    ],
    
    # Medium难度：更复杂但仍可预测
    2: [
        {
            "name": "hexagon_30deg",
            "ball_sides": 6,
            "container_sides": 8,
            "container_diameter": 300.0,
            "velocity_angle": math.pi / 6,
            "speed": 45.0,
            "initial_offset": (-40.0, 0.0),
            "theoretical_period": 2.31,
            "description": "六边形球在八边形容器中30度运动"
        }
    ]
}

# 不规则形状的预定义模式（用于H+E组合）
PREDEFINED_IRREGULAR_SHAPES = {
    "notched_triangle": {
        "base_sides": 3,
        "vertices_pattern": "triangle_with_notch",
        "description": "缺角三角形",
        "irregularity_factor": 0.15
    },
    "indented_square": {
        "base_sides": 4, 
        "vertices_pattern": "square_with_indent",
        "description": "凹陷正方形",
        "irregularity_factor": 0.20
    },
    "wavy_hexagon": {
        "base_sides": 6,
        "vertices_pattern": "hexagon_with_waves", 
        "description": "波浪六边形",
        "irregularity_factor": 0.25
    }
}

def calculate_theoretical_period(velocity: List[float], container_diameter: float, 
                               container_sides: int, ball_radius: float = 25.0) -> float:
    """
    基于精确几何计算理论周期。
    对于偶数边正多边形（存在平行对边），在以下约束下可退化为一维直线往返：
      - 球与容器同中心、同初始角度、同为正多边形
      - 初速度沿某一对边的法向（指向边中点）
      - 无重力、无摩擦、完全弹性、无自旋影响
    在该条件下：
      周期 T = 4 * (a_c - a_b) / speed
    其中 a_c = R_c * cos(pi/N), a_b = R_b * cos(pi/N)，R_c=container_diameter/2，R_b=ball_radius。

    若 container_sides 为奇数，则无法保证直线往返，退回到保守估算（圆近似或一般估计）。
    """
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
    if speed <= 0:
        return 0.0

    N = container_sides
    Rc = container_diameter / 2.0
    Rb = ball_radius

    if N % 2 == 0 and N >= 4:
        # 精确一维往返周期（中心到中心），半程距离 = (a_c - a_b)
        a_c = Rc * math.cos(math.pi / N)
        a_b = Rb * math.cos(math.pi / N)
        gap = max(a_c - a_b, 0.0)
        return 2.0 * gap / speed if gap > 0 else 0.0

    # 非偶数边：回退到保守估算
    # 有效运动半径（考虑球的大小）
    effective_radius = (container_diameter / 2.0) - ball_radius

    if N >= 8:
        estimated_period = 4.0 * effective_radius / speed
    elif N == 6:
        estimated_period = 3.46 * effective_radius / speed
    elif N == 4:
        estimated_period = 4.0 * effective_radius / speed
    else:
        estimated_period = 2.0 * math.pi * effective_radius / speed

    return estimated_period

def get_predictable_h_config(difficulty: int, config_index: int = 0) -> Dict[str, Any]:
    """
    获取可预测的H场景配置
    
    Args:
        difficulty: 难度级别 (0-2)
        config_index: 配置索引
        
    Returns:
        验证过的配置字典
    """
    if difficulty not in VERIFIED_PERIODIC_CONFIGS:
        raise ValueError(f"不支持的难度级别: {difficulty}")
    
    configs = VERIFIED_PERIODIC_CONFIGS[difficulty]
    if config_index >= len(configs):
        config_index = config_index % len(configs)  # 循环使用
    
    return configs[config_index].copy()

def get_irregular_shape_config(difficulty: int) -> Dict[str, Any]:
    """
    获取H+E组合的不规则形状配置
    
    Args:
        difficulty: 难度级别
        
    Returns:
        不规则形状配置
    """
    shape_names = list(PREDEFINED_IRREGULAR_SHAPES.keys())
    
    # 根据难度选择不规则形状
    if difficulty == 0:
        shape_name = "notched_triangle"
    elif difficulty == 1:
        shape_name = "indented_square"  
    else:
        shape_name = "wavy_hexagon"
    
    return PREDEFINED_IRREGULAR_SHAPES[shape_name].copy()

def validate_periodic_config(config: Dict[str, Any]) -> bool:
    """
    验证配置是否能产生周期性运动
    
    Args:
        config: 场景配置
        
    Returns:
        是否有效
    """
    try:
        # 计算速度向量
        angle = config["velocity_angle"]
        speed = config["speed"]
        velocity = [speed * math.cos(angle), speed * math.sin(angle)]
        
        # 计算理论周期
        theoretical_period = calculate_theoretical_period(
            velocity, 
            config["container_diameter"],
            config["container_sides"]
        )
        
        # 检查周期是否合理
        if 1.0 <= theoretical_period <= 10.0:
            return True
        else:
            return False
            
    except Exception:
        return False 


def get_predictable_i_config(container_sides: int,
                            diameter: float,
                            speed: float,
                            ball_radius: float = 25.0,
                            upward: bool = True) -> Dict[str, Any]:
    """
    生成 I 场景配置：
    - 偶数边正多边形容器，保证存在一对水平边
    - 设置 rotation 使一对边水平
    - 速度垂直于水平边（竖直向上/向下），实现面-面碰撞的一维往返
    - 返回理论周期（基于面-面法向往返）

    理论周期推导（面-面，速度垂直于边）：
      设外接半径 Rc = diameter/2，偶数边 N，速度大小 s，球半径 r
      水平边的外法向方向的支撑距 h(u) = Rc * cos(pi/N)
      有效半程 L_eff = h(u) - r
      周期 T = 4 * L_eff / s
    """
    import math

    if container_sides % 2 != 0 or container_sides < 4:
        raise ValueError("I 场景要求偶数边且不少于4边的正多边形")

    Rc = diameter / 2.0
    N = container_sides

    # 设置 rotation 使两条边水平：
    # 对于正 N 边形，边法向角度集合为 k*pi/N（k=0..N-1）。
    # 令 rotation = 0 则存在法向为 0（水平外法向）的边，即边本身水平。
    rotation = 0.0

    # 速度垂直于水平边：竖直方向
    vx = 0.0
    vy = -speed if upward else speed

    # 面-面方向的支撑距（外法向为 0 方向）
    h = Rc * math.cos(math.pi / N)
    L_eff = max(h - ball_radius, 0.0)
    theoretical_period = 4.0 * L_eff / speed if speed > 0 and L_eff > 0 else 0.0

    return {
        "container_sides": N,
        "container_diameter": diameter,
        "rotation": rotation,
        "ball_radius": ball_radius,
        "velocity": [vx, vy],
        "speed": speed,
        "period": theoretical_period,
        "description": "I 场景：偶数边，多边形含水平边，速度垂直于水平边，面-面一维往返"
    } 

def get_predictable_i_config_by_period(container_sides: int,
                                      speed: float,
                                      period: float,
                                      ball_radius: float = 25.0,
                                      upward: bool = True) -> Dict[str, Any]:
    """
    I 场景便捷接口：按给定周期 period（两位小数即可）与速度/半径，
    先确定到边距离 a，再反推外接半径 Rc 和直径；速度竖直、边水平。

    公式：
      a = r + s * period / 4
      Rc = a / cos(pi/N)
      diameter = 2 * Rc
    返回与 get_predictable_i_config 一致的结构。
    """
    import math

    if container_sides % 2 != 0 or container_sides < 4:
        raise ValueError("I 场景要求偶数边且不少于4边的正多边形")
    if speed <= 0 or period <= 0:
        raise ValueError("speed 与 period 需为正数")

    N = container_sides
    a = ball_radius + speed * period / 4.0
    Rc = a / math.cos(math.pi / N)
    diameter = 2.0 * Rc

    return get_predictable_i_config(
        container_sides=N,
        diameter=diameter,
        speed=speed,
        ball_radius=ball_radius,
        upward=upward,
    ) 