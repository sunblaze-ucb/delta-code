"""
场景空间配置文件
定义场景生成器的空间参数，完全独立于显示系统
"""

# 基础场景空间配置（单一配置）
SCENE_SPACE_CONFIG = {
    # 场景基准尺寸（米）
    "base_width": 1500.0,
    "base_height": 1500.0,
    "box_base_diameter": 300.0,
    # 多容器布局参数
    "multi_container": {
        "min_gap": 50.0
    },

    # 场景元数据
    "units": "meters",
}

# 统一全局常量（原先分散在生成逻辑中的硬编码）
GLOBAL_SCENE_CONSTANTS = {
    # 验证与采样基准
    "validation": {
        "enable": True,
        "tolerance_px": 15.0,
        "timestamps": [0.5, 1, 1.5]
    },
    # 外平移频率（按难度的分段示意）
    "outer_translation": {
        "freq_low_diff": 0.3,
        "freq_high_diff": 0.6,
        "low_diff_max": 1  # diff<=1 使用低频
    },
    # 外旋时变幅度占比
    "outer_rotation_profile": {
        "omega_amp_ratio": 0.4
    },
    # 重力模式优先级（Envelope 合并时使用）
    "gravity_modes_priority": ["extreme_chaotic", "chaotic", "time_var", "tilted", "vertical", "tiny"],
    # 统一恢复系数（碰撞弹性）
    "restitution": 1.0,
    # 场景中心位置比例（相对宽高）
    "center_fraction": [0.5, 0.5],
    # 默认值集合
    "defaults": {
        "zero": 0.0,
        "box": {"rotation": 0.0, "friction": 0.0, "rotation_speed": 0.0},
        "ball": {"density": 1.0}
    },
    # 数学常量
    "math": {
        "tau": 6.283185307179586  # 2pi
    },
    # 重力参数
    "gravity": {
        "g": 1000.0,
        "tiny": 100.0,
        "small": 1000.0,
        "large": 10000.0,
        "tilt_max_deg": 120.0,
        "chaotic_mag": [5.0, 15.0],
        "extreme_chaotic_mag": [15.0, 25.0],
        "time_variation": {
            "chaotic_amp_ratio": 0.4,
            "chaotic_freq_range": [0.5, 2.0],
            "extreme_amp_ratio": 0.6,
            "extreme_freq_range": [1.0, 3.0]
        },
        "time_var_default": {"amp_ratio": 0.3, "freq": 1.0}
    },
    # 路径/平移配置
    "translation_profiles": {
        "C": {
            "basic_freq": 0.1,
            "easy_freq": 0.5,
            "medium_freq": 1.0,
            "lissajous_fx": 0.6,
            "lissajous_fy": 0.9,
            "piecewise_segments_basic": 4,
            "piecewise_segments_envelope": 6
        }
    },
    # 几何参数
    "geometry": {
        "min_polygon_sides": 3,
        "irregularity_min_scale": 0.3
    },
    # 生成器 CLI 默认
    "cli_defaults": {
        "types": ["A"],
        "difficulty": 0,
        "scenes_per_config": 10,
        "seed": None,
        "output_dir": "scenes",
        "skip_existing": False,
        "max_retries": 60
    },
    # 合法字母与 H 允许组合
    "valid_letters": list("ABCDEFGHI"),
    "valid_h_combinations": ["H", "EH", "AH", "BH", "ABH", "ABEH"],
    # 周期估计参数
    "period": {
        "speed_divisor": 2.0,
        "min_period": 0.5,
        "max_period": 5.0,
        "key_divisions": 4
    },
    # 周期性检查参数
    "period_check": {
        "min_speed": 0.1,
        "min_period": 0.1,
        "max_period": 20.0,
        "displacement_ratio": 0.01
    },
    # H/I 预设参数
    "H": {
        0: {"N": 6, "speed": 200.0, "ball_radius": 25.0},
        1: {"N": 8, "speed": 300.0, "ball_radius": 25.0},
        2: {"N": 10, "speed": 400.0, "ball_radius": 25.0}
    },
    "I": {
        0: {"N": 6,  "speed": 20.0, "target_period": 9.60, "ball_radius": 25.0, "upward": False},
        1: {"N": 8,  "speed": 30.0, "target_period": 6.40, "ball_radius": 25.0, "upward": False},
        2: {"N": 10, "speed": 40.0, "target_period": 3.20, "ball_radius": 25.0, "upward": False},
        3: {"N": 12, "speed": 50.0, "target_period": 2.40, "ball_radius": 25.0, "upward": False},
        4: {"N": 14, "speed": 60.0, "target_period": 1.60, "ball_radius": 25.0, "upward": False}
    },
    # 统一的初始随机偏移（A–F）按半径倍数（完全随机，统一上界）
    "spawn_random": {
        "radius_multiplier_max": 2.0,
        "multi_object_radial_fraction_range": (0.15, 0.85)
    },
    # 近似重叠检测边距（用于 boxes 与 balls 的最终断言）
    "overlap_check_margin": 5.0,
    # 放置与采样参数
    "placement": {
        # 放置期的球-球最小间距阈值（比最终断言更宽松，用于快速采样去重）
        "overlap_distance_threshold": 20.0,
        "tries_per_point": 200,
        "extra_radius_margin": 4.0,
        "container_clearance": 2.0,
        "container_max_attempts": 250,
        "ball_max_attempts": 100
    },
    # 组合生成参数
    "combinations": {
        "min_size": 2
    },
    # 数字格式化
    "format": {
        "round_decimals": 2,
        "timestamp_decimals": 3,
        "json_indent": 2
    },
    # 路径设置
    "paths": {
        "data_dir": "data",
        "scenes_dir": "scenes"
    }
}

# 新增：难度名称与全局上限/弹性系数（由生成器迁移）
DIFF_NAMES = {0: "basic", 1: "easy", 2: "medium", 3: "hard", 4: "extreme"}
INTERNAL_SIDES_MAX = 16
OUTER_SIDES_RECOMMENDED_MAX = 32
RESTITUTION = 1.0

# 新增：统一颜色配置（按轴默认颜色与随机颜色范围）
AXIS_COLORS = {
    "A": [255.0, 100.0, 50.0],
    "B": [60.0, 160.0, 255.0],
    "C": [120.0, 220.0, 80.0],
    "D": [240.0, 180.0, 60.0],
    "E": [200.0, 90.0, 200.0],
    "F": [100.0, 255.0, 200.0],
    # G 轴为多球场景，默认使用随机颜色范围
    "G": None,
    # 周期类/可预测类使用中性灰
    "H": [128.0, 128.0, 128.0],
    "I": [128.0, 128.0, 128.0],
}

# 随机颜色范围（RGB 每通道最小/最大值，含边界）
COLOR_RANDOM_RANGE = (50, 255)

AXIS_RANGES = {
    # A轴：内旋转（Inner Rotation）——球/内部多边形自旋与角速度设置
    "A": {
        0: dict(inner_sides=(3, 4),  ang=(0.1, 0.2),  lin_speed=(200, 400), box_diameter_factor=(1.5, 1.5), box_sides=(3, 4), ball_radius=40.0),
        1: dict(inner_sides=(5, 6),  ang=(0.2, 0.5),  lin_speed=(400, 600), box_diameter_factor=(1.4, 1.4), box_sides=(3, 5), ball_radius=35.0),
        2: dict(inner_sides=(6, 7), ang=(0.5, 1.0), lin_speed=(600, 800), box_diameter_factor=(1.3, 1.3), box_sides=(3, 6), ball_radius=30.0),
        3: dict(inner_sides=(7, 8), ang=(1.0, 2.0), lin_speed=(600, 800), time_var=True, box_diameter_factor=(1.2, 1.2), box_sides=(3, 7), ball_radius=30.0),
        4: dict(inner_sides=(8, 8), ang=(2.0, 2.5), lin_speed=(600, 800), time_var=True, box_diameter_factor=(1.0, 1.0), box_sides=(3, 7), ball_radius=30.0),
    },
    # B轴：外旋转（Outer Rotation）——外框旋转与可能的时间变角速度
    "B": {
        0: dict(inner_sides=(3, 4),  box_sides=(3, 4),  ang=(0.1, 0.2),  lin_speed=(200, 400),  box_diameter_factor=(1.5, 1.5),  ball_radius=40.0),
        1: dict(inner_sides=(5, 6),  box_sides=(5, 6), ang=(0.2, 0.5),  lin_speed=(400, 600),  box_diameter_factor=(1.4, 1.4),  ball_radius=35.0),
        2: dict(inner_sides=(6, 7), box_sides=(6, 7), ang=(0.5, 1.0),  lin_speed=(600, 800), box_diameter_factor=(1.3, 1.3),  ball_radius=30.0),
        3: dict(inner_sides=(7, 8), box_sides=(7, 8), ang=(1.0, 1.5), lin_speed=(800, 1000), time_var=True,  box_diameter_factor=(1.2, 1.2),  ball_radius=25.0),
        4: dict(inner_sides=(8, 10), box_sides=(8, 10), ang=(2.0, 3.0),  lin_speed=(1000, 1200), time_var=True, box_diameter_factor=(0.8, 0.8),  ball_radius=25.0),
    }, 
    # C轴：外平移（Outer Translation/Path）——外框沿轨迹平移（正弦/Lissajous/分段）
    "C": {
        0: dict(inner_sides=(3, 4),  amp=(0, 10),   lin_speed=(200, 400),  box_diameter_factor=(1.5, 1.5), box_sides=(3, 4),  ball_radius=40.0),
        1: dict(inner_sides=(5, 6),  amp=(20, 40),  lin_speed=(400, 600),  box_diameter_factor=(1.4, 1.4), box_sides=(5, 6),  ball_radius=35.0),
        2: dict(inner_sides=(6, 7), amp=(40, 60),  lin_speed=(600, 800), box_diameter_factor=(1.3, 1.3), box_sides=(6, 7), ball_radius=30.0),
        3: dict(inner_sides=(7, 8), amp=(60, 90), lin_speed=(800, 1000), complex_path=True, box_diameter_factor=(1.2, 1.2), box_sides=(7, 8), ball_radius=25.0),
        4: dict(inner_sides=(8, 10), amp=(90, 120), lin_speed=(1000, 1200), complex_path=True, chaotic_path=True, box_diameter_factor=(1.0, 1.0), box_sides=(8, 10), ball_radius=20.0),
    },
    # D轴：重力（Gravity）——不同重力模式（微重力/竖直/倾斜/混沌/极端混沌）
    "D": {
        0: dict(inner_sides=(3, 4),  g_mode="tiny",  lin_speed=(200, 400),  box_diameter_factor=(1.5, 1.5), box_sides=(3, 4),  ball_radius=40.0),
        1: dict(inner_sides=(5, 6),  g_mode="small", lin_speed=(400, 600),  box_diameter_factor=(1.4, 1.4), box_sides=(5, 6),  ball_radius=35.0),
        2: dict(inner_sides=(6, 7), g_mode="large",         lin_speed=(600, 800), box_diameter_factor=(1.3, 1.3), box_sides=(6, 7), ball_radius=30.0),
        3: dict(inner_sides=(7, 8), g_mode="tilted",       lin_speed=(800, 1000), box_diameter_factor=(1.2, 1.2), box_sides=(7, 8), ball_radius=25.0),
        4: dict(inner_sides=(8, 10), g_mode="tilted", lin_speed=(1000, 1200), box_diameter_factor=(1.0, 1.0), box_sides=(8, 10), ball_radius=20.0),
    },
    # FIXME: simulation
    # E轴：不规则内部形状（Irregular Inner Shape）——顶点扰动的不规则凸多边形
    "E": {
        0: dict(inner_sides=(3, 4),  irregularity=1.05, lin_speed=(200, 400),  box_diameter_factor=(1.5, 1.5), box_sides=(3, 4),  ball_radius=40.0),
        1: dict(inner_sides=(5, 6),  irregularity=1.10, lin_speed=(400, 600),  box_diameter_factor=(1.4, 1.4), box_sides=(5, 6),  ball_radius=35.0),
        2: dict(inner_sides=(6, 7), irregularity=1.20, lin_speed=(600, 800), box_diameter_factor=(1.3, 1.3), box_sides=(6, 7), ball_radius=30.0),
        3: dict(inner_sides=(7, 8), irregularity=1.50, lin_speed=(800, 1000), box_diameter_factor=(1.2, 1.2), box_sides=(7, 8), ball_radius=25.0),
        4: dict(inner_sides=(8, 10), irregularity=1.70, lin_speed=(1000, 1200), box_diameter_factor=(1.0, 1.0), box_sides=(8, 10), ball_radius=20.0),
    },
    # F轴：多容器（Multiple Containers/Layout）——多个外框容器布局与群体运动
    "F": {
        0: dict(container_count=2,        inner_sides=(3, 4),  box_sides=(3, 4),   lin_speed=(200, 400),  box_diameter_factor=(1.5, 1.5), ball_radius=40.0),
        1: dict(container_count=2,        inner_sides=(5, 6),  box_sides=(5, 6),  lin_speed=(400, 600),  box_diameter_factor=(1.4, 1.4), ball_radius=35.0),
        2: dict(container_count=3,        inner_sides=(6, 7), box_sides=(6, 7), lin_speed=(600, 800), box_diameter_factor=(1.3, 1.3), ball_radius=30.0),
        3: dict(container_count=4,        inner_sides=(7, 8), box_sides=(7, 8), lin_speed=(800, 1000), box_diameter_factor=(1.2, 1.2), ball_radius=25.0),
        4: dict(container_count=6,        inner_sides=(8, 10), box_sides=(8, 10), lin_speed=(1000, 1200), box_diameter_factor=(1.0, 1.0), ball_radius=20.0),
    },
    # G轴：多球/多内部多边形（Multiple Internal Polygons）——同容器内多个球/多边形
    "G": {
        0: dict(count=(2, 2),  inner_sides=(3, 6),  lin_speed=(200, 400),  box_diameter_factor=(2.5, 2.5), box_sides=(3, 6),  ball_radius=20.0),
        1: dict(count=(3, 3),  inner_sides=(3, 6),  lin_speed=(400, 600),  box_diameter_factor=(2.5, 2.5), box_sides=(3, 6),  ball_radius=20.0),
        2: dict(count=(4, 5),  inner_sides=(3, 6), lin_speed=(600, 800), box_diameter_factor=(2.5, 2.5), box_sides=(3, 6), ball_radius=20.0),
        3: dict(count=(5, 6), inner_sides=(3, 6), lin_speed=(800, 1000), box_diameter_factor=(2.5, 2.5), box_sides=(3, 6), ball_radius=20.0),
        4: dict(count=(7, 9), inner_sides=(3, 6), lin_speed=(1000, 1200), box_diameter_factor=(2.5, 2.5), box_sides=(3, 6), ball_radius=20.0),
    },
    # H轴：可预测/周期性运动预设（Predictable/Periodic）——使用固定的可预测配置
    "H": {
        0: dict(use_predictable_config=True),
        1: dict(use_predictable_config=True),
        2: dict(use_predictable_config=True),
    }
} 