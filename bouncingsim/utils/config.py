# Screen settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 1080
FPS = 1000

# Physics settings
TIME_STEP = 1.0 / FPS
# High precision TIME_STEP for ground truth generation (predict.py)
GROUND_TRUTH_TIME_STEP = 1.0 / 4000
# Baseline TIME_STEP for validation/comparison during scene generation
VALIDATION_BASELINE_TIME_STEP = 1.0 / 3000
PPM = 1.0  # 1 meter = 1 pixel, values in scene files are treated directly as meters
VELOCITY_ITERATIONS = 100
POSITION_ITERATIONS = 100
# Remove ground constants as ground is deleted
# GROUND_HEIGHT = 50
# GROUND_FRICTION = 0.5

# Colors
COLORS = {
    'background': (135, 206, 235),  # Sky blue
    # 'ground': (139, 69, 19),      # Removed
}

# Note: Scene generator constants are now defined per-feature in src.scene_generation.scene_config
