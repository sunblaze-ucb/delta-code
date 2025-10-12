import os
import sys
import json
import math
import argparse
from typing import List, Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from utils.scene_loader import SceneLoader
    from utils.config import GROUND_TRUTH_TIME_STEP
    from utils.math_parser import parse_rotation_speed
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all required files are in the correct directories")
    sys.exit(1)


def get_project_root():
    """Get the absolute path to the project root directory"""
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # If currently in the ballsim folder, return directly
    if os.path.basename(current_dir) == "ballsim":
        return current_dir

    # If currently in a subfolder of ballsim, search upwards
    while current_dir != os.path.dirname(current_dir):  # Avoid infinite loop
        if os.path.basename(current_dir) == "ballsim":
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # If not found, use relative path
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _simulate_and_get_positions(scene_name: str, timestamp: float, dataset_name: str = "default", time_step: float = None):
    """
    Run simulation and return list of (x,y) meter coordinates for balls.
    """
    # Import here to avoid circular import
    try:
        from simulation.core.physics_world import PhysicsWorld
        from simulation.core.ball import Ball
        from simulation.core.box import Box
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all required files are in the correct directories")
        sys.exit(1)
    
    # 1. Load Scene
    # Get project root directory
    project_root = get_project_root()
    
    # Scene path format: dataset_name/scenes_type/difficulty/scene_name
    # Dynamically determine scenes directory based on dataset_name
    scenes_dir = os.path.join(project_root, "data", "scenes", dataset_name)
    
    scene_loader = SceneLoader(scenes_dir)
    try:
        config = scene_loader.load_scene(scene_name)
    except FileNotFoundError:
        print(f"Error: Scene '{scene_name}' not found in '{scenes_dir}'")
        print(f"Expected path: {os.path.join(scenes_dir, scene_name)}.json")
        sys.exit(1)

    # 2. Setup Physics World and Objects
    # Apply physics settings
    physics_config = config.get("physics", {})
    meta = config.get("meta", {})  # Added: read metadata
    gravity_enabled = physics_config.get("gravity_enabled")
    gravity_x = physics_config.get("gravity_x")
    gravity_y = physics_config.get("gravity_y")
    
    # Use passed time_step, otherwise use GROUND_TRUTH_TIME_STEP
    effective_time_step = time_step if time_step is not None else GROUND_TRUTH_TIME_STEP
    physics_world = PhysicsWorld(gravity_x, gravity_y, gravity_enabled, time_step=effective_time_step)
    
    # Added: support gravity time variation
    gp = meta.get("gravity_profile")
    if gp and isinstance(gp, dict):
        tv = gp.get("time_variation")
        if tv:
            amp = tv.get("amp", 0.0)
            freq = tv.get("freq", 1.0)
            physics_world.set_gravity_time_variation(gravity_x, gravity_y, amp, freq)
    
    # Create boxes (convex only)
    boxes_config = config.get("boxes", [])
    rotation_profile = meta.get("outer_rotation_profile")  # Added: unified outer rotation time variation
    for i, box_config in enumerate(boxes_config):
        center = box_config.get("center")
        diameter = box_config.get("diameter")
        sides = box_config.get("sides")
        rotation_degrees = box_config.get("rotation", 0.0)
        rotation_radians = math.radians(rotation_degrees)
        friction = box_config.get("friction", 0.0)
        restitution = box_config.get("restitution", 1.0)
        rotation_speed = parse_rotation_speed(box_config, "rotation_speed", 0.0)
        translation_path = box_config.get("translation_path")
        
        physics_world.create_box(
            center_x=center[0],
            center_y=center[1],
            diameter=diameter,
            sides=sides,
            rotation=rotation_radians,
            friction=friction,
            restitution=restitution,
            rotation_speed=rotation_speed,
            translation_path=translation_path,
            rotation_profile=rotation_profile
        )

    # Create balls (convex only)
    balls = []
    balls_config = config.get("balls", [])
    for i, ball_config in enumerate(balls_config):
        position = ball_config.get("position", [100, 100])
        velocity = ball_config.get("velocity", [0, 0])
        radius = ball_config.get("radius", 20)
        density = ball_config.get("density", 1.0)
        restitution = ball_config.get("restitution", 1.0)
        color = ball_config.get("color", [255, 0, 0])
        sides = ball_config.get("sides", 0)
        rotation_affected = ball_config.get("rotation_affected", True)
        
        ball = physics_world.create_ball(
            x=position[0], 
            y=position[1], 
            radius=radius, 
            density=density, 
            restitution=restitution, 
            initial_velocity=velocity, 
            color=tuple(color), 
            sides=sides, 
            rotation_affected=rotation_affected,
            non_convex=False,
            concavity_count=0,
            concavity_depth_ratio=0.0,
            angular_velocity=ball_config.get("angular_velocity", 0.0)
        )
        balls.append(ball)

    # 3. Run Simulation with detailed logging
    current_time = 0.0
    step_count = 0
    
    # Print detailed information every 10 steps, or every 0.1 seconds
    print_interval = max(1, int(0.1 / effective_time_step))
    
    while current_time < timestamp:
        physics_world.step()
        current_time += effective_time_step
        step_count += 1
        
        if step_count % print_interval == 0 or step_count == 1:
            for i, ball in enumerate(balls):
                _ = ball.get_screen_position()
                _ = ball.body.linearVelocity
                _ = math.degrees(ball.body.angle)
    
    # print(f"\n=== Simulation completed ===")
    # print(f"Final time: {current_time:.3f}s")
    # print(f"Total steps: {step_count}")
    # print("Final state:")
    # for i, ball in enumerate(balls):
    #     pos = ball.get_screen_position()
    #     vel = ball.body.linearVelocity
    #     angle = math.degrees(ball.body.angle)
    #     print(f"  Ball {i+1}: position=({pos[0]:7.2f}, {pos[1]:7.2f}) | velocity=({vel[0]:7.3f}, {vel[1]:7.3f}) | angle={angle:7.3f}°")

    # 4. Collect positions
    return [ball.get_screen_position() for ball in balls]


def predict_ball_positions(scene_name: str, timestamp: float, dataset_name: str = "default", time_step: float = None):
    """
    Simulates a scene for a given duration and returns the final positions of the balls.
    
    Args:
        scene_name: The name of the scene to load (e.g., 'scene_A_basic_1')
        timestamp: The timestamp in seconds to predict positions for
        dataset_name: The dataset name (default: "default")
        time_step: Optional custom physics time step. If None, uses GROUND_TRUTH_TIME_STEP.
    """
    positions = _simulate_and_get_positions(scene_name, timestamp, dataset_name, time_step=time_step)
    return positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ball positions at a specific timestamp for a given scene.")
    parser.add_argument("scene_name", type=str, help="The name of the scene to load (e.g., 'scene_1').")
    parser.add_argument("timestamp", type=float, help="The timestamp in seconds to predict positions for.")
    
    args = parser.parse_args()
    
    positions = predict_ball_positions(args.scene_name, args.timestamp)
    print(f"Ball positions at timestamp {args.timestamp:.2f}s for scene '{args.scene_name}':")
    for i, pos in enumerate(positions):
        print(f"  Ball {i + 1}: Position=({pos[0]}, {pos[1]})")
