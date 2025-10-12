import Box2D
import sys
import os
import random
import math
from typing import List, Tuple

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.config import *
from simulation.core.ball import Ball
from simulation.core.box import Box

class PhysicsWorld:
    def __init__(self, gravity_x=0.0, gravity_y=0.0, gravity_enabled=True, time_step=None):
        # 完全使用米单位，场景文件中的数值直接当作米
        if gravity_enabled:
            # 重力直接使用米单位，不转换
            self.gravity = (gravity_x, gravity_y)
        else:
            self.gravity = (0.0, 0.0)
        
        # 设置时间步长，如果没有指定则使用默认的TIME_STEP
        self.time_step = time_step if time_step is not None else TIME_STEP
        
        # Create Box2D world with gravity (米单位)
        self.world = Box2D.b2World(gravity=self.gravity, doSleep=True)
        
        # Initialize boxes list (only keep boxes, remove balls)
        self.boxes = []
        self.sim_time = 0.0  # 新增: 累计时间
        self.gravity_time_var = None  # {base:(gx,gy), amp, freq} or None
        self.dynamic_boxes = []  # 记录平移/时间变旋转
    
    def set_gravity(self, gravity_x, gravity_y, enabled=True):
        """Update gravity settings"""
        if enabled:
            self.world.gravity = (gravity_x, gravity_y)
        else:
            self.world.gravity = (0.0, 0.0)
    
    def set_time_step(self, time_step):
        """动态设置时间步长"""
        self.time_step = time_step
    
    def set_gravity_time_variation(self, base_gx, base_gy, amp, freq):
        """启用重力时间变化 (仅对 y 分量施加振荡或通用幅度)"""
        self.gravity_time_var = {
            "base": (base_gx, base_gy),
            "amp": amp,
            "freq": freq
        }
    
    def create_box(self, center_x, center_y, diameter, sides=6, rotation=0.0,
                   friction=0.0, restitution=1.0, rotation_speed=0.0,
                   translation_path=None, rotation_profile=None):
        """
        translation_path: dict or None, e.g.
            {type:'sin1d', axis:'x', amplitude:A, freq:f}
            {type:'lissajous', ax:A, ay:B, fx:fx, fy:fy}
            {type:'piecewise', segments:n, amplitude:A}
        rotation_profile: dict or None, e.g. {omega_base, omega_amp}
        """
        box = Box(self.world, center_x, center_y, diameter, sides, rotation, friction, restitution, rotation_speed)
        self.boxes.append(box)
        if translation_path or rotation_profile:
            self.dynamic_boxes.append({
                "box": box,
                "origin": (center_x, center_y),
                "translation_path": translation_path,
                "rotation_profile": rotation_profile,
                "base_angle": rotation
            })
        return box
    
    def create_ball(self, x, y, radius, density=1.0, restitution=0.9,
                    initial_velocity=(0, 0), color=(255, 0, 0), sides=0, 
                    rotation_affected=True, non_convex=False, concavity_count=0, 
                    concavity_depth_ratio=0.0, angular_velocity=0.0):
        """Create a ball (所有参数都是米单位)"""
        ball = Ball(self.world, x, y, radius, density, restitution,
                   initial_velocity, color, sides, rotation_affected,
                   non_convex, concavity_count, concavity_depth_ratio,
                   angular_velocity=angular_velocity)
        return ball
    
    def create_non_convex_box(self, center_x, center_y, diameter, sides, 
                             concavity_count=1, concavity_depth_ratio=0.1, friction=0.0):
        """Create a non-convex box (所有参数都是米单位)"""
        # 这里可以添加非凸盒子的创建逻辑
        # 暂时使用普通的create_box方法

        raise NotImplementedError("Non-convex boxes are not supported yet")
        return self.create_box(
            center_x=center_x,
            center_y=center_y,
            diameter=diameter,
            sides=sides,
            rotation=0.0,
            friction=friction,
            restitution=1.0,
            rotation_speed=0.0
        )
    
    def apply_explosion(self, x, y, force, balls):
        """Apply explosion force at position (x, y) to all balls"""
        for ball in balls:
            dx = ball.body.position.x - x
            dy = ball.body.position.y - y
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 0:
                # 标准化方向向量
                dx /= distance
                dy /= distance
                # 应用力（米单位）
                ball.body.ApplyForceToCenter((dx * force, dy * force), True)
    
    def _eval_translation(self, spec, t, origin):
        """Evaluate translation at time t"""
        if not spec:
            return origin
        
        ox, oy = origin
        t_local = t
        tp = spec.get("type")
        if tp == "sin1d":
            amp = spec.get("amplitude", 0)
            axis = spec.get("axis", "x")
            freq = spec.get("freq", 1.0)
            disp = amp * math.sin(2 * math.pi * freq * t_local)
            if axis == "x":
                return ox + disp, oy
            else:
                return ox, oy + disp
        if tp == "lissajous":
            ax = spec.get("ax", 0)
            ay = spec.get("ay", 0)
            fx = spec.get("fx", 1.0)
            fy = spec.get("fy", 1.0)
            x = ox + ax * math.sin(2 * math.pi * fx * t_local)
            y = oy + ay * math.sin(2 * math.pi * fy * t_local + math.pi / 2)
            return x, y
        # if tp == "linear":
        #     vx = spec.get("vx", 0.0)
        #     vy = spec.get("vy", 0.0)
        #     return ox + vx * t_local, oy + vy * t_local
        if tp == "piecewise":
            segs = max(1, spec.get("segments", 4))
            amp = spec.get("amplitude", 0)
            seg_len = 1.0 / segs
            phase = (t_local % 1.0) / seg_len
            k = int(phase)
            # 简易折线: 在每段内线性往返 (三角波)
            tri = 2 * abs((phase - k) - 0.5)
            dx = amp * (1 - tri)
            return ox + dx, oy
        
        return origin
    
    def _update_dynamics(self, dt):
        # 重力时间变
        if self.gravity_time_var:
            base_gx, base_gy = self.gravity_time_var["base"]
            amp = self.gravity_time_var["amp"]
            freq = self.gravity_time_var["freq"]
            # 对 y 分量加正弦扰动 (可扩展)
            gy = base_gy + amp * math.sin(2 * math.pi * freq * self.sim_time)
            self.world.gravity = (base_gx, gy)
        
        # 外框动态
        for entry in self.dynamic_boxes:
            box = entry["box"]
            body = box.body
            
            # 目标平移位置（根据当前时间）
            cx, cy = self._eval_translation(entry["translation_path"], self.sim_time, entry["origin"])
            # 用差分速度驱动而不是瞬移，避免穿透
            if dt and dt > 0.0:
                vx = (cx - body.position.x) / dt
                vy = (cy - body.position.y) / dt
            else:
                vx, vy = 0.0, 0.0
            body.linearVelocity = (vx, vy)
            
            # 目标角度：正确积分时间变化的角速度
            omega = box.rotation_speed  # 原有匀速
            prof = entry["rotation_profile"]
            if prof:
                # 对于 omega(t) = omega_base + omega_amp * sin(t)
                # 积分得到: angle(t) = omega_base * t - omega_amp * cos(t) + omega_amp
                omega_base = prof.get("omega_base", omega)
                omega_amp = prof.get("omega_amp", 0.0)
                target_angle = entry["base_angle"] + omega_base * self.sim_time - omega_amp * math.cos(self.sim_time) + omega_amp
            else:
                # 匀速旋转的情况
                target_angle = entry["base_angle"] + omega * self.sim_time
            if dt and dt > 0.0:
                ang_vel = (target_angle - body.angle) / dt
            else:
                ang_vel = 0.0
            body.angularVelocity = ang_vel
    
    def step(self):
        """Advance physics simulation by one step"""
        self.sim_time += self.time_step
        self._update_dynamics(self.time_step)
        self.world.Step(self.time_step, VELOCITY_ITERATIONS, POSITION_ITERATIONS)
        self.world.ClearForces()
    
    def get_nearest_ball(self, x, y, balls):
        """Get the ball closest to position (x, y)"""
        if not balls:
            return None
        
        nearest_ball = None
        min_distance = float('inf')
        
        for ball in balls:
            ball_x, ball_y = ball.body.position.x, ball.body.position.y
            distance = math.sqrt((x - ball_x)**2 + (y - ball_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_ball = ball
        
        return nearest_ball
    
    def clear_boxes(self):
        """Clear all boxes from the physics world"""
        # 销毁所有盒子的物理体
        for box in self.boxes:
            if hasattr(box, 'body') and box.body:
                self.world.DestroyBody(box.body)
        
        # 清空列表
        self.boxes.clear()
        self.dynamic_boxes.clear()
