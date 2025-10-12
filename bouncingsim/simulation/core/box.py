import Box2D
import math
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.config import *

class Box:
    def __init__(self, world, center_x, center_y, diameter, sides=6, rotation=0.0, friction=0.0, restitution=1.0, rotation_speed=0.0):
        self.world = world
        self.center = (center_x, center_y)
        self.diameter = diameter
        self.sides = sides
        self.rotation = rotation  # 初始旋转角度（弧度）
        self.friction = friction
        self.restitution = restitution
        self.rotation_speed = rotation_speed
        self.radius = diameter / 2
        
        # 直接使用米坐标，不转换
        self.center_x_meters = center_x
        self.center_y_meters = center_y
        self.radius_meters = self.radius
        
        # Create the kinematic body for the box
        self.body = self.create_kinematic_polygon_box()
        self.body.angularVelocity = self.rotation_speed
    
    def create_kinematic_polygon_box(self):
        """Create a kinematic body for the polygon box using a chain shape"""
        body = self.world.CreateKinematicBody(
            position=(self.center_x_meters, self.center_y_meters),
            angle=self.rotation
        )
        
        # Calculate vertices relative to the body's origin
        vertices = []
        angle_step = 2 * math.pi / self.sides
        for i in range(self.sides):
            angle = i * angle_step
            x = self.radius_meters * math.cos(angle)
            y = self.radius_meters * math.sin(angle)
            vertices.append((x, y))
        
        # Create a chain shape for the polygon boundary
        fixture = body.CreateChainFixture(
            vertices_loop=vertices,
            friction=self.friction,
            restitution=self.restitution
        )
        
        return body

    def get_screen_vertices(self):
        """Get vertices for rendering the box outline"""
        vertices = []
        
        # Get current position and rotation from the physics body
        transform = self.body.transform
        
        # The fixture's shape is a b2ChainShape
        chain_shape = self.body.fixtures[0].shape
        
        for vertex in chain_shape.vertices:
            # Transform local vertex to world coordinates
            world_vertex = transform * vertex
            
            # 直接使用米坐标，因为PPM=1，米=像素
            screen_x = world_vertex.x
            screen_y = world_vertex.y
            vertices.append((int(screen_x), int(screen_y)))
        
        return vertices
    
    def get_screen_vertices_corrected(self):
        """Get vertices for rendering the box outline with corrected Y-axis"""
        vertices = []
        
        # Get current position and rotation from the physics body
        transform = self.body.transform
        
        # The fixture's shape is a b2ChainShape
        chain_shape = self.body.fixtures[0].shape
        
        for vertex in chain_shape.vertices:
            # Transform local vertex to world coordinates
            world_vertex = transform * vertex
            
            # 转换坐标：物理世界Y轴向上，屏幕Y轴向下
            # 由于PPM=1，直接转换，但需要翻转Y轴
            screen_x = world_vertex.x
            screen_y = SCREEN_HEIGHT - world_vertex.y  # 翻转Y轴
            vertices.append((int(screen_x), int(screen_y)))
        
        return vertices
