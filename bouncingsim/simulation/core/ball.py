import Box2D
import math
import random
import sys
import os
import ast
import operator

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.config import *


def evaluate_angular_velocity(angular_velocity):
    """
    Evaluate angular velocity expressions that may be strings containing mathematical expressions.

    Args:
        angular_velocity: Either a numeric value or a string expression like "2*pi/(3*sqrt(3))"

    Returns:
        float: The evaluated angular velocity value

    Raises:
        ValueError: If the expression cannot be evaluated or contains unsafe operations
    """
    if isinstance(angular_velocity, (int, float)):
        return float(angular_velocity)

    if isinstance(angular_velocity, str):
        return evaluate_math_expression(angular_velocity)

    # If it's neither numeric nor string, try to convert to float
    try:
        return float(angular_velocity)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid angular velocity value: {angular_velocity}")


def evaluate_math_expression(expression):
    """
    Safely evaluate a mathematical expression string.

    Args:
        expression: String containing mathematical expression like "2*pi/(3*sqrt(3))"

    Returns:
        float: The evaluated result

    Raises:
        ValueError: If the expression cannot be evaluated or contains unsafe operations
    """
    # Safe evaluation of mathematical expressions
    # Allowed operators and functions
    allowed_names = {
        'pi': math.pi,
        'e': math.e,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'pow': math.pow,
        'abs': abs,
        'round': round,
    }

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_expr(node):
        if isinstance(node, ast.Num):  # Python 3.8+ uses ast.Constant, but Num still works
            return node.n
        elif isinstance(node, ast.Constant):  # For newer Python versions
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            else:
                raise ValueError(f"Unknown name: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = eval_expr(node.left)
            right = eval_expr(node.right)
            op = allowed_operators.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = eval_expr(node.operand)
            op = allowed_operators.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_names:
                func = allowed_names[node.func.id]
                args = [eval_expr(arg) for arg in node.args]
                return func(*args)
            else:
                raise ValueError(f"Unsupported function call: {node.func}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")

    try:
        # Parse and evaluate the expression
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        return float(result)
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"Failed to evaluate mathematical expression '{expression}': {e}")


def evaluate_velocity(velocity):
    """
    Evaluate velocity arrays where each component may be a string expression.

    Args:
        velocity: Either a list/tuple of numeric values or a list/tuple where
                 each element can be either numeric or a string expression

    Returns:
        tuple: (vx, vy) as floats

    Raises:
        ValueError: If velocity format is invalid or expressions cannot be evaluated
    """
    if not isinstance(velocity, (list, tuple)):
        raise ValueError(f"Velocity must be a list or tuple, got {type(velocity)}")

    if len(velocity) != 2:
        raise ValueError(f"Velocity must have exactly 2 components, got {len(velocity)}")

    # Evaluate each component
    vx = evaluate_math_expression(velocity[0]) if isinstance(velocity[0], str) else float(velocity[0])
    vy = evaluate_math_expression(velocity[1]) if isinstance(velocity[1], str) else float(velocity[1])

    return (vx, vy)

class Ball:
    def __init__(self, world, x, y, radius=20, density=1.0, restitution=0.9,
                 initial_velocity=(0, 0), color=(255, 0, 0), sides=0, rotation_affected=True,
                 non_convex=False, concavity_count=0, concavity_depth_ratio=0.0,
                 angular_velocity=0.0):
        # Evaluate velocity components if they contain string expressions
        initial_velocity = evaluate_velocity(initial_velocity)
        # Evaluate angular velocity if it's a string expression
        angular_velocity = evaluate_angular_velocity(angular_velocity)
        """
        Note: Although self.fixture is not used below, it is actually placed in the world when Create is called
        """
        self.radius = radius
        self.color = color
        self.initial_pos = (x, y)
        self.density = density  # Store for reference
        self.restitution = restitution  # Store for reference
        self.sides = sides  # 0 means circle, >2 means polygon
        self.rotation_affected = rotation_affected
        
        # Directly use meter coordinates, no conversion
        x_meters = x
        y_meters = y
        radius_meters = radius
        
        # Create Box2D body
        if non_convex:
            self._create_non_convex_polygon(world, x, y, radius, sides if sides > 2 else 6,
                                            concavity_count, concavity_depth_ratio,
                                            density, restitution)
        else:
            # Directly use m/s velocity, no conversion
            velocity_x_mps = initial_velocity[0]
            velocity_y_mps = initial_velocity[1]
            
            self.body = world.CreateDynamicBody(
                position=(x_meters, y_meters),
                angularVelocity=angular_velocity,
                linearVelocity=(velocity_x_mps, velocity_y_mps)
            )
            self.body.bullet = True
        
            # Set rotation behavior
            if not rotation_affected:
                self.body.fixedRotation = True
            
            # Create fixture based on shape type
            if sides > 2:
                self.create_polygon_fixture(radius_meters, sides, density, restitution)
            else:
                self.create_circle_fixture(radius_meters, density, restitution)
    
    def create_circle_fixture(self, radius_meters, density, restitution):
        """Create circular fixture"""
        # Try different parameter options for fixture creation
        self.fixture = self.body.CreateCircleFixture(
            radius=radius_meters,
            density=density,
            restitution=restitution,
            friction=0.0
        )
    
    def create_polygon_fixture(self, radius_meters, sides, density, restitution):
        """Create polygon fixture"""
        # Generate vertices for regular polygon
        vertices = []
        angle_step = 2 * math.pi / sides
        
        for i in range(sides):
            angle = i * angle_step
            x = radius_meters * math.cos(angle)
            y = radius_meters * math.sin(angle)
            vertices.append((x, y))
        
        try:
            # Create polygon fixture
            self.fixture = self.body.CreatePolygonFixture(
                vertices=vertices,
                density=density,
                restitution=restitution,
                friction=0.0
            )
        except TypeError:
            shape = Box2D.b2PolygonShape(vertices=vertices)
            area = self.calculate_polygon_area(vertices)
            mass = density * area
            self.fixture = self.body.CreateFixture(
                shape=shape,
                density=density
            )
            self.fixture.restitution = restitution
            self.fixture.friction = 0.0
    
    def get_world_position(self):
        """Return the ball's position in world coordinates (meters)"""
        return self.body.position

    def calculate_polygon_area(self, vertices):
        """Calculate area of polygon using shoelace formula"""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0
    
    def get_screen_position(self):
        # NOT SCREEN POSITION NOW! FIXME
        world_pos = self.body.position
        return (round(world_pos.x, 2), round(world_pos.y, 2))
    
    def get_screen_position_corrected(self):
        """Return the ball's position in corrected screen coordinates (pixels)"""
        world_pos = self.body.position
        # Transform coordinates: physics world Y-axis points up, screen Y-axis points down
        # Since PPM=1, direct transformation, but need to flip Y-axis
        screen_x = world_pos.x
        screen_y = SCREEN_HEIGHT - world_pos.y  # Flip Y-axis
        return (int(screen_x), int(screen_y))
    
    def reset_position(self):
        """Reset ball to initial position (meter units)"""
        self.body.position = self.initial_pos
        self.body.linearVelocity = (0, 0)
        self.body.angularVelocity = 0
    
    def apply_force(self, force_x, force_y):
        """Apply force to the ball"""
        self.body.ApplyForceToCenter((force_x, force_y), True)
    
    def get_render_vertices(self):
        """Get vertices for rendering polygon shapes in world coordinates (meters)"""
        if self.sides <= 2:
            return None  # Circle, no vertices needed
        
        # Fixed: handle the case where fixture may not exist in non-convex situations
        if not hasattr(self, 'fixture') or self.fixture is None:
            return None
            
        transform = self.body.transform
        shape = self.fixture.shape
        
        # Added: check if shape has vertices attribute
        if not hasattr(shape, 'vertices'):
            return None
            
        vertices = [transform * v for v in shape.vertices]
        return vertices

    # ---------- Non-convex internal support ----------
    def _generate_star_polygon(self, base_sides, concavity_count, depth_ratio, radius):
        base_sides = max(3, base_sides)
        concavity_count = min(base_sides, concavity_count)
        depth_ratio = max(0.0, min(0.9, depth_ratio))
        concave_indices = set(random.sample(range(base_sides), concavity_count)) if concavity_count > 0 else set()
        verts = []
        for i in range(base_sides):
            theta = 2 * math.pi * i / base_sides
            scale = 1 - depth_ratio if i in concave_indices else 1
            r = radius * scale
            verts.append((r * math.cos(theta), r * math.sin(theta)))
        return verts

    def _triangulate_fan(self, verts):
        if len(verts) < 3:
            return []
        tris = []
        for i in range(1, len(verts)-1):
            tris.append((verts[0], verts[i], verts[i+1]))
        return tris

    def _create_non_convex_polygon(self, world, x, y, radius, base_sides,
                                   concavity_count, concavity_depth_ratio,
                                   density, restitution):
        verts = self._generate_star_polygon(base_sides, concavity_count, concavity_depth_ratio, radius)
        tris = self._triangulate_fan(verts)
        # Directly use meter coordinates
        self.body = world.CreateDynamicBody(position=(x, y))
        
        first_fixture = None  # Added: record the first fixture
        for i, tri in enumerate(tris):
            # Directly use meter coordinates
            fixture = self.body.CreatePolygonFixture(
                vertices=[(vx, vy) for vx, vy in tri],
                density=density,
                restitution=restitution
            )
            if i == 0:  # Added: save the first fixture as the main reference
                first_fixture = fixture
        
        self.fixture = first_fixture  # Added: set self.fixture reference
        self.radius = radius
    # ---------- Non-convex internal support END ----------
