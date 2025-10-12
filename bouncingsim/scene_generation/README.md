# 场景生成器重构说明

## 概述

本次重构将场景生成器从屏幕显示系统完全分离，引入了独立的场景空间概念。场景空间可以独立于显示系统进行配置和扩展。

## 主要变化

### 1. 移除屏幕依赖
- 删除了对 `SCREEN_WIDTH` 和 `SCREEN_HEIGHT` 的依赖
- 场景空间现在完全基于物理单位（米）
- 显示系统负责将场景空间映射到屏幕空间

### 2. 引入场景空间配置
- `SCENE_SPACE_CONFIG`: 定义场景空间的基础参数
- 支持根据难度动态缩放场景空间
- 可配置的容器尺寸、间距等参数

### 3. 新的配置结构
```python
SCENE_SPACE_CONFIG = {
    "base_width": 1000.0,      # 基础宽度（米）
    "base_height": 1000.0,     # 基础高度（米）
    "difficulty_scaling": {     # 难度缩放因子
        0: 1.0,    # Basic
        1: 1.2,    # Easy
        2: 1.5,    # Medium
        3: 2.0,    # Hard
        4: 3.0,    # Extreme
    },
    # ... 其他配置
}
```

### 4. 场景空间函数
- `get_scene_dimensions(difficulty)`: 获取指定难度的场景尺寸
- `get_scene_center(difficulty)`: 获取场景中心点
- `get_outer_container_diameter(difficulty)`: 获取外框容器直径

## 使用方式

### 基本场景生成
```python
from scene_generation.scene_generator import generate

# 生成A轴场景，难度2（Medium）
generate("A", 2, 5, "output_scenes")
```

### 自定义场景空间
```python
from scene_generation.scene_config import get_scene_config, validate_scene_config

# 获取大型场景配置
config = get_scene_config("large")
if validate_scene_config(config):
    # 使用配置生成场景
    pass
```

## 预设场景空间

- `small`: 500x500米，适合简单测试
- `medium`: 1000x1000米，标准配置
- `large`: 2000x2000米，复杂场景
- `extreme`: 5000x5000米，极限测试

## 优势

1. **独立性**: 场景生成完全独立于显示系统
2. **可扩展性**: 场景空间可以根据需要无限扩展
3. **物理准确性**: 所有尺寸都基于物理单位
4. **配置灵活性**: 支持多种预设和自定义配置
5. **难度适配**: 场景空间随难度动态调整

## 迁移说明

现有代码无需修改，所有接口保持兼容。新生成的场景文件将包含场景空间元数据：

```json
{
  "meta": {
    "scene_space": {
      "width": 1000.0,
      "height": 1000.0,
      "center": [500.0, 500.0],
      "scale_factor": 1.0,
      "units": "meters"
    }
  }
}
```

## 未来扩展

1. **动态场景空间**: 支持运行时调整场景大小
2. **多区域场景**: 支持复杂的多区域布局
3. **自适应缩放**: 根据内容自动调整场景空间
4. **性能优化**: 大规模场景的生成优化 