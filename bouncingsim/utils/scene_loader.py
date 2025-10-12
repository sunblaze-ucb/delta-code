import json
import os
from typing import Dict, List, Any

class SceneLoader:
    def __init__(self, scenes_dir: str = "data/scenes"):
        self.scenes_dir = scenes_dir
        if not os.path.exists(scenes_dir):
            os.makedirs(scenes_dir)
    
    def load_scene(self, scene_name: str) -> Dict[str, Any]:
        """Load scene configuration from JSON file"""
        # 支持新的目录结构：data/scenes/数据集名称/scenes_类型/难度/场景文件
        # scene_name 格式可能是: "scenes_A/basic/scene_A_basic_1" 或 "scenes_ABC/hard/scene_ABC_hard_3"
        
        # 首先尝试直接加载（向后兼容）
        scene_path = os.path.join(self.scenes_dir, f"{scene_name}.json")
        
        if not os.path.exists(scene_path):
            # 尝试从新的目录结构中加载
            # 解析场景路径：scenes_类型/难度/场景名
            if "/" in scene_name:
                parts = scene_name.split("/")
                if len(parts) >= 3:
                    # 格式: scenes_类型/难度/场景名
                    scene_path = os.path.join(self.scenes_dir, *parts) + ".json"
                elif len(parts) == 2:
                    # 格式: scenes_类型/场景名
                    scene_path = os.path.join(self.scenes_dir, *parts) + ".json"
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene '{scene_name}' not found. Tried path: {scene_path}")
        
        with open(scene_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_scene(self, scene_name: str, config: Dict[str, Any]):
        """Save scene configuration to JSON file"""
        scene_path = os.path.join(self.scenes_dir, f"{scene_name}.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(scene_path), exist_ok=True)
        
        with open(scene_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def list_scenes(self) -> List[str]:
        """List available scene files"""
        if not os.path.exists(self.scenes_dir):
            return []
        
        scenes = []
        # 递归搜索所有子目录中的场景文件
        for root, dirs, files in os.walk(self.scenes_dir):
            for file in files:
                if file.endswith('.json'):
                    # 计算相对路径
                    rel_path = os.path.relpath(os.path.join(root, file), self.scenes_dir)
                    scene_name = rel_path[:-5]  # Remove .json extension
                    scenes.append(scene_name)
        return scenes
    
    def find_scene_by_id(self, scene_id: str) -> str:
        """根据场景ID查找场景文件路径"""
        if not os.path.exists(self.scenes_dir):
            return None
        
        # 递归搜索包含场景ID的文件
        for root, dirs, files in os.walk(self.scenes_dir):
            for file in files:
                if file.endswith('.json') and scene_id in file:
                    rel_path = os.path.relpath(os.path.join(root, file), self.scenes_dir)
                    scene_name = rel_path[:-5]  # Remove .json extension
                    return scene_name
        
        return None