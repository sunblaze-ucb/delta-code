#!/usr/bin/env python3
"""
演示不同预设和难度的效果
"""

from scene_config import get_scene_config, explain_preset_difficulty_relationship, get_recommended_preset_for_difficulty

def demo_config_effects():
    print("=== 不同预设 + 不同难度的效果 ===")
    print("预设\\难度  0    4")
    print("-" * 25)
    
    presets = ['compact', 'standard', 'spacious', 'massive']
    difficulties = [0, 4]
    
    for preset in presets:
        config = get_scene_config(preset)
        size0 = config['base_width'] * config['difficulty_scaling'][0]
        size4 = config['base_width'] * config['difficulty_scaling'][4]
        print(f"{preset:8} {size0:5.0f} {size4:5.0f}")
    
    print("\n=== 推荐组合 ===")
    for diff in range(5):
        recommended = get_recommended_preset_for_difficulty(diff)
        config = get_scene_config(recommended)
        final_size = config['base_width'] * config['difficulty_scaling'][diff]
        print(f"难度{diff}: 推荐 {recommended} 预设 -> 最终尺寸: {final_size:.0f}x{final_size:.0f}米")

def main():
    demo_config_effects()
    print("\n" + "="*50)
    explain_preset_difficulty_relationship()

if __name__ == "__main__":
    main() 