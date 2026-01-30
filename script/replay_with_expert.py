"""
环境复现与专家轨迹回放脚本

使用 ReproductionConfig 复现环境，运行专家轨迹并渲染保存视频。

使用示例:
    # 回放所有 episode
    python script/replay_with_expert.py \
        --config data/adjust_bottle/demo_clean/reproduction_configs.json \
        --output_dir replay_videos/
    
    # 只回放指定 episode
    python script/replay_with_expert.py \
        --config data/adjust_bottle/demo_clean/reproduction_configs.json \
        --episode 0 \
        --output replay_video.mp4
"""

import sys
sys.path.append("./")

import os
import argparse
import json
import imageio
import numpy as np

from envs.utils import ReproductionConfig


def load_all_configs(config_path: str) -> list:
    """
    从统一的 JSON 文件加载所有 episode 配置
    
    Args:
        config_path: reproduction_configs.json 文件路径
        
    Returns:
        配置列表，每个元素包含 episode_id 和 ReproductionConfig
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    configs = []
    for episode_data in data["episodes"]:
        episode_id = episode_data["episode_id"]
        config = ReproductionConfig(
            task_name=episode_data["task_name"],
            seed=episode_data["seed"],
            embodiment=episode_data["embodiment"],
            domain_randomization=episode_data.get("domain_randomization", {}),
        )
        configs.append({
            "episode_id": episode_id,
            "config": config,
        })
    
    return configs


def replay_single_episode(
    config: ReproductionConfig,
    episode_id: int,
    output_path: str = None,
    render: bool = False,
):
    """
    回放单个 episode 并录制视频
    
    Args:
        config: ReproductionConfig 实例
        episode_id: Episode ID
        output_path: 输出视频路径（可选）
        render: 是否显示 viewer
        
    Returns:
        结果字典
    """
    print(f"\n--- Episode {episode_id} ---")
    print(f"task_name: {config.task_name}, seed: {config.seed}")
    
    import tempfile
    import shutil
    
    # 使用临时目录保存数据
    temp_dir = tempfile.mkdtemp()
    
    # 创建环境，启用数据保存以录制视频
    extra_args = {
        "render_freq": 1 if render else 0,
        "save_data": True if output_path else False,
        "save_path": temp_dir,
        "save_freq": 15,  # 与数据收集一致，每15步保存一帧
        "need_plan": True,
        "data_type": {
            "rgb": True,
            "depth": False,
            "pointcloud": False,
            "endpose": False,
            "qpos": False,
        },
    }
    task_env = config.create_env(**extra_args)
    
    # 执行 play_once
    task_env.play_once()
    
    # 获取结果
    plan_success = task_env.plan_success
    check_success = task_env.check_success()
    
    print(f"  plan_success: {plan_success}, check_success: {check_success}")
    
    # 保存视频
    if output_path:
        # 合并为视频
        task_env.merge_pkl_to_hdf5_video()
        
        # 复制视频到目标路径
        video_dir = os.path.join(temp_dir, "video")
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
            if video_files:
                src_video = os.path.join(video_dir, video_files[0])
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                shutil.copy(src_video, output_path)
                print(f"  视频保存到: {output_path}")
    
    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 关闭环境
    task_env.close_env()
    
    if render and hasattr(task_env, 'viewer'):
        task_env.viewer.close()
    
    return {
        "episode_id": episode_id,
        "plan_success": plan_success,
        "check_success": check_success,
    }


def replay_all_episodes(
    config_path: str,
    output_dir: str = None,
    episode_ids: list = None,
    render: bool = False,
):
    """
    回放所有或指定的 episode
    
    Args:
        config_path: reproduction_configs.json 文件路径
        output_dir: 输出视频目录（可选）
        episode_ids: 要回放的 episode ID 列表（None 表示全部）
        render: 是否显示 viewer
    """
    print("=" * 60)
    print("环境复现与专家轨迹回放")
    print("=" * 60)
    
    # 加载所有配置
    print(f"\n加载配置: {config_path}")
    all_configs = load_all_configs(config_path)
    print(f"共 {len(all_configs)} 个 episode")
    
    # 筛选要回放的 episode
    if episode_ids is not None:
        all_configs = [c for c in all_configs if c["episode_id"] in episode_ids]
        print(f"将回放 {len(all_configs)} 个指定 episode: {episode_ids}")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 回放每个 episode
    results = []
    for item in all_configs:
        episode_id = item["episode_id"]
        config = item["config"]
        
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"episode{episode_id}.mp4")
        
        result = replay_single_episode(config, episode_id, output_path, render)
        results.append(result)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("回放结果统计")
    print("=" * 60)
    
    total = len(results)
    plan_success_count = sum(1 for r in results if r["plan_success"])
    check_success_count = sum(1 for r in results if r["check_success"])
    
    print(f"总 episode 数: {total}")
    print(f"plan_success: {plan_success_count}/{total} ({plan_success_count/total*100:.1f}%)")
    print(f"check_success: {check_success_count}/{total} ({check_success_count/total*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description="环境复现与专家轨迹回放")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="reproduction_configs.json 文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="输出视频目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出视频路径（仅用于单个 episode）"
    )
    parser.add_argument(
        "--episode",
        type=int,
        nargs="+",
        default=None,
        help="要回放的 episode ID（可指定多个，不指定则回放全部）"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="是否显示 viewer"
    )
    
    args = parser.parse_args()
    
    # 处理输出路径
    output_dir = args.output_dir
    if args.output and args.episode and len(args.episode) == 1:
        # 单个 episode 指定输出路径
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    
    replay_all_episodes(
        config_path=args.config,
        output_dir=output_dir,
        episode_ids=args.episode,
        render=args.render,
    )
