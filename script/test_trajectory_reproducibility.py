"""
测试脚本：验证相同 seed 是否能产生相同的规划轨迹

实验目的：
确定 seed 一致是否是规划轨迹一致的充分条件

实验方法：
1. 使用相同的 seed 运行两次环境初始化和 play_once
2. 对比两次产生的轨迹是否完全一致
"""

import sys
sys.path.append("./")

import numpy as np
import yaml
import os
import importlib
from copy import deepcopy

from envs import *

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def prepare_args(task_name, task_config="demo_clean"):
    """准备任务参数"""
    config_path = f"./task_config/{task_config}.yml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise Exception("missing embodiment files")
        return robot_file
    
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    args["render_freq"] = 0
    args["need_plan"] = True
    args["save_data"] = False
    args["save_path"] = "./data"
    
    return args


def compare_trajectories_strict(traj1, traj2, name="trajectory"):
    """严格对比：完全一致"""
    if len(traj1) != len(traj2):
        print(f"  {name}: ❌ 轨迹段数不同 ({len(traj1)} vs {len(traj2)})")
        return False
    
    if len(traj1) == 0:
        print(f"  {name}: ✅ 均为空轨迹")
        return True
    
    all_match = True
    for i, (t1, t2) in enumerate(zip(traj1, traj2)):
        if isinstance(t1, dict) and isinstance(t2, dict):
            for key in t1.keys():
                if key not in t2:
                    print(f"  {name}[{i}]: key '{key}' 缺失")
                    all_match = False
                    continue
                
                v1, v2 = t1[key], t2[key]
                if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                    if v1.shape != v2.shape:
                        print(f"  {name}[{i}]['{key}']: ❌ 形状不同 {v1.shape} vs {v2.shape}")
                        all_match = False
                        continue
                    
                    if not np.allclose(v1, v2, rtol=1e-10, atol=1e-10):
                        max_diff = np.max(np.abs(v1 - v2))
                        print(f"  {name}[{i}]['{key}']: ❌ 数值不同，最大差异 = {max_diff:.2e}")
                        all_match = False
                elif v1 != v2:
                    print(f"  {name}[{i}]['{key}']: ❌ 值不同 ({v1} vs {v2})")
                    all_match = False
        else:
            if not np.array_equal(t1, t2):
                print(f"  {name}[{i}]: ❌ 不匹配")
                all_match = False
    
    if all_match:
        print(f"  {name}: ✅ 完全一致 (段数={len(traj1)})")
    
    return all_match


def compare_trajectories_semantic(traj1, traj2, name="trajectory"):
    """语义对比：只关心段数（任务步骤数）是否相同"""
    if len(traj1) != len(traj2):
        print(f"  {name}: ❌ 段数不同 ({len(traj1)} vs {len(traj2)}) - 任务步骤不一致!")
        return False
    
    if len(traj1) == 0:
        print(f"  {name}: ✅ 均为空轨迹")
        return True
    
    # 统计 step 差异（仅供参考）
    total_steps_1 = 0
    total_steps_2 = 0
    for t1, t2 in zip(traj1, traj2):
        if isinstance(t1, dict) and 'position' in t1:
            total_steps_1 += len(t1['position'])
        if isinstance(t2, dict) and 'position' in t2:
            total_steps_2 += len(t2['position'])
    
    step_diff_percent = abs(total_steps_1 - total_steps_2) / max(total_steps_1, total_steps_2) * 100
    print(f"  {name}: ✅ 段数一致 ({len(traj1)}段), 总步数 {total_steps_1} vs {total_steps_2} (差异 {step_diff_percent:.1f}%)")
    
    return True


def run_experiment(task_name="adjust_bottle", task_config="demo_clean", seed=42, num_runs=2):
    """运行实验"""
    print("=" * 60)
    print(f"实验: 验证 seed={seed} 的轨迹复现性")
    print(f"任务: {task_name}, 配置: {task_config}")
    print("=" * 60)
    
    args = prepare_args(task_name, task_config)
    
    trajectories = []
    actor_positions = []
    
    for run_id in range(num_runs):
        print(f"\n--- 运行 {run_id + 1}/{num_runs} ---")
        
        task_env = class_decorator(task_name)
        
        try:
            # 设置 mplib 的全局随机种子
            import mplib
            mplib.set_global_seed(seed)
            
            task_env.setup_demo(now_ep_num=0, seed=seed, **args)
            
            # 记录物体位置（验证环境是否一致）
            positions = {}
            for attr_name in dir(task_env):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(task_env, attr_name)
                    if hasattr(attr, 'get_pose') and hasattr(attr, 'actor'):
                        pose = attr.get_pose()
                        positions[attr_name] = {
                            'p': pose.p.tolist(),
                            'q': pose.q.tolist()
                        }
                except:
                    pass
            actor_positions.append(positions)
            
            # 运行 play_once 生成轨迹
            task_env.play_once()
            
            # 保存轨迹
            left_traj = deepcopy(task_env.left_joint_path)
            right_traj = deepcopy(task_env.right_joint_path)
            trajectories.append({
                'left': left_traj,
                'right': right_traj,
                'plan_success': task_env.plan_success,
                'check_success': task_env.check_success()
            })
            
            print(f"  plan_success: {task_env.plan_success}")
            print(f"  check_success: {task_env.check_success()}")
            print(f"  left_joint_path 长度: {len(left_traj)}")
            print(f"  right_joint_path 长度: {len(right_traj)}")
            
            task_env.close_env()
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            try:
                task_env.close_env()
            except:
                pass
            return False
    
    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    # 对比物体位置
    print("\n1. 物体位置对比:")
    pos1, pos2 = actor_positions[0], actor_positions[1]
    positions_match = True
    for key in pos1.keys():
        if key not in pos2:
            print(f"  {key}: 缺失")
            positions_match = False
            continue
        p1, p2 = pos1[key], pos2[key]
        if p1['p'] != p2['p'] or p1['q'] != p2['q']:
            print(f"  {key}: 位置不同")
            print(f"    Run 1: p={p1['p']}, q={p1['q']}")
            print(f"    Run 2: p={p2['p']}, q={p2['q']}")
            positions_match = False
    
    if positions_match:
        print("  ✅ 所有物体位置完全一致")
    
    # 对比轨迹 - 严格模式
    print("\n2. 轨迹对比 (严格模式 - 完全一致):")
    traj1, traj2 = trajectories[0], trajectories[1]
    
    left_strict = compare_trajectories_strict(traj1['left'], traj2['left'], "left_joint_path")
    right_strict = compare_trajectories_strict(traj1['right'], traj2['right'], "right_joint_path")
    
    # 对比轨迹 - 语义模式（只关心段数/步骤是否相同）
    print("\n3. 轨迹对比 (语义模式 - 只关心任务步骤数):")
    left_approx = compare_trajectories_semantic(traj1['left'], traj2['left'], "left_joint_path")
    right_approx = compare_trajectories_semantic(traj1['right'], traj2['right'], "right_joint_path")
    
    # 对比任务成功情况
    print("\n4. 任务成功情况:")
    plan_match = traj1['plan_success'] == traj2['plan_success']
    success_match = traj1['check_success'] == traj2['check_success']
    print(f"  plan_success: {'✅ 一致' if plan_match else '❌ 不一致'} ({traj1['plan_success']} vs {traj2['plan_success']})")
    print(f"  check_success: {'✅ 一致' if success_match else '❌ 不一致'} ({traj1['check_success']} vs {traj2['check_success']})")
    
    # 总结
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    
    print("\n【关键问题1】seed 一致时，环境是否完全一致？")
    if positions_match:
        print("  ✅ 是 - 物体位置完全一致")
    else:
        print("  ❌ 否 - 物体位置不一致")
    
    print("\n【关键问题2】seed 一致时，专家能否按相同步骤完成任务？")
    semantic_match = left_approx and right_approx and plan_match and success_match
    if semantic_match:
        print("  ✅ 是 - 任务步骤数相同、规划成功、任务成功")
    else:
        print("  ❌ 否")
        if not (left_approx and right_approx):
            print("     - 任务步骤数不一致")
        if not plan_match:
            print("     - plan_success 不一致")
        if not success_match:
            print("     - check_success 不一致")
    
    return positions_match and semantic_match


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="adjust_bottle", help="任务名称")
    parser.add_argument("--config", type=str, default="demo_clean", help="配置名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--runs", type=int, default=2, help="运行次数")
    args = parser.parse_args()
    
    success = run_experiment(
        task_name=args.task,
        task_config=args.config,
        seed=args.seed,
        num_runs=args.runs
    )
    
    exit(0 if success else 1)
