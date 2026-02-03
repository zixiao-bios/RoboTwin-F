"""
使用 ReproductionEnvWrapper 进行数据采集

与 collect_data.py 相比：
- 使用 wrapper 记录 EEF 轨迹和多视角图像
- 每个相机视角保存为单独的视频
- 轨迹数据保存为 JSON 文件
- 禁用原有的 hdf5/pkl 保存逻辑
"""

import sys
sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
from envs.utils import ReproductionEnvWrapper
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


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


def main(task_name=None, task_config=None, save_fps=30.0):

    task = class_decorator(task_name)
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
            raise "missing embodiment files"
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
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print(f"\033[94mSave FPS:\033[0m {save_fps}")
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"] + "_reproduction")
    
    # 禁用原有保存逻辑
    args["save_data"] = False
    args["save_freq"] = None
    
    run(task, args, save_fps)


def run(TASK_ENV, args, save_fps=30.0):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Create Save Directory ===========
    os.makedirs(args["save_path"], exist_ok=True)

    # =========== Collect Seed & Data ===========
    print("\033[93m" + "[Start Data Collection with Wrapper]" + "\033[0m")
    args["need_plan"] = True

    # 恢复已有进度
    if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            if len(seed_list) != 0:
                seed_list = [int(i) for i in seed_list]
                suc_num = len(seed_list)
                epid = max(seed_list) + 1
        print(f"Exist seed file, Start from: {epid} / {suc_num}")

    # 包装环境
    env = ReproductionEnvWrapper(TASK_ENV, save_fps=save_fps)

    while suc_num < args["episode_num"]:
        try:
            # 初始化环境
            env.setup_demo(now_ep_num=suc_num, seed=epid, **args)
            
            # 开始记录
            env.start_recording(seed=epid, embodiment=args["embodiment"])
            
            # 执行轨迹
            env.play_once()
            
            # 停止记录
            env.stop_recording()

            if env.plan_success and env.check_success():
                print(f"Episode {suc_num} success! (seed = {epid})")
                seed_list.append(epid)
                
                # 保存数据（视频 + 轨迹 JSON）
                env.save_episode(args["save_path"], suc_num)
                
                suc_num += 1
            else:
                print(f"Episode {suc_num} fail! (seed = {epid})")
                fail_num += 1

            env.close_env()

            if args["render_freq"]:
                env.viewer.close()
                
        except UnStableError as e:
            print(" -------------")
            print(f"Episode {suc_num} fail! (seed = {epid})")
            print("Error: ", e)
            print(" -------------")
            fail_num += 1
            env.close_env()

            if args["render_freq"]:
                env.viewer.close()
            time.sleep(0.3)
        except Exception as e:
            print(" -------------")
            print(f"Episode {suc_num} fail! (seed = {epid})")
            print("Error: ", e)
            traceback.print_exc()
            print(" -------------")
            fail_num += 1
            env.close_env()

            if args["render_freq"]:
                env.viewer.close()
            time.sleep(1)

        epid += 1

        # 保存 seed 进度
        with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
            for sed in seed_list:
                file.write("%s " % sed)

    print(f"\nComplete! Success: {suc_num}, Failed: \033[91m{fail_num}\033[0m / {epid} tries\n")


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str, help="Task name")
    parser.add_argument("task_config", type=str, help="Task config file name")
    args = parser.parse_args()

    main(task_name=args.task_name, task_config=args.task_config, save_fps=5)
