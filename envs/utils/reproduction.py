"""
环境复现配置模块

用于保存和加载环境复现所需的最小参数集。
"""

import json
import os
import yaml
import importlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

from envs._GLOBAL_CONFIGS import CONFIGS_PATH


@dataclass
class ReproductionConfig:
    """
    环境复现的最小参数集
    
    包含复现一个 rollout 环境所需的全部参数：
    - task_name: 任务名称
    - seed: 随机种子
    - embodiment: 机器人类型
    - domain_randomization: 域随机化设置
    
    使用示例:
        # 保存配置
        config = ReproductionConfig.from_env(task_env, seed=42)
        config.save("rollout_001.json")
        
        # 加载并复现环境
        config = ReproductionConfig.load("rollout_001.json")
        task_env = config.create_env()
        task_env.play_once()
    """
    
    task_name: str
    seed: int
    embodiment: List[Union[str, float]]
    domain_randomization: Dict[str, Any] = field(default_factory=lambda: {
        "random_background": False,
        "cluttered_table": False,
        "clean_background_rate": 1.0,
        "random_head_camera_dis": 0,
        "random_table_height": 0,
        "random_light": False,
        "crazy_random_light_rate": 0,
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """
        保存配置到 JSON 文件
        
        Args:
            path: 保存路径，如 "rollout_001.json"
        """
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "ReproductionConfig":
        """
        从 JSON 文件加载配置
        
        Args:
            path: JSON 文件路径
            
        Returns:
            ReproductionConfig 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            task_name=data["task_name"],
            seed=data["seed"],
            embodiment=data["embodiment"],
            domain_randomization=data.get("domain_randomization", {}),
        )
    
    @classmethod
    def from_env(cls, task_env, seed: int, embodiment: List[Union[str, float]]) -> "ReproductionConfig":
        """
        从现有环境实例提取配置
        
        Args:
            task_env: 任务环境实例 (Base_Task 子类)
            seed: 当前使用的随机种子
            embodiment: 机器人类型配置
            
        Returns:
            ReproductionConfig 实例
        """
        domain_randomization = {
            "random_background": task_env.random_background,
            "cluttered_table": task_env.cluttered_table,
            "clean_background_rate": task_env.clean_background_rate,
            "random_head_camera_dis": task_env.random_head_camera_dis,
            "random_table_height": task_env.random_table_height,
            "random_light": task_env.random_light,
            "crazy_random_light_rate": task_env.crazy_random_light_rate,
        }
        
        return cls(
            task_name=task_env.task_name,
            seed=seed,
            embodiment=embodiment,
            domain_randomization=domain_randomization,
        )
    
    def create_env(self, **extra_args):
        """
        根据配置创建并初始化环境
        
        Args:
            **extra_args: 额外参数，会覆盖默认值，如:
                - render_freq: 渲染频率 (默认 0)
                - save_data: 是否保存数据 (默认 False)
                - camera: 相机配置
                
        Returns:
            初始化好的任务环境实例，可直接调用 play_once()
        """
        # 动态导入任务类
        envs_module = importlib.import_module(f"envs.{self.task_name}")
        env_class = getattr(envs_module, self.task_name)
        task_env = env_class()
        
        # 准备参数
        args = self._prepare_args(**extra_args)
        
        # 初始化环境
        task_env.setup_demo(
            now_ep_num=0,
            seed=self.seed,
            **args
        )
        
        return task_env
    
    def _prepare_args(self, **extra_args) -> Dict[str, Any]:
        """准备 setup_demo 所需的参数"""
        # 加载 embodiment 配置
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        def get_embodiment_file(embodiment_type: str) -> str:
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise ValueError(f"Missing embodiment files for {embodiment_type}")
            return robot_file
        
        def get_embodiment_config(robot_file: str) -> Dict:
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # 解析 embodiment
        if len(self.embodiment) == 1:
            left_robot_file = get_embodiment_file(self.embodiment[0])
            right_robot_file = get_embodiment_file(self.embodiment[0])
            dual_arm_embodied = True
            embodiment_dis = None
        elif len(self.embodiment) == 3:
            left_robot_file = get_embodiment_file(self.embodiment[0])
            right_robot_file = get_embodiment_file(self.embodiment[1])
            dual_arm_embodied = False
            embodiment_dis = self.embodiment[2]
        else:
            raise ValueError("embodiment should have 1 or 3 elements")
        
        # 构建参数
        args = {
            "task_name": self.task_name,
            "domain_randomization": self.domain_randomization,
            "left_robot_file": left_robot_file,
            "right_robot_file": right_robot_file,
            "dual_arm_embodied": dual_arm_embodied,
            "left_embodiment_config": get_embodiment_config(left_robot_file),
            "right_embodiment_config": get_embodiment_config(right_robot_file),
            # 默认值
            "render_freq": 0,
            "save_data": False,
            "save_path": "./data",
            "save_freq": 15,
            "need_plan": True,
            "camera": {
                "head_camera_type": "D435",
                "wrist_camera_type": "D435",
                "collect_head_camera": True,
                "collect_wrist_camera": True,
            },
            "data_type": {
                "rgb": True,
                "depth": False,
                "pointcloud": False,
                "endpose": True,
                "qpos": True,
            },
        }
        
        if embodiment_dis is not None:
            args["embodiment_dis"] = embodiment_dis
        
        # 应用额外参数
        args.update(extra_args)
        
        return args
    
    def __repr__(self) -> str:
        return (
            f"ReproductionConfig(\n"
            f"  task_name='{self.task_name}',\n"
            f"  seed={self.seed},\n"
            f"  embodiment={self.embodiment},\n"
            f"  domain_randomization={self.domain_randomization}\n"
            f")"
        )


class ReproductionEnvWrapper:
    """
    环境包装器，用于记录轨迹和复现环境
    
    功能：
    - 包装 Base_Task 环境，透明代理所有 env 的属性和方法
    - 记录 EEF 轨迹（xyz, 四元数, 夹爪状态）和多视角观测图像
    - 轨迹和图像以相同的 save_fps 保存
    - 自动适配单双臂模式
    - 集成 ReproductionConfig 用于环境复现
    
    使用示例:
        # 包装环境，指定保存帧率
        env = ReproductionEnvWrapper(task_env, save_fps=30.0)
        
        # 开始记录（每次传入当前 episode 的 seed 和 embodiment）
        env.start_recording(seed=42, embodiment=["arx5"])
        
        # 直接调用 env 的方法（无需 env.env.xxx）
        env.play_once()
        
        # 停止记录
        env.stop_recording()
        
        # 获取数据
        trajectory = env.get_trajectory()      # EEF 轨迹（按 save_fps 采样）
        observations = env.get_observations()  # 图像（按 save_fps 采样）
        
        # 保存视频
        env.save_video("episode_001.mp4")
        
        # 获取复现配置
        env.reproduction_config.save("episode_001.json")
    """
    
    def __init__(self, env, save_fps: float = 30.0):
        """
        初始化环境包装器
        
        Args:
            env: Base_Task 环境实例
            save_fps: 保存帧率，轨迹和图像都以该帧率保存（默认 30 FPS）
        """
        # 使用 object.__setattr__ 避免触发 __getattr__
        object.__setattr__(self, 'env', env)
        object.__setattr__(self, 'dual_arm', None)  # 延迟到 start_recording 时获取
        object.__setattr__(self, '_recording', False)
        object.__setattr__(self, '_trajectory', None)
        object.__setattr__(self, '_observations', None)
        object.__setattr__(self, '_step_count', 0)
        object.__setattr__(self, '_original_step', None)
        object.__setattr__(self, 'reproduction_config', None)
        
        # 保存帧率相关（save_freq 和 sim_timestep 延迟到 start_recording 时计算）
        object.__setattr__(self, '_save_fps', save_fps)
        object.__setattr__(self, '_save_freq', None)  # 每多少仿真步保存一帧
        object.__setattr__(self, '_sim_timestep', None)
    
    def __getattr__(self, name):
        """代理访问底层 env 的属性和方法"""
        return getattr(self.env, name)
    
    def _init_trajectory_storage(self):
        """根据单双臂模式初始化轨迹存储"""
        self._trajectory = {
            "left_arm": {
                "position": [], "orientation": [], 
                "gripper_action": [],  # 目标/命令值
                "gripper_state": []    # 真实物理状态
            },
            "timestamps": []
        }
        if self.dual_arm:
            self._trajectory["right_arm"] = {
                "position": [], "orientation": [], 
                "gripper_action": [], 
                "gripper_state": []
            }
    
    def start_recording(self, seed: int, embodiment: List[Union[str, float]]):
        """
        清除旧轨迹，开始记录新轨迹
        
        Args:
            seed: 当前 episode 的随机种子
            embodiment: 机器人类型配置
        """
        # 更新 dual_arm（环境初始化后才能获取）
        self.dual_arm = getattr(self.env, 'dual_arm', True)
        
        # 计算保存间隔（环境初始化后才能获取 scene.timestep）
        sim_timestep = self.env.scene.get_timestep()  # 默认 1/250 秒
        sim_freq = 1.0 / sim_timestep  # 仿真频率，如 250 Hz
        self._save_freq = max(1, int(round(sim_freq / self._save_fps)))  # 如 250/30 ≈ 8
        self._sim_timestep = sim_timestep
        
        # 创建当前 episode 的复现配置
        self.reproduction_config = ReproductionConfig.from_env(
            self.env, seed, embodiment
        )
        
        # 清除旧数据，初始化新轨迹
        self._init_trajectory_storage()
        self._observations = []
        self._step_count = 0
        
        # 保存原始 scene.step 方法
        self._original_step = self.env.scene.step
        
        # 包装 scene.step 方法
        def wrapped_step():
            self._original_step()
            self._record_step()
        
        self.env.scene.step = wrapped_step
        self._recording = True
    
    def stop_recording(self):
        """停止记录并恢复原始 scene.step"""
        if self._original_step is not None:
            self.env.scene.step = self._original_step
            self._original_step = None
        self._recording = False
    
    def _record_step(self):
        """记录当前步的 EEF 状态和观测"""
        if not self._recording:
            return
        
        # 按 _save_freq 间隔记录（轨迹和图像同步保存）
        if self._step_count % self._save_freq == 0:
            self._record_eef_state()
            self._record_observation()
        
        # 更新步数
        self._step_count += 1
    
    def _record_eef_state(self):
        """记录当前 EEF 状态（xyz, 四元数, 夹爪 action 和 state）"""
        # 左臂
        left_pose = self.env.robot.get_left_ee_pose()  # [x, y, z, qw, qx, qy, qz]
        left_gripper_action = self.env.robot.get_left_gripper_val()   # 目标/命令值
        left_gripper_state = self.env.robot.get_left_gripper_state()  # 真实物理状态
        
        self._trajectory["left_arm"]["position"].append(left_pose[:3])
        self._trajectory["left_arm"]["orientation"].append(left_pose[3:])
        self._trajectory["left_arm"]["gripper_action"].append(left_gripper_action)
        self._trajectory["left_arm"]["gripper_state"].append(left_gripper_state)
        
        # 右臂（仅双臂模式）
        if self.dual_arm:
            right_pose = self.env.robot.get_right_ee_pose()
            right_gripper_action = self.env.robot.get_right_gripper_val()
            right_gripper_state = self.env.robot.get_right_gripper_state()
            
            self._trajectory["right_arm"]["position"].append(right_pose[:3])
            self._trajectory["right_arm"]["orientation"].append(right_pose[3:])
            self._trajectory["right_arm"]["gripper_action"].append(right_gripper_action)
            self._trajectory["right_arm"]["gripper_state"].append(right_gripper_state)
        
        # 记录时间戳
        self._trajectory["timestamps"].append(self._step_count)
    
    def _record_observation(self):
        """记录当前步的观测图像"""
        import numpy as np
        
        self.env._update_render()
        self.env.cameras.update_picture()
        rgb_dict = self.env.cameras.get_rgb()
        
        obs = {}
        
        # 头部相机
        if self.env.cameras.collect_head_camera and "head_camera" in rgb_dict:
            obs["head_camera"] = rgb_dict["head_camera"].get("rgb")
        
        # 腕部相机（根据配置和单双臂模式）
        if self.env.cameras.collect_wrist_camera:
            if "left_camera" in rgb_dict:
                obs["left_camera"] = rgb_dict["left_camera"].get("rgb")
            if self.dual_arm and "right_camera" in rgb_dict:
                obs["right_camera"] = rgb_dict["right_camera"].get("rgb")
        
        self._observations.append(obs)
    
    def get_trajectory(self) -> Dict[str, Any]:
        """
        获取记录的轨迹数据
        
        Returns:
            字典包含:
            - left_arm: {"position": (N,3), "orientation": (N,4), 
                         "gripper_action": (N,), "gripper_state": (N,)}
            - right_arm: 同上（仅双臂模式）
            - timestamps: (N,) 步数索引
        """
        import numpy as np
        
        if self._trajectory is None:
            return None
        
        result = {
            "left_arm": {
                "position": np.array(self._trajectory["left_arm"]["position"]),
                "orientation": np.array(self._trajectory["left_arm"]["orientation"]),
                "gripper_action": np.array(self._trajectory["left_arm"]["gripper_action"]),
                "gripper_state": np.array(self._trajectory["left_arm"]["gripper_state"]),
            },
            "timestamps": np.array(self._trajectory["timestamps"])
        }
        
        if self.dual_arm and "right_arm" in self._trajectory:
            result["right_arm"] = {
                "position": np.array(self._trajectory["right_arm"]["position"]),
                "orientation": np.array(self._trajectory["right_arm"]["orientation"]),
                "gripper_action": np.array(self._trajectory["right_arm"]["gripper_action"]),
                "gripper_state": np.array(self._trajectory["right_arm"]["gripper_state"]),
            }
        
        return result
    
    def get_observations(self) -> List[Dict[str, Any]]:
        """
        获取记录的观测图像列表
        
        Returns:
            列表，每个元素是一个字典，包含:
            - head_camera: (H, W, 3) RGB 图像
            - left_camera: (H, W, 3) RGB 图像
            - right_camera: (H, W, 3) RGB 图像（仅双臂模式）
        """
        return self._observations if self._observations is not None else []
    
    def is_recording(self) -> bool:
        """检查是否正在记录"""
        return self._recording
    
    def get_step_count(self) -> int:
        """获取已记录的步数"""
        return self._step_count
    
    def get_save_freq(self) -> int:
        """获取保存间隔（每多少仿真步保存一帧）"""
        return self._save_freq
    
    def get_save_fps(self) -> float:
        """获取保存帧率"""
        return self._save_fps
    
    def save_video(self, path: str, camera_name: str = "head_camera"):
        """
        保存记录的图像为视频
        
        Args:
            path: 视频保存路径，如 "episode_001.mp4"
            camera_name: 使用的相机名称，默认 "head_camera"
        """
        from .images_to_video import images_to_video
        import numpy as np
        
        if not self._observations:
            print("Warning: No observations recorded, cannot save video.")
            return
        
        # 提取指定相机的图像帧
        frames = []
        for obs in self._observations:
            if camera_name in obs and obs[camera_name] is not None:
                frames.append(obs[camera_name])
        
        if not frames:
            print(f"Warning: No frames found for camera '{camera_name}'.")
            return
        
        # 保存视频
        images_to_video(np.array(frames), path, fps=self._save_fps)
    
    def save_episode(self, save_dir: str, episode_id: int):
        """
        保存当前 episode 的所有数据
        
        Args:
            save_dir: 保存目录
            episode_id: episode 编号
        
        保存内容：
        - video/{id}_head_camera.mp4
        - video/{id}_left_camera.mp4
        - video/{id}_right_camera.mp4 (仅双臂)
        - trajectory/{id}.json (EEF轨迹 + 复现配置)
        """
        import numpy as np
        
        # 创建目录
        video_dir = os.path.join(save_dir, "video")
        trajectory_dir = os.path.join(save_dir, "trajectory")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(trajectory_dir, exist_ok=True)
        
        # 保存各相机视频
        camera_names = ["head_camera", "left_camera"]
        if self.dual_arm:
            camera_names.append("right_camera")
        
        for camera_name in camera_names:
            video_path = os.path.join(video_dir, f"{episode_id}_{camera_name}.mp4")
            self.save_video(video_path, camera_name)
        
        # 准备轨迹数据
        trajectory = self.get_trajectory()
        if trajectory is None:
            print("Warning: No trajectory recorded, cannot save trajectory.")
            return
        
        # 转换为可序列化格式
        trajectory_serializable = {
            "left_arm": {
                "position": trajectory["left_arm"]["position"].tolist(),
                "orientation": trajectory["left_arm"]["orientation"].tolist(),
                "gripper_action": trajectory["left_arm"]["gripper_action"].tolist(),
                "gripper_state": trajectory["left_arm"]["gripper_state"].tolist(),
            },
            "timestamps": trajectory["timestamps"].tolist()
        }
        if self.dual_arm and "right_arm" in trajectory:
            trajectory_serializable["right_arm"] = {
                "position": trajectory["right_arm"]["position"].tolist(),
                "orientation": trajectory["right_arm"]["orientation"].tolist(),
                "gripper_action": trajectory["right_arm"]["gripper_action"].tolist(),
                "gripper_state": trajectory["right_arm"]["gripper_state"].tolist(),
            }
        
        # 构建保存数据
        episode_data = {
            "episode_id": episode_id,
            "reproduction_config": self.reproduction_config.to_dict() if self.reproduction_config else None,
            "trajectory": trajectory_serializable,
            "metadata": {
                "save_fps": self._save_fps,
                "save_freq": self._save_freq,
                "sim_timestep": self._sim_timestep,
                "total_steps": self._step_count,
                "total_frames": len(self._observations) if self._observations else 0,
                "dual_arm": self.dual_arm,
            }
        }
        
        # 保存 JSON
        trajectory_path = os.path.join(trajectory_dir, f"{episode_id}.json")
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        print(f"Episode {episode_id} saved: {len(self._observations)} frames, {self._step_count} steps")
    
    def __repr__(self) -> str:
        return (
            f"ReproductionEnvWrapper(\n"
            f"  dual_arm={self.dual_arm},\n"
            f"  save_fps={self._save_fps},\n"
            f"  save_freq={self._save_freq},\n"
            f"  recording={self._recording},\n"
            f"  step_count={self._step_count},\n"
            f"  reproduction_config={self.reproduction_config}\n"
            f")"
        )
