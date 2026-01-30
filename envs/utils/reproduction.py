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
