# RoboTwin 环境复现方案

## 需求背景

**场景**：用 policy 在环境中推理，保存 seed，随后用该 seed 重建环境，在重建环境中运行专家轨迹，对比 policy 轨迹和专家轨迹。

**核心需求**：
1. **seed 一致 → 环境完全一致**（物体位置、灯光、纹理等）
2. **seed 一致 → 专家能按相同步骤完成任务**（不关心轨迹的绝对 pose 差异）

## 实验验证

### 测试脚本

```bash
python script/test_trajectory_reproducibility.py --task adjust_bottle --config demo_clean --seed 0
```

### 实验结果

| seed | 环境 | 段数（任务步骤） | 步数差异 | plan_success | check_success |
|------|------|------------------|----------|--------------|---------------|
| 0 | ✅ 一致 | 5 vs 5 ✅ | 0.7% | True vs True ✅ | True vs True ✅ |
| 1 | ✅ 一致 | 5 vs 5 ✅ | 2.3% | True vs True ✅ | True vs True ✅ |
| 2 | ✅ 一致 | 5 vs 5 ✅ | 1.7% | True vs True ✅ | True vs True ✅ |
| 3 | ✅ 一致 | 5 vs 5 ✅ | 0.4% | True vs True ✅ | True vs True ✅ |
| 10 | ✅ 一致 | 5 vs 5 ✅ | 3.4% | True vs True ✅ | True vs True ✅ |
| 42 | ✅ 一致 | 5 vs 5 ✅ | 1.3% | True vs True ✅ | True vs True ✅ |

**randomized 模式同样通过测试。**

## 结论

| 问题 | 答案 |
|------|------|
| **seed 一致 → 环境完全一致？** | ✅ 是 |
| **seed 一致 → 任务步骤相同？** | ✅ 是（段数一致） |
| **seed 一致 → 任务成功？** | ✅ 是（plan_success、check_success 一致） |

**只用 seed 就够了，不需要保存轨迹数据。**

---

## 轨迹规划的差异问题

### 现象

相同 seed 下，两次运行 `play_once()` 产生的轨迹存在微小差异：

```
left_joint_path[0]['position']: 形状 (556, 6) vs (540, 6)  # 步数不同
left_joint_path[4]['position']: 数值差异 3.84e-03 rad      # 约 0.22 度
```

### 原因

规划器使用 **RRT (Rapidly-exploring Random Tree)** 算法，这是一个随机采样算法：
- mplib 底层使用 OMPL，RRT 内部有随机数生成器
- 虽然 `mplib.set_global_seed()` 可以设置种子，但无法完全控制规划器的确定性
- **TOPP (Time-Optimal Path Parameterization)** 时间参数化也存在浮点运算差异

### 差异量化

| 指标 | 差异范围 | 影响 |
|------|----------|------|
| 总步数 | 0.4% ~ 3.4% | 无影响（只是时间参数化不同） |
| 段数（任务步骤数） | 0% | ✅ 完全一致 |
| 起终点 pose | < 0.001 rad (大多数) | 无影响 |
| 任务成功 | 100% 一致 | ✅ |

### 结论

**这些差异不影响任务语义**：
- 任务步骤数（段数）完全一致
- 任务成功情况完全一致
- 只是每段内部的时间步数略有不同（TOPP 参数化导致）

---

## 最终方案

### 最小参数集

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_name` | string | 任务名称 |
| `seed` | int | 随机种子 |
| `embodiment` | list | 机器人类型，如 `["aloha-agilex"]` |
| `domain_randomization` | dict | 域随机化设置 |

### JSON 格式示例

**多轨迹统一保存格式**（`reproduction_configs.json`）：

```json
{
  "episodes": [
    {
      "episode_id": 0,
      "task_name": "beat_block_hammer",
      "seed": 1,
      "embodiment": ["aloha-agilex"],
      "domain_randomization": {
        "random_background": false,
        "cluttered_table": false,
        "clean_background_rate": 1.0,
        "random_head_camera_dis": 0,
        "random_table_height": 0,
        "random_light": false,
        "crazy_random_light_rate": 0
      }
    },
    {
      "episode_id": 1,
      "task_name": "beat_block_hammer",
      "seed": 2,
      "embodiment": ["aloha-agilex"],
      "domain_randomization": { ... }
    },
    ...
  ]
}
```

### 使用方式

#### 方式1：使用 `ReproductionConfig` 类（推荐）

```python
from envs.utils import ReproductionConfig
import json

# ========== 保存多个轨迹配置 ==========
all_configs = {"episodes": []}

for episode_id, seed in enumerate(successful_seeds):
    # 创建环境并运行
    task_env.setup_demo(seed=seed, ...)
    task_env.play_once()
    
    # 提取配置
    config = ReproductionConfig.from_env(
        task_env, seed=seed, embodiment=["aloha-agilex"]
    )
    
    # 添加到列表
    all_configs["episodes"].append({
        "episode_id": episode_id,
        **config.to_dict()
    })

# 保存到统一 JSON 文件
with open("reproduction_configs.json", "w") as f:
    json.dump(all_configs, f, indent=2)

# ========== 复现环境 ==========
# 加载配置
with open("reproduction_configs.json", "r") as f:
    data = json.load(f)

for episode_data in data["episodes"]:
    config = ReproductionConfig(
        task_name=episode_data["task_name"],
        seed=episode_data["seed"],
        embodiment=episode_data["embodiment"],
        domain_randomization=episode_data["domain_randomization"],
    )
    
    # 一行代码复现环境
    task_env = config.create_env()
    task_env.play_once()
    
    print(f"Episode {episode_data['episode_id']}: success={task_env.check_success()}")
    task_env.close_env()
```

#### 方式2：使用回放脚本（命令行）

```bash
# 回放所有轨迹并保存视频
python script/replay_with_expert.py \
    --config data/beat_block_hammer/demo_clean/reproduction_configs.json \
    --output_dir data/beat_block_hammer/demo_clean/video_replay/

# 只回放指定 episode
python script/replay_with_expert.py \
    --config data/beat_block_hammer/demo_clean/reproduction_configs.json \
    --output_dir data/beat_block_hammer/demo_clean/video_replay/ \
    --episode 0 1 3

# 显示 viewer（不保存视频）
python script/replay_with_expert.py \
    --config data/beat_block_hammer/demo_clean/reproduction_configs.json \
    --render
```

**输出示例**：
```
============================================================
环境复现与专家轨迹回放
============================================================

加载配置: data/beat_block_hammer/demo_clean/reproduction_configs.json
共 5 个 episode

--- Episode 0 ---
task_name: beat_block_hammer, seed: 1
  plan_success: True, check_success: True
  视频保存到: data/beat_block_hammer/demo_clean/video_replay/episode0.mp4

--- Episode 1 ---
...

============================================================
回放结果统计
============================================================
总 episode 数: 5
plan_success: 5/5 (100.0%)
check_success: 5/5 (100.0%)
```

### API 参考

#### `ReproductionConfig` 类

```python
class ReproductionConfig:
    # 属性
    task_name: str              # 任务名称
    seed: int                   # 随机种子
    embodiment: list            # 机器人类型
    domain_randomization: dict  # 域随机化设置
    
    # 方法
    @classmethod
    def from_env(cls, task_env, seed: int, embodiment: list) -> ReproductionConfig
        """从现有环境实例提取配置"""
    
    def create_env(self, **extra_args) -> Base_Task
        """根据配置创建并初始化环境"""
    
    def to_dict(self) -> dict
        """转换为字典"""
```

#### `replay_with_expert.py` 脚本

| 参数 | 类型 | 说明 |
|------|------|------|
| `--config` | str | reproduction_configs.json 文件路径（必需） |
| `--output_dir` | str | 输出视频目录 |
| `--episode` | int... | 要回放的 episode ID（可多个，不指定则全部） |
| `--render` | flag | 是否显示 viewer |

---

## 技术细节

### seed 控制的随机化

1. **环境随机化**（完全确定）：
   - 桌子高度偏移 (`table_z_bias`)
   - 灯光颜色（`random_light=True` 时）
   - 背景纹理（`random_background=True` 时）
   - 相机偏移（`random_head_camera_dis>0` 时）
   - 物体初始位置（`load_actors()` 中）
   - 杂乱物体（`get_cluttered_table()` 中）

2. **轨迹规划**（步数有微小差异，但步骤一致）：
   - RRT 路径规划
   - TOPP 时间参数化

### 前提条件

- **代码不能修改**：任何对随机调用的增删改都会破坏复现性
- **随机调用顺序不能变**：seed 产生固定序列，改变调用顺序会导致后续值不同

---

## 数据目录结构

```
data/{task_name}/{task_config}/
├── seed.txt                    # 成功的 seed 列表
├── reproduction_configs.json   # 所有 episode 的复现配置（核心文件）
├── data/
│   ├── episode0.hdf5           # 观测数据
│   └── ...
├── scene_info.json             # 场景信息
├── video/                      # 原始采集视频
│   ├── episode0.mp4
│   └── ...
└── video_replay/               # 回放生成的视频（可选）
    ├── episode0.mp4
    └── ...
```

**核心文件**：
- `reproduction_configs.json`：包含所有 episode 的复现参数，用于环境复现和专家轨迹回放

**注意**：`_traj_data/` 目录可选，如果只需要复现任务步骤和成功情况，不需要保存轨迹数据。
