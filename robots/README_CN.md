# 书写机器人配置（robots/*.yaml）

每个 YAML 文件描述一台书写机器人。新增机器人只需要写一个 YAML，不用写 Python 代码，
也不用改 `train.py` / `args.py`。

```bash
python -m writing --list                                   # 列出所有机器人
python -m writing --robot g1_right --text 永 --hold         # 书写，参数与 python -m X02.write 相同
python -m writing --robot g1_right --help
```

原来的入口 `python -m PIPER.write`、`WheelArmR.write`、`X02.write`、`x02cTpose.write` 保持不变，
已训练的权重、数据缓存全部照常可用。

## 新增一台机器人（以 G1 右臂为例）

**1. 看 URDF 里有哪些关节和 link**

```bash
python -m writing.build_model --inspect /path/to/robot.urdf
```

**2. 写 `robots/<名字>.yaml`**（可从 [g1_right.yaml](g1_right.yaml) 复制修改）

```yaml
name: g1_right
train_env: G1RightWriting            # train.py --env 用的名字，也是 results/ 下的目录名
model: robots/g1_right/assets/g1_right_write.xml   # 生成的模型放这里
joints: [right_shoulder_pitch_joint, ..., right_wrist_yaw_joint]   # 书写关节，基座→末端
write_defaults:
  center: [0.35, -0.15]              # 纸面中心 XY（世界坐标，m）
  height: 0.75                       # 纸面高度 Z（m）
build:
  urdf: ../PACE/.../g1/main.urdf     # 相对仓库根目录，或绝对路径；支持 package:// 网格路径
  base_pos: [0, 0, 0.793]            # 基座放在世界的位置
  pen:
    link: right_rubber_hand          # 握笔的 link
    mount: [0.08, 0, 0]              # 握笔点，该 link 坐标系（m）
    direction: [0, 0, -1]            # 握笔点指向笔尖的方向，该 link 坐标系
    length: 0.12
```

**3. 生成模型**

```bash
python -m writing.build_model --robot g1_right
```

输出会打印每个关节的范围、home 姿态下笔尖的位置和笔的方向、纸面离第一个关节的水平距离。
据此调整 `pen` 和 `write_defaults`：笔尖方向应大致朝下（`pen axis` ≈ `[0, 0, -1]`），
纸面应在手臂够得到的范围内。

**4. 先不训练，直接用 baseline 书写**

```bash
python -m writing --robot g1_right --text 二 --hold
```

第一次运行会自动搜索 IK 起始姿态，并打印一行 `- [...]`。把它复制到 YAML 的 `ik.seeds` 下，
以后就跳过搜索（只有改了纸面位置、字、旋转等导致这个姿态不再可用时，才会重新自动搜索）。

**5. 训练 DBKN 并用 MPC 书写**（与原有机器人流程相同）

```bash
python -m writing --robot g1_right --mode plan --text 二 --headless
python train.py --env G1RightWriting --model DBKN --mode train --suffix g1_demo \
  --train_samples 1500 --train_steps 30 --test_samples 32 --test_steps 60 \
  --num_epochs 100 --eval_interval 10 --batch_size 128 --eval_batch_size 128 --device cpu
python -m writing --robot g1_right --text 二 --mode mpc --hold \
  --checkpoint results/G1RightWriting/g1_demo/DBKN/best_model.pt
```

## 生成的模型包含什么

- 只有 `joints` 列出的关节能动；其余关节（腿、腰、另一只手臂、手指）固定在 0
  或 `build.fixed_joints` 给定的角度（单位：度），基座固定在世界中
- 每个书写关节一个 `<velocity>` 执行器，顺序与 `joints` 相同（控制量 = rad/s）
- 所有 body 重力补偿，不计算碰撞（与原有机器人一致：不模拟纸面接触）
- 笔（capsule）、`pen_tip` site（+Z 沿笔指向笔尖）、`paper` 纸面

已经有 MJCF、不需要从 URDF 生成的机器人，可以不写 `build:`，只要 MJCF 满足上面的约定
（关节名、`<velocity>` 执行器、`pen_tip`、`paper`）。加载时会逐条检查并报告缺什么。

## 关节数与笔姿态（`ik.mode`，默认 `auto` 按关节数自动选）

| 模式 | 自动选择条件 | 笔姿态 | `--pen-yaw-deg` |
|---|---|---|---|
| `full_pose` | ≥ 6 个关节 | 笔尖竖直朝下，绕笔的转角也固定 | 支持 |
| `pen_axis` | 5 个关节 | 笔尖竖直朝下，绕笔的转角自由 | 必须为 0 |
| `position_soft_tilt` | ≤ 4 个关节 | 只跟踪笔尖位置，尽量减小倾斜，不保证竖直 | 必须为 0 |

## 全部字段

| 字段 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `name` | 是 | | `--robot` 用的名字 |
| `train_env` | 是 | | `train.py --env` 的值，不能与其他机器人重复 |
| `model` | 是 | | MJCF 路径 |
| `description` | | | `--list` 显示的说明 |
| `joints` | 有 `build:` 时必填 | 模型里执行器驱动的关节 | 书写关节，基座→末端 |
| `reference` | | `robots/<name>/data/writing_reference.npz` | `--mode plan` 保存的训练参考 |
| `cache_dir` | | `robots/<name>/data` | 训练数据缓存 |
| `output` | | `robots/<name>/outputs` | 书写结果 |
| `metadata_name` | | `<name>_writing_data.json` | 权重目录里的元信息文件名 |
| `write_defaults.center` / `height` | | `[0.30, 0]` / `0` | 纸面位置（m） |
| `write_defaults.home_deg` | | 全 0 | 初始关节角（度） |
| `write_defaults.rotation_deg` / `pen_yaw_deg` | | 0 | 命令行 `--rotation-deg` / `--pen-yaw-deg` 的默认值 |
| `write_defaults.camera_distance` / `camera_height` | | 1.1 / 纸面+0.1 | 窗口视角 |
| `ik.mode` | | `auto` | 见上表 |
| `ik.seeds` | | 无 | IK 起始姿态（rad），按顺序尝试 |
| `ik.random_seeds` | | 300 | seeds 都不行时随机搜索的数量；0 表示不搜索 |
| `ik.continuity_weight` | | 按模式 | 冗余关节靠近上一个姿态的权重 |
| `build.urdf` | | | 源 URDF |
| `build.pen.link` / `mount` / `direction` / `length` | link 必填 | `[0,0,0]` / `[0,0,-1]` / 0.12 | 笔的安装 |
| `build.base_pos` | | URDF 原点 | 基座位置 |
| `build.fixed_joints` | | 全部 0 | 非书写关节的固定角度，`{关节名: 度}` |
| `build.package_dirs` | | 自动向上查找 | `package://` 找不到时手动指定 `{包名: 路径}` |
| `build.copy_meshes` | | true | 把网格复制到模型旁边 |
| `build.kv` / `default_force` | | 60 / 40 | 速度执行器增益 / URDF 没写 effort 时的力矩上限 |
| `build.damping` / `armature` | | 0.2 / 0.02 | 书写关节阻尼 / 转子惯量 |
| `build.floor_height` | | 无地面 | 地面高度 |
| `overrides.env_class` / `planner` / `data_generator` | | 通用实现 | `模块:名字`，用自己的类替换通用实现（原有 4 台机器人用它保持原行为） |

## 仍然做不到的

- 只能在水平纸面上写（文字只绕竖直轴旋转），不支持白板等竖直平面
- 基座固定、理想重力补偿：人形机器人站立写字需要另外的平衡控制
- 笔刚性固定在 link 上；不模拟灵巧手的抓握，也不模拟纸面接触和压力
- 关节必须是单自由度的转动或平移关节，速度控制
- 只在仿真中验证；除 SO-ARM101 的旧流程外没有真机驱动
- DBKN 权重不能跨机器人复用；换字、换位置后要重新 plan 和训练
