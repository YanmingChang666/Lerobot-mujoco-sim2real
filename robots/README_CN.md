# 书写机器人配置（robots/*.yaml）

每个 YAML 文件描述一台书写机器人。新增机器人只需要写一个 YAML，不用写 Python 代码，
也不用改 `train.py` / `args.py`。

**框架总览（名字 ↔ 目录）见 [docs/机器人目录框架.md](../docs/机器人目录框架.md)。**

```bash
python -m writing --list                                   # 列出所有机器人
python -m writing --robot g1_right --text 永 --hold         # 书写，参数与 python -m X02.write 相同
python -m writing --robot g1_right --help
```

原来的入口 `python -m PIPER.write`、`WheelArmR.write`、`X02.write`、`x02cTpose.write` 保持不变，
已训练的权重照常可用。

## 名字对照

| `--robot` | 配置 | 模型目录 | 说明 |
|---|---|---|---|
| `piper` | [piper.yaml](piper.yaml) | [piper/](piper/) | AgileX Piper 六轴 |
| `wheelarm_r` | [wheelarm_r.yaml](wheelarm_r.yaml) | [wheelarm_r/](wheelarm_r/) | WheelArm 单独右臂 |
| `x02` | [x02.yaml](x02.yaml) | [x02/](x02/) | X02 上身右臂七轴 |
| `x02c_tpose` | [x02c_tpose.yaml](x02c_tpose.yaml) | [x02c_tpose/](x02c_tpose/) | x02c T-pose 右臂四轴 |
| `so101` | [so101.yaml](so101.yaml) | [so101/](so101/) | SO-ARM101 |
| `g1_right` | [g1_right.yaml](g1_right.yaml) | [g1_right/](g1_right/) | Unitree G1 右臂 |
| `wheel_robot_r` | [wheel_robot_r.yaml](wheel_robot_r.yaml) | [wheel_robot_r/](wheel_robot_r/) | 轮式整机右臂 |
| `robot0309_r` | [robot0309_r.yaml](robot0309_r.yaml) | [robot0309_r/](robot0309_r/) | robot0309 右臂 |
| `x03_right` | [x03_right.yaml](x03_right.yaml) | [x03_right/](x03_right/) | x3f2 28DoF 右臂 |
| `dofbot` | [dofbot.yaml](dofbot.yaml) | [dofbot/](dofbot/) | Yahboom Dofbot 五轴桌面臂 |

## 目录结构

所有机器人的模型文件（URDF、MJCF、网格）都在本目录下，随仓库提供，克隆后即可运行，不需要先生成模型。
约定：`robots/<名字>.yaml` + `robots/<名字>/{source,assets,outputs}/`。

```text
robots/
├── README_CN.md                     本说明
├── <名字>.yaml                      每台书写机器人一个配置文件
├── piper/assets/                    Piper：piper.xml、源 URDF、meshes/
├── wheelarm_r/assets/               WheelArm 单独右臂：wheel_arm_r_write.xml、旧的 wheel_arm_r.xml、STL
├── x02/assets/                      X02：x02_right_write.xml、urdf/a07.*、meshes/
├── x02c_tpose/assets/               x02c T-pose：right_write.xml、x2c_TPose.urdf、meshes/
├── x2c/assets/                      X2C 全身模型（只作存档，没有书写配置）
├── so101/
│   ├── SO101/                       SO-ARM101 原始模型；旧 Koopman 实验与真机脚本也用它
│   └── assets/so101_write.xml       书写模型（由 SO101/ 生成，网格直接引用 SO101/assets）
├── g1_right/
│   ├── source/unitree_description/  G1 源 URDF + 用到的 35 个网格
│   └── assets/g1_right_write.xml    书写模型（由 source/ 生成）
├── wheel_robot_r/
│   ├── source/                      轮式机器人整机 URDF + meshes/
│   └── assets/wheel_robot_r_write.xml
├── robot0309_r/
│   ├── source/robot0309/            robot0309 ROS 包：URDF、meshes/、launch/、config/
│   └── assets/                      书写模型 + 简化后的底座网格（原底座超过 MuJoCo 面数上限）
├── x03_right/
│   ├── source/                      x3f2_28dof.urdf + meshes/
│   └── assets/x03_right_write.xml   书写模型（右臂 7 轴）
└── dofbot/
    ├── source/dofbot_info/          Dofbot ROS 包：dofbot.urdf、meshes/、package.xml
    └── assets/dofbot_write.xml      书写模型（5 轴，笔装在夹爪侧向）
```

- `source/` 是原始模型，`assets/` 是书写用的 MJCF。由 YAML 生成的模型通过相对路径直接引用
  `source/` 里的网格，不重复复制（`build.copy_meshes: false`）。
- 运行时生成的数据不在这里，也不入库：训练参考与数据缓存在 YAML 的 `reference` / `cache_dir`，
  书写结果在 `output`（新机器人默认是 `robots/<名字>/data`、`robots/<名字>/outputs`）。
- robot0309 的 Isaac USD 导出文件（每个 172 MB）超过 GitHub 上限，不入库，见 `.gitignore`。
- 仓库根目录的 `robot0309/`、`unitree_description/` 是遗留拷贝；正式源分别在
  `robots/robot0309_r/source/`、`robots/g1_right/source/`。

## 新增一台机器人（以 G1 右臂为例）

**1. 看 URDF 里有哪些关节和 link**

```bash
python -m writing.build_model --inspect /path/to/robot.urdf
```

**2. 把源模型放到 `robots/<名字>/source/`**（URDF 和它引用的网格；ROS 包整个放进来，
`package://包名/...` 会自动找到），**然后写 `robots/<名字>.yaml`**（可从 [g1_right.yaml](g1_right.yaml) 复制修改）

```yaml
name: g1_right
train_env: G1RightWriting            # train.py --env 用的名字，也是 results/ 下的目录名
model: robots/g1_right/assets/g1_right_write.xml   # 生成的书写模型放这里
joints: [right_shoulder_pitch_joint, ..., right_wrist_yaw_joint]   # 书写关节，基座→末端
write_defaults:
  center: [0.35, -0.15]              # 纸面中心 XY（世界坐标，m）
  height: 0.75                       # 纸面高度 Z（m）
build:
  urdf: robots/g1_right/source/unitree_description/urdf/g1/main.urdf   # 相对仓库根目录；支持 package://
  copy_meshes: false                 # 引用 source/ 里的网格，不复制
  base_pos: [0, 0, 0.793]            # 基座放在世界的位置
  pen:
    link: right_rubber_hand          # 握笔的 link
    mount: [0.08, 0, 0]              # 握笔点，该 link 坐标系（m）
    direction: [0, 0, -1]            # 握笔点指向笔尖的方向，该 link 坐标系
    length: 0.12
```

**3. 生成模型**（生成的 `assets/` 要一起提交到 git，别人克隆后就不用再生成）

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

## 显示背景

所有机器人加载时统一加上蓝色渐变天空和蓝色网格地面（[PIPER/scene.py](../PIPER/scene.py)），模型文件本身不改，
所以已训练权重的模型校验不受影响；地面不参与碰撞，运动结果与不加背景时完全相同。

- 模型里已有地面（`type="plane"` 的 geom）：只换成网格材质，位置不变。
- 没有地面：自动加一块，高度取“机器人最低点”和“纸面”中较低者再往下 4 mm，不会穿过机身。
  命令行改了 `--height` 时，地面跟着纸面调整。

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
| `write_defaults.approach_seconds` | | 5 | 进场时长；初始姿态离书写姿态很远时加大 |
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
| `build.copy_meshes` | | true | true：把网格复制到 `assets/meshes/`；false：用相对路径引用源网格（推荐，源模型在仓库内时）。超过 20 万面的 STL 都会另存一份简化版 |
| `build.skip_missing_meshes` | | false | 找不到的网格跳过（该零件不显示），不报错 |
| `build.kv` / `default_force` | | 60 / 40 | 速度执行器增益（手臂很重时加大，如 robot0309 用 800） / URDF 没写 effort 时的力矩上限 |
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
