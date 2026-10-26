# X02 右臂汉字书写

[全部机械臂使用教程与命令速查](../docs/机械臂书写使用教程.md)


使用 `robots/x02/assets/urdf/a07.xml` 双臂模型，控制右臂 `joint1_r` 至
`joint7_r`，保留躯干和左臂外观，左臂固定在原模型零位。
复用 Piper/WheelArmR 的笔画规划、连续姿态逆解、5 秒进场、红绿轨迹和 DBKN/MPC 流程。

## 直接书写

```bash
conda activate koopman-sim2real
cd /home/cym/ROS/Arms/isaac_so_arm101/Lerobot-mujoco-sim2real
python -m X02.write --text 永 --hold
```

默认是无需训练的关节速度前馈 + 位置反馈基线。右臂从七关节零位出发，GUI 显示
初始姿态 1 秒，再以 250 个控制步、5 秒平滑运动到第一笔上方。检查到达误差与
速度后，连续执行落笔、书写、抬笔和笔画转移，不在运行中重置关节位置。
红色为参考笔尖路径，绿色为实际路径；浅色为非落笔段。

```bash
# 从世界 +Z 向下看，顺时针旋转 90 度
python -m X02.write --text 永 --rotation-deg -90 --hold

# 仅验证进场，初始角度按 joint1_r ... joint7_r 排列，单位为度
python -m X02.write --text 二 --approach-only --hold \
  --home-deg 0 0 0 0 0 0 0 --approach-seconds 5

# 无窗口，保存带红绿轨迹的三维截图
MUJOCO_GL=egl python -m X02.write --text 永 --headless \
  --snapshot X02/outputs/x02_baseline.png
```

## 坐标、模型和工具

| 项目 | 默认设置 |
|---|---|
| 受控关节 | `joint1_r` … `joint7_r`，7 个独立运动关节 |
| 状态 / 输入 | 笔尖 xyz + 7 个关节角，共 10 维 / 7 维目标关节速度 |
| 控制周期 | 0.02 s，MuJoCo 积分周期 0.002 s |
| 初始关节角 | 七轴全零（仿真参考，不代表已校准真机零位） |
| 零位笔尖位置 | 约 `(0, -0.195991, -0.313305)` m |
| 纸面中心 | `(0.20, -0.25, -0.23)` m，模型基座坐标系 |
| 字框 / 抬笔 | 0.08 m / 0.03 m |
| 笔尖 TCP | `link7_r` 局部 `(0, 0, -0.16)` m |
| 笔姿态 | site 的 +Z 竖直朝下，完整姿态 IK |

纸面 z 为负数是相对模型原点的位置，不是相对现实地板的高度。
X02 第六关节只有约 ±0.52 rad、第七关节约 ±1.3 rad；不能直接使用 WheelArmR 的
纸面位置或 IK 种子。当前默认工作区已验证完整“永”字及其顺时针旋转版本。
调整 `--center`、`--height`、`--size` 后会重新做 IK 与速度检查；不可达则拒绝执行。
改变初始角度后，若 5 秒不足以满足速度限制，需要增加 `--approach-seconds`。

`robots/x02/assets/x02_right_write.xml` 是专用书写模型，原始 URDF/MJCF 未修改。
网格相对引用 `robots/x02/assets/meshes`，无需移动或重复复制 STL。重新生成模型：

```bash
python -m X02.prepare_assets
```

生成器修正原 MJCF 网格相对路径，去掉重叠的双份 visual/collision mesh，
固定左臂关节，添加笔和纸面，并将右臂执行器换成速度接口。
保留右臂原关节轴、范围和惯性。当前采用理想重力补偿、速度增益 60、阻尼 0.2、
armature 0.02、速度限制 ±0.5 rad/s、力矩限制 ±40 N·m；这些是仿真验证参数，
并非厂商提供或实测辨识参数。笔为固定工具，不包含活动夹爪。

## 独立训练 DBKN 并运行 MPC

不能使用 Piper 或 WheelArmR 的权重。即使同为七轴，几何、工作区和动态也不同。

```bash
python -m X02.write --mode plan --text 永 --headless

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train.py \
  --env X02Writing --model DBKN --mode train --suffix x02_write_demo \
  --train_samples 1500 --train_steps 30 --test_samples 32 --test_steps 60 \
  --num_epochs 100 --eval_interval 10 --batch_size 128 --eval_batch_size 128 --device cpu

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m X02.write \
  --text 永 --mode mpc --hold \
  --checkpoint results/X02Writing/x02_write_demo/DBKN/best_model.pt
```

已有本地验证权重位于上述路径，训练缓存、权重和输出受现有 `.gitignore` 规则管理。
标准 DBKN 的 encoder 为 `[10,64,64,64,24]`，提升状态 34 维；不启用 `--u_z`
或 `--use_stable`。采样围绕规划关节路径进行，加载时检查 XML 哈希、周期、维度
及训练关节范围。旋转字或修改工作区后若超出训练覆盖，应重新规划并训练。

## 验证与输出

```bash
python -m unittest X02.test_writing PIPER.test_workflow WheelArmR.test_writing -v
```

检查模型和网格可迁移、左臂固定、零指令稳定、不可达目标拒绝、旋转多笔画、
5 秒连续进场和笔尖姿态跟踪，并回归已有 Piper/WheelArmR 流程。
输出默认在 `X02/outputs/`：

- `<字>_approach.npz/json`：进场参考与实际关节状态、动作、时长、到达误差。
- `<字>_baseline.npz/json/png`：基线轨迹及误差。
- `<字>_mpc.npz/json/png`：MPC 轨迹及误差。

无窗口运行会加速，5 秒表示仿真时间；GUI 使用真实时间调度。
当前禁用碰撞与纸面接触，没有压力、摩擦或落墨模型，未连接真机。
进场是关节插值，不是避障规划；实机需要重新标定零位、TCP、纸面，匹配驱动接口，
并验证碰撞与接触控制，不能直接把仿真零位轨迹发送到机器人。

### 2026-10-19 本地实测

默认字框 8 cm，“永”字 2354 个轨迹点；误差覆盖落笔、抬笔与笔画转移，启动单独记录。

| 流程 | 笔尖 RMSE | 最大位置误差 |
|---|---:|---:|
| 默认基线 | 0.0353 mm | 0.1760 mm |
| 顺时针 90° 基线 | 0.0345 mm | 0.1622 mm |
| 独立 DBKN + MPC | 0.2182 mm | 0.3943 mm |

全部完整执行，通过最大误差 2 mm 检查。9 项集成/回归测试通过。
“永”字启动到达最大角度误差 0.00194°、最大残余速度 0.000292 rad/s。
GUI “二”字启动实测 5.000071 秒，程序正常退出。
这些数值来自理想化无接触仿真，不代表真机书写精度；本例中基线精度优于 MPC。

## 在机器人正前方中间书写

```bash
python -m X02.write --text 永 --center 0.20 0 --height -0.10 --pen-yaw-deg 90 --hold
```

世界坐标 Y=0 为机器人左右中线。此方案将纸面从侧面的 `(0.20,-0.25,-0.23)` m
移到 `(0.20,0,-0.10)` m，并调整笔尖的水平姿态，使其满足 X02 较窄的关节范围。
`--pen-yaw-deg 90` 只改变笔姿态，笔尖仍朝下，不旋转汉字；若需要旋转汉字，
另加 `--rotation-deg -90`，需重新验证整条轨迹是否可达。
原默认侧面设置保留，上述参数显式选择中间方案。

已验证默认 8 cm “永”字基线：2354 点完整执行、5 秒进场，RMSE 0.0237 mm，
最大位置误差 0.1104 mm。结果在 `outputs/centered/`。
这是无接触跟踪验证，不包含自碰撞、纸面接触或实机可行性认证。

侧面工作区的 DBKN 权重不应直接作为中间工作区的已验证模型。
如需中间位置 MPC，请先规划新参考（会替换当前训练参考），以新 suffix 训练：

```bash
python -m X02.write --mode plan --text 永 --center .20 0 --height -.10 --pen-yaw-deg 90 --headless
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train.py \
  --env X02Writing --model DBKN --mode train --suffix x02_center_demo \
  --train_samples 1500 --train_steps 30 --test_samples 32 --test_steps 60 \
  --num_epochs 100 --eval_interval 10 --batch_size 128 --eval_batch_size 128 --device cpu
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m X02.write \
  --text 永 --center .20 0 --height -.10 --pen-yaw-deg 90 --mode mpc --hold \
  --checkpoint results/X02Writing/x02_center_demo/DBKN/best_model.pt
```

## 英文单词

支持单线大写 A–Z，小写输入自动按大写书写；整段英文按 `--size` 等比例缩放。

```bash
python -m X02.write --text Hello --center .20 0 --height -.10 --pen-yaw-deg 90 --rotation-deg -90 --hold
```

多词加引号，如 `--text "HELLO WORLD"`。详细限制和训练命令见 [英文教程](../docs/机械臂书写使用教程.md#英文单词书写)。
