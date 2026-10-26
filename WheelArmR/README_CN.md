# WheelArmR 七轴机械臂汉字书写

[全部机械臂使用教程与命令速查](../docs/机械臂书写使用教程.md)


本目录按 `PIPER` 的流程增加了完整的汉字书写入口。原七轴右臂 STL
直接由 `robots/wheelarm_r/assets/wheel_arm_r_write.xml` 相对引用，不需要移动或重复复制网格。

## 原来为什么不能正常书写

- `wheelarm_sim2sim.py` 原本生成 YZ 平面的空间 8 字，没有汉字笔顺、落笔和抬笔。
- 原模型速度执行器使用默认 `kv=1`，关节阻尼 2、摩擦 0.3，且没有重力补偿。
  从 `[.3,-.6,.2,-1.8,0,0,0]` rad 开始，零速度指令运行 5 秒，实测最大关节漂移约 105.09°。
- 原 `eeframe` 在末端关节原点，不是笔尖，也没有保证笔朝下。
- 旧 MPC 入口不区分原状态参考和提升空间参考，DBKN 会出现参数维度不匹配。
  该参考分支已修复，同时移除了覆盖控制器内部限幅历史指令的操作。

新的 `write.py` 复用 Piper 的控制、日志和可视化流程。原 `wheel_arm_r.xml`
及旧训练入口保留作历史实验；**汉字书写请使用下面的新入口**。

## 直接运行

在仓库根目录执行：

```bash
conda activate koopman-sim2real
cd /home/cym/ROS/Arms/isaac_so_arm101/Lerobot-mujoco-sim2real
python -m WheelArmR.write --text 永 --hold
```

默认使用“关节速度前馈 + 位置误差反馈”基线，无需训练。流程如下：

1. 七个关节从零位开始，GUI 初始显示 1 秒。
2. 用 5 秒连续关节运动到第一笔上方；仿真由速度指令积分，不逐帧改写关节位置。
3. 检查到达角度误差和残余速度，通过后连续落笔、书写、抬笔。
4. 红色显示参考轨迹，绿色显示实际笔尖轨迹；浅色表示进场、抬笔和转移。
5. 输出误差图、NPZ 数据和 JSON 指标；`--hold` 保持窗口。

默认纸面中心 `(0.35,-0.40,0)` m，字框 8 cm，抬笔 3 cm。
世界坐标系对应原模型腰部输出坐标系；底盘和腰部未加载。
零位笔尖坐标为 `(-0.006,-0.9725,0.2495)` m。
关节顺序为 `Link1_R_joint` 至 `Link7_R_joint`，动作单位 rad/s，周期 0.02 秒。

```bash
# 仅检查 5 秒进场，七个 home 参数单位为度
python -m WheelArmR.write --text 永 --approach-only --hold \
  --home-deg 0 0 0 0 0 0 0 --approach-seconds 5

# 无窗口验证并保存三维截图
MUJOCO_GL=egl python -m WheelArmR.write --text 永 --headless \
  --snapshot WheelArmR/outputs/wheelarm_trails.png
```

自定义初始角度若不能在指定时间内满足速度限制，程序会拒绝执行并提示延长时间。
`--headless` 的 5 秒指仿真时间；GUI 按真实时间调度。运行较慢的机器可能超时。

## DBKN + MPC 流程

书写环境为 `WheelArmRWriting`：10 维状态（笔尖 xyz + 7 关节角），7 维速度动作。
与旧 `WheelArmR` 数据、Piper 六轴模型完全分开，不能混用旧 checkpoint。

```bash
python -m WheelArmR.write --mode plan --text 永 --headless

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train.py \
  --env WheelArmRWriting --model DBKN --mode train --suffix wheelarm_write_demo \
  --train_samples 1500 --train_steps 30 --test_samples 32 --test_steps 60 \
  --num_epochs 100 --eval_interval 10 --batch_size 128 --eval_batch_size 128 --device cpu

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m WheelArmR.write \
  --text 永 --mode mpc --hold \
  --checkpoint results/WheelArmRWriting/wheelarm_write_demo/DBKN/best_model.pt
```

训练数据来自参考关节路径附近的随机、正弦和 chirp 激励；采样缓存使用文件哈希。
MPC 显式约束关节速度为 ±0.5 rad/s，加载时校验模型哈希、周期、维度和关节采样覆盖。
改变纸面、字框或模型后应重新规划、训练，并重新检查误差。
当前支持标准 DBKN，不启用 `--use_stable` 或 `--u_z`。

## 模型与适用范围

- 保留原七轴结构、关节范围和惯性；速度执行器 `kv=80`，阻尼 0.2，摩擦 0，
  理想重力补偿，`implicitfast` 积分。它们是书写验证参数，不是已辨识的真机参数。
- 笔沿第七轴坐标系的局部 -Y 方向安装，笔尖距该关节原点 0.16 m；这是演示 TCP。
  没有虚构夹爪模型，笔是刚性固定工具。
- IK 约束完整笔尖姿态，并用相邻关节解的连续性惩罚处理七轴冗余，避免笔画间跳解。
- 当前是无接触轨迹验证：未启用机器人/纸面碰撞，没有压力、摩擦或落墨模型，
  也没有接入真机、UKF 或 KESO。进场插值不是碰撞规划。
- 接真机前需要标定实际 TCP、纸面、关节零位，匹配底层控制接口与动态，并验证接触控制。

## 文件与验证

| 文件 | 用途 |
|---|---|
| `robots/wheelarm_r/assets/wheel_arm_r_write.xml` | 七轴书写模型、笔尖和纸面 |
| `env.py` | 七轴仿真接口 |
| `trajectory.py` | 七轴姿态 IK；复用 Piper 多笔画时间规划 |
| `write.py` | 基线/MPC、5 秒启动与红绿显示入口 |
| `data_collection.py` | 书写任务训练数据适配 |
| `test_writing.py` | 零指令稳定、不可达拒绝、5 秒启动、多笔画跟踪测试 |

```bash
python -m unittest PIPER.test_workflow WheelArmR.test_writing -v
```

结果在 `WheelArmR/outputs/`，训练缓存和结果目录由现有 `.gitignore` 忽略。
`永_approach.npz` 保存启动时间、参考/实测关节、动作与实际关节名称；
`永_baseline.*` / `永_mpc.*` 保存书写误差、参考/实际轨迹及落笔标记。

### 本次实测（2026-10-18）

默认参数、“永”字 2354 个轨迹点，误差包含落笔、抬笔和笔画间转移：

| 控制方式 | RMSE | 最大位置误差 | 最大笔倾角 |
|---|---:|---:|---:|
| 速度前馈 + 位置反馈 | 0.0417 mm | 0.1447 mm | 0.0186° |
| 独立训练 DBKN + MPC | 0.1939 mm | 0.3745 mm | 0.1915° |

两者均通过最大误差 2 mm 的检查。启动为 250 个控制步、5 秒仿真时间，
到达最大关节误差 0.0147°、最大残余速度 0.00175 rad/s。
Piper + WheelArmR 共 6 项测试通过，原 Piper checkpoint 的“二”字 MPC 回归也通过。
上述为理想重力补偿、无接触模型结果，不能作为真实机械臂精度；本例中 MPC 精度低于基线。

## 将字顺时针旋转 90°

```bash
python -m WheelArmR.write --text 永 --rotation-deg -90 --hold
```

角度按世界 +Z 轴右手规则：从 +Z 向纸面看，负数为顺时针，正数为逆时针。
围绕 `--center` 指定的字框中心旋转整个文字布局，纸面也同角度旋转；
笔尖保持向下，5 秒进场的目标由旋转后的第一笔重新计算。
默认 `--rotation-deg 0` 保留原布局。Piper 入口支持相同参数。

MPC 运行也添加 `--rotation-deg -90`。如果超出训练关节覆盖范围，
先用 `--mode plan --text 永 --rotation-deg -90` 重新规划，再按上文训练命令训练新模型。
