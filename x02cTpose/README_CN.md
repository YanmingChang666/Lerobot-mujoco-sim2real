# x02cTpose 四关节右臂书写

本模块以 `robots/x02c_tpose/assets/x2c_TPose.urdf` 为几何依据，保留全身外观，并固定底座、腿部、腰部和左臂。
仅驱动右肩俯仰、右肩侧摆、右肩偏航和右肘四个关节；原 URDF 的腕部为固定关节。
原始 URDF 与 `x2cT.xml` 均保留，专用场景为 `robots/x02c_tpose/assets/right_write.xml`。

## 直接运行

```bash
conda activate koopman-sim2real
cd /home/cym/ROS/Arms/isaac_so_arm101/Lerobot-mujoco-sim2real
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# 汉字，默认无需训练的前馈反馈基线
python -m x02cTpose.write --text 永 --hold

# 字形顺时针旋转 90 度
python -m x02cTpose.write --text 永 --rotation-deg -90 --hold

# 英文单词，输入小写也按大写单线字母书写
python -m x02cTpose.write --text HELLO --rotation-deg -90 --hold

# 从 URDF 零位开始，只验证 5 秒进场
python -m x02cTpose.write --text 二 --approach-only --hold \
  --home-deg 0 0 0 0 --approach-seconds 5

# 无窗口保存三维场景
MUJOCO_GL=egl python -m x02cTpose.write --text 永 --headless \
  --snapshot x02cTpose/outputs/scene.png
```

默认纸面中心为世界坐标 `(0.30,-0.20,0.90)` m，字框 8 cm，抬笔 3 cm。
英文按整段文字缩放进字框；多单词加引号，例如 `--text "HELLO WORLD"`。
默认初始四关节全零，保留 URDF 定义的 T-pose 几何；模型根部固定于世界 z=1.042 m。
零位笔尖约为 `(-0.16733,-0.75264,1.33480)` m。

初始显示 1 秒后，右臂用 5 秒连续速度控制到第一笔上方，检查到达误差和残余速度，
再连续落笔书写。红色为参考路径，绿色为实际路径，浅色为非落笔段。
`--headless` 使用仿真时间，GUI 按真实时间调度；`--hold` 保持 GUI 窗口。

## 与 X02 七轴版本的区别

| 项目 | x02cTpose 实际设置 |
|---|---|
| 关节顺序 | `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `right_shoulder_yaw_joint`, `right_elbow_joint` |
| 关节范围 rad | `[-2.749,2.749]`, `[-0.908,1.57]`, `[-2.618,2.618]`, `[-2.443,0]` |
| 状态 / 动作 | 笔尖 xyz + 4 关节角，共 7 维 / 4 维目标速度 |
| DBKN 编码器 | `[7,64,64,64,24]`，提升状态 31 维 |
| 工具 TCP | `right_wrist_roll_link` 局部 `(0.10,0,-0.16)` m，演示固定笔 |
| 约束方式 | 优先跟踪笔尖位置，软倾角目标和相邻关节连续性约束 |
| 笔姿态限制 | 不保证全程竖直，不支持独立指定工具水平朝向 |

四自由度不能一般性地实现任意六维位姿。本模块没有虚构腕关节，也没有修改真实关节范围。
默认“永”字基线的笔轴最大倾角约 17.7°；`max_pen_tilt_deg` 会记录实际倾斜。
`tracking_pass_2mm` 只判断完整执行和位置误差，不代表笔轴满足竖直要求。

**不要复制 X02 的 `--pen-yaw-deg 90`。** 非零笔姿态参数在此模块被明确拒绝；
旋转字形仍使用 `--rotation-deg`。也不要直接照搬 X02 的 `--center .20 0 --height -.10`，
两套模型的原点、关节结构与工作区不同。默认高度下将中心直接移到 Y=0 的整字测试未通过，
目前仅提供已验证的右侧工作区命令。

## 模型生成与假设

```bash
python -m x02cTpose.prepare_assets
```

生成器从 URDF 读取关节轴、零位变换、限制、惯性和视觉网格，使用已有 `robots/x02c_tpose/assets/meshes`。
T-pose URDF 与原 XML 在肩侧摆、肘部零位和范围上不同，不能混用。
非受控关节焊接在 URDF 零位，根部固定，不执行全身平衡或站立控制。

使用理想重力补偿、速度伺服增益 60、阻尼 0.2、armature 0.02、零摩擦，
控制周期 0.02 s、积分周期 0.002 s、速度限幅 ±0.5 rad/s。
执行器力矩上限沿用 URDF effort（肩俯仰 18 N·m，其余 14 N·m）。这些伺服设置未经真机辨识。
笔、纸面及全身关闭碰撞，不包含接触力、压力或落墨模型。进场也未做避障规划。

## 独立训练和 MPC

已训练的本地模型可直接运行默认工作区：

```bash
python -m x02cTpose.write --text 永 --mode mpc --hold \
  --checkpoint results/X02CTPoseWriting/tpose_write_demo/DBKN/best_model.pt
```

完整重训命令：

```bash
python -m x02cTpose.write --mode plan --text 永 --headless
python train.py --env X02CTPoseWriting --model DBKN --mode train --suffix tpose_write_custom \
  --train_samples 1500 --train_steps 30 --test_samples 32 --test_steps 60 \
  --num_epochs 100 --eval_interval 10 --batch_size 128 --eval_batch_size 128 --device cpu
python -m x02cTpose.write --text 永 --mode mpc --hold \
  --checkpoint results/X02CTPoseWriting/tpose_write_custom/DBKN/best_model.pt
```

权重必须重新训练，不能使用 X02 或 SO-ARM101 的权重。改变文字、旋转、字框或工作区时，
应先用相同参数规划并验证基线，必要时重新训练。当前支持标准 DBKN，不启用 `--u_z` 或 `--use_stable`。
默认训练参考保存于 `x02cTpose/data/writing_reference.npz`，再次 plan 会覆盖它。

## 测试和结果

```bash
python -m unittest x02cTpose.test_writing -v
```

测试检查四关节维度、URDF 限位、固定腕和底座、不可达拒绝、笔姿态参数拒绝、
5 秒连续进场、英文多笔画及位置跟踪。结果默认在 `x02cTpose/outputs/`。
JSON 中 `orientation_constraint=position_with_soft_tilt` 明确标记该模式。
已有位置、旋转、启动和截图参数均可复用，参见 [统一使用教程](../docs/机械臂书写使用教程.md)。
