"""
打开一个 MuJoCo 实时窗口，把仿真机械臂定在 'home' 关键帧姿态，方便对照着摆真机、
读 read_current_pose.py 的数值来校准 joint_offsets。

窗口里可以自己拖动鼠标转视角/滚轮缩放，看清楚每个关节朝向。关掉窗口结束。
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "SOARM101/SO101/scene_with_table_v.xml"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise RuntimeError("模型里没有找到名为 'home' 的 keyframe")
    d.qpos[:] = m.key_qpos[key_id]
    mujoco.mj_forward(m, d)

    print("Home 姿态目标角度 (度)，对照真机摆成一样的姿态:")
    for name, deg in zip(JOINT_NAMES, np.degrees(m.key_qpos[key_id][:6])):
        print(f"  {name:15s} {deg:8.2f}")
    print("\n窗口已打开，机械臂固定在 Home 姿态。摆好真机后另开终端跑 read_current_pose.py。")
    print("关掉窗口结束。")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # 姿态是固定的 home keyframe，这里只是保持渲染/响应窗口交互，不做物理步进
            mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(0.02)
