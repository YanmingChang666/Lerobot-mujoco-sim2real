# 论文与实验补充材料

## 当前文件

| 文件或目录 | 用途 |
|---|---|
| **[全文_双线性Koopman_MPC多机械臂书写.md](全文_双线性Koopman_MPC多机械臂书写.md)** | **合并稿（Markdown 源）**：SO-ARM101 抗扰/架构/Sim2Real + 多机械臂书写 + YAML 统一框架 |
| **[koopman_mpc_多机械臂书写_中文.docx](koopman_mpc_多机械臂书写_中文.docx)** | **国内投稿用中文 Word**：由全文 Markdown 生成，含表、四张图与修正后的 17.5× 等表述 |
| [koopman_mpc_soarm101.docx](koopman_mpc_soarm101.docx) | 英文 Word：已同步修正倍数、损失措辞、文献编号 [8]–[12]，文末指向中文完整稿；原稿备份为 `koopman_mpc_soarm101.docx.bak` |
| [build_docx.py](build_docx.py) | 从 Markdown 重建中文 Word，并再次对英文 Word 套用关键替换 |
| [多机械臂书写功能与实验补充.md](多机械臂书写功能与实验补充.md) | 早期多臂补充章节；内容已并入全文，保留便于对照 |
| [evidence/manifest.json](evidence/manifest.json) | 七组结果来源、文件哈希、配置标签及轨迹时长 |
| `evidence/*_trace.npz` | 独立保存的参考/实际笔尖轨迹、落笔标记、控制周期 |
| `evidence/*.json` | 每组运行原始指标的快照 |
| [verify_results.py](verify_results.py) | 从本目录归档轨迹复算位置 RMSE 与最大误差 |
| `figures/` | 全文引用的四张场景/结果图 |

直接运行命令仍统一维护在 [机械臂书写使用教程](../docs/机械臂书写使用教程.md) 与 [演示操作说明](../docs/演示操作说明.md)。

## 全文相对原稿的主要修正

1. **摘要倍数：** “0.263 → 0.015” 写为约 **17.5×**（相对仅 UKF 的负载条件），不再写 56×；完全无补偿的噪声+负载仍约两个数量级改善。
2. **损失：** 谱半径为软惩罚；标准 DBKN 固定投影；\(H\) 的 L1 当前未进总损失。
3. **文献编号：** UKF [8]、ADRC [9]、MuJoCo [10]、CasADi [11]、笔画 [12]。
4. **指标：** SO-ARM101 的 Frobenius 全轨迹范数与多臂每点 RMSE 分节、禁止混表。
5. **范围：** 多臂无接触仿真 ≠ 真机书写；YAML 新平台（G1、整机等）在演示文档有抽检，论文定量归档仍以七组 evidence 为主。
6. **主张：** 不写“首次提出提升空间 UKF”；贡献定为组合、消融、软件复用与可复算实证。

## 重新生成 Word

修改 Markdown 全文后执行：

```bash
conda activate koopman-sim2real   # 需已安装 python-docx
cd /path/to/Lerobot-mujoco-sim2real
python paper/build_docx.py
```

会覆盖写入 `koopman_mpc_多机械臂书写_中文.docx`，并对英文 `koopman_mpc_soarm101.docx` 再次套用关键替换（首次会生成 `.bak`）。

国内投稿请直接使用中文 Word；英文稿仍以 SO-ARM101 主线为主，多臂与 YAML 细节以中文完整稿为准。若主张 YAML 全平台定量结果，先补 `evidence/` 与配对实验。

## 复核数据

```bash
conda activate koopman-sim2real
cd /home/cym/ROS/Arms/isaac_so_arm101/Lerobot-mujoco-sim2real
python paper/verify_results.py
```

数据快照日期为 2026-10-19。复算脚本不依赖原 `outputs/`，也不连接硬件。
