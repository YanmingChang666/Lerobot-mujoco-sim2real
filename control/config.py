"""
控制实验的配置管理与结果落盘/对比工具。

DBKN_KESO.py / IBKN_UKF.py 每跑一次闭环仿真（某个轨迹形状 + 是否加噪声/
负载/UKF/ESO 的某种组合），都会实例化一个 Config，得到统一规则拼出来的
结果文件路径，跑完仿真调用 save_result() 落盘成 .npz。

ConfigGenerator / ConfigGenerator_DBKN 则是"实验矩阵"工具：给定一个参数
空间（比如噪声开关 x UKF开关 x 轨迹形状的笛卡尔积），批量生成所有组合对应
的 Config，再挨个 load_result() 读回来汇总，画成 README 里那些
"noise_robustness.png"/"payload_robustness_3rows.png" 对比图和误差表格。
两者分别对应 IBKN+UKF 和 DBKN+KESO 两组实验，因为后者比前者多了负载(payload)
和扩张状态观测器(ESO)两个维度，所以拆成了两个类而不是共用一个。
"""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))


class Config:
    """
    单次控制实验的配置 + 结果读写。

    所有参数打包存在 self.dict 里，同时按需要的几个 flag 属性(KEM_FLAG/
    NOISE_FLAG/ESO_FLAG)拼出一个能唯一标识"这次实验用了哪些开关组合"的
    文件名后缀，保证不同配置的结果不会互相覆盖，也能反查某个具体配置
    对应的 .npz 文件。
    """

    def __init__(self, **kwargs):
        self.dict = kwargs
        # 实验基础配置
        self.suffix = kwargs["suffix"]
        self.env_name = kwargs["env_name"]
        self.method = kwargs["method"]
        # 功能开关
        self.use_KEM = kwargs["use_KEM"]
        self.use_nosie = kwargs["use_nosie"]
        self.use_eso = kwargs["use_eso"]
        self.use_payload = kwargs["use_payload"]
        # method 参数
        self.traj_name = kwargs["traj_name"]
        self.save_path = os.path.join(current_dir, "ControlResults", self.suffix)

    # 派生参数（自动生成）
    @property
    def KEM_FLAG(self):
        """是否启用 UKF(无迹卡尔曼滤波) 状态估计，用于结果文件名区分。"""
        return "_UKF" if self.use_KEM else ""

    @property
    def NOISE_FLAG(self):
        """是否在仿真中注入观测/过程噪声，用于结果文件名区分。"""
        return "_noise" if self.use_nosie else ""

    @property
    def ESO_FLAG(self):
        """是否启用扩张状态观测器(ESO)估计并补偿未建模扰动，用于结果文件名区分。"""
        return "_eso" if self.use_eso else ""

    def save_result(self, u, actual_traj, traj, JointAngles):
        """
        把一次闭环仿真的控制序列/实际轨迹/参考轨迹/关节角，连同误差指标
        和完整配置一起存成 .npz，文件名由 method+各 flag+payload 等级拼接
        而成，保证同一 suffix/traj_name 下不同实验配置各自独立存档。
        """
        save_path = os.path.join(self.save_path, self.traj_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sub_path = f"{self.method}{self.KEM_FLAG}{self.ESO_FLAG}{self.NOISE_FLAG}_{self.use_payload}"
        # 用整条轨迹的全局范数衡量跟踪误差：位置误差看末端 XYZ(actual_traj[:, :3])，
        # 姿态/关节误差看剩余的关节角部分，是粗粒度的"一次实验一个数"总评分，
        # 更细粒度的逐时间步误差需要另外从 actual_traj/traj 里重新计算。
        error_dis = np.linalg.norm(actual_traj[:, :3] - traj)
        error_angle = np.linalg.norm(actual_traj[:, 3:] - JointAngles)
        data_dict = {
            "u": u,
            "traj": actual_traj,
            "ref_traj": traj,
            "error_dis": error_dis,
            "error_angle": error_angle,
            "config_dict": self.dict,
        }
        print(error_dis, error_angle)
        np.savez(os.path.join(save_path, f"{sub_path}.npz"), **data_dict)

    def load_result(self):
        """按当前配置对应的文件名读回之前 save_result() 存下的 .npz 结果。"""
        save_path = os.path.join(self.save_path, self.traj_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sub_path = f"{self.method}{self.KEM_FLAG}{self.ESO_FLAG}{self.NOISE_FLAG}_{self.use_payload}"
        data_dict = np.load(os.path.join(save_path, f"{sub_path}.npz"), allow_pickle=True)

        return data_dict


class ConfigGenerator:
    """
    IBKN + UKF 实验（噪声鲁棒性）的配置矩阵生成器与结果汇总/画表工具。
    只涉及"是否加噪声" x "是否开 UKF" 两个开关，比 DBKN 那组实验简单，
    所以单独用一个更精简的类。
    """

    def __init__(self, base_config):
        self.base = base_config
        # 需要变化的参数空间
        self.parameter_space = {
            "method": ["IBKN"],
            "traj_name": ["Fig8", "FigStar"],
            "use_nosie": [True, False],
            "use_KEM": [True, False],
        }
        # 轨迹配置信息
        self.traj_config = {"Fig8": {"title": "Fig8"}, "FigStar": {"title": "FigStar"}}
        # 方法配置信息
        self.method_config = {
            "IBKN": {"name": "IBKN", "color": "red", "line_style": "--"},
        }
        self.robustness_config = {
            "none": {"name": "IBKN-δMPC", "alpha": 1.0, "color": "red", "line_style": "-"},
            "KEM": {"name": "IBKN-δMPC-UKF", "alpha": 1.0, "color": "blue", "line_style": "-"},
        }

    # 生成所有有效配置组合
    def generate_configs(self, space):
        """对参数空间(dict of list)做笛卡尔积，展开成一组组具体的配置字典。"""
        keys = list(space.keys())
        value_combinations = itertools.product(*(space[key] for key in keys))

        configs = []
        seen = set()
        for combo in value_combinations:
            config = self.base.copy()
            for i, key in enumerate(keys):
                config[key] = combo[i]
            # 不加噪声时"是否开 UKF"这个开关没有意义(UKF 是用来抗噪声/滤波的)，
            # 强制关闭避免生成一堆等价于"无噪声无 UKF"的重复配置。
            if not config["use_nosie"]:
                config["use_KEM"] = False
            # 创建唯一标识符
            config_id = tuple(sorted(config.items()))
            # 检查是否重复
            if config_id not in seen:
                seen.add(config_id)
                configs.append(config)

        return configs

    def generate_for_comparison(self, scenario):
        """生成特定场景的配置"""
        scenarios = {
            # InvertKoopmanBlinear噪声鲁棒性测试
            "noise_robustness": {
                "method": ["IBKN"],
                "traj_name": ["Fig8", "FigStar"],
                "use_nosie": [True],
                "use_KEM": [True, False],
            },
            "base_performance": {
                "method": ["IBKN"],
                "traj_name": ["Fig8", "FigStar"],
                "use_nosie": [False],
                "use_KEM": [False],
            },
        }

        if scenario not in scenarios:
            raise ValueError(f"未知场景: {scenario}. 可用场景: {list(scenarios.keys())}")

        return self.generate_configs(scenarios[scenario])

    def get_result(self, scenario):
        """按场景批量 load_result()，重新组织成 {轨迹名: {方法/鲁棒性类型: 结果}} 的嵌套字典，方便画对比图。"""
        if scenario == "base_performance":
            base_performance_list = self.generate_for_comparison(scenario)
            results = {}
            for config_dict in base_performance_list:
                config = Config(**config_dict)
                data_dict = config.load_result()
                traj_name = config_dict["traj_name"]
                method = config_dict["method"]
                # 初始化数据结构
                if traj_name not in results:
                    results[traj_name] = {"ref_traj": data_dict["ref_traj"], "u": data_dict["u"]}
                # 存储方法特定的预测轨迹
                results[traj_name][method] = {
                    "pre_traj": data_dict["traj"],
                    "u": data_dict["u"],
                    "error_dis": data_dict["error_dis"],
                    "error_angle": data_dict["error_angle"],
                }

        if scenario == "noise_robustness":
            robustness_performance_list = self.generate_for_comparison(scenario)
            results = {}
            for config_dict in robustness_performance_list:
                config = Config(**config_dict)
                data_dict = config.load_result()
                traj_name = config_dict["traj_name"]
                method = config_dict["method"]
                # 确定鲁棒性方法类型
                robustness_type = "none"
                if config_dict["use_KEM"]:
                    robustness_type = "KEM"
                # 初始化数据结构
                if traj_name not in results:
                    results[traj_name] = {"ref_traj": data_dict["ref_traj"], "u": data_dict["u"]}

                if method not in results[traj_name]:
                    results[traj_name][method] = {}

                # 存储结果
                results[traj_name][method][robustness_type] = {
                    "pre_traj": data_dict["traj"],
                    "u": data_dict["u"],
                    "error_dis": data_dict["error_dis"],
                    "error_angle": data_dict["error_angle"],
                }

        return results

    def save_table(self, results_dict, save_Fig_path):
        """把 base_performance / noise_robustness 两组结果的误差指标整理成一张 matplotlib 表格图并保存。"""
        # 添加误差表格
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        error_table_data = []

        results = results_dict["noise_robustness"]
        for traj_name in ["Fig8", "FigStar"]:
            for method in self.method_config:
                if method not in results[traj_name]:
                    continue
                row_data = [f"{self.method_config[method]['name']} - {traj_name}"]

                error_dis = results_dict["base_performance"][traj_name][method]["error_dis"]
                error_angle = results_dict["base_performance"][traj_name][method]["error_angle"]
                # 用位置误差和角度误差的平方和开根号得到一个综合误差指标，
                # 方便在同一张表格里给出单一的"整体表现"数字。
                error_all = np.sqrt(error_dis**2 + error_angle**2)
                row_data.append(f"{error_dis:.4f},{error_angle:.4f},{error_all:.4f}")
                for rob_type in ["none", "KEM"]:
                    if rob_type in results[traj_name][method]:
                        error_dis = results[traj_name][method][rob_type]["error_dis"]
                        error_angle = results[traj_name][method][rob_type]["error_angle"]
                        error_all = np.sqrt(error_dis**2 + error_angle**2)
                        row_data.append(f"{error_dis:.4f},{error_angle:.4f},{error_all:.4f}")
                    else:
                        row_data.append("N/A")
                error_table_data.append(row_data)

        # 创建表格
        table = ax.table(
            cellText=error_table_data,
            colLabels=["Scenario", "base", "No Filter", "KF Only"],
            loc="center",
            # bbox=[0.0, -0.3, 1.0, 0.2],
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        # --- 给表格留空间 ---
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(os.path.join(save_Fig_path, "table.png"), format="png", dpi=500, bbox_inches="tight")
        plt.show()


class ConfigGenerator_DBKN:
    """
    DBKN + KESO 实验（噪声/负载鲁棒性）的配置矩阵生成器与结果汇总/画表工具。
    比 ConfigGenerator 多了 use_eso(扩张状态观测器) 和 use_payload(负载等级)
    两个维度，对应 README 里"抗噪声"/"抗负载"/"抗噪+抗负载"三组实验图。
    """

    def __init__(self, base_config):
        self.base = base_config
        # 轨迹配置信息
        self.traj_config = {
            "Fig8": {
                "title": "Fig8",
            },
            "FigStar": {
                "title": "FigStar",
            },
        }
        # 方法配置信息
        self.method_config = {
            "DBKN": {"name": "DBKN", "color": "red", "line_style": "--"},
        }
        # 四种鲁棒性配置的画图样式：none(无任何补偿) / KEM(仅卡尔曼滤波) /
        # ESO(仅扩张状态观测器) / both(两者都开，即 KESO)。
        self.robustness_config = {
            "none": {"name": "DBKMPC", "alpha": 1.0, "linewidth": 1.5, "color": "purple", "line_style": ":"},
            "KEM": {"name": "DBKMPC-KF", "alpha": 1.0, "linewidth": 1.5, "color": "green", "line_style": "-."},
            "ESO": {"name": "DBKMPC-ESO", "alpha": 1.0, "linewidth": 1.8, "color": "blue", "line_style": "--"},
            "both": {"name": "DBKMPC-KESO", "alpha": 1.0, "linewidth": 2.0, "color": "red", "line_style": "-"},
        }

    # 生成所有有效配置组合
    def generate_configs(self, space):
        """对参数空间做笛卡尔积展开，逻辑同 ConfigGenerator.generate_configs（这里没有额外的互斥规则）。"""
        keys = list(space.keys())
        value_combinations = itertools.product(*(space[key] for key in keys))

        configs = []
        seen = set()
        for combo in value_combinations:
            config = self.base.copy()
            for i, key in enumerate(keys):
                config[key] = combo[i]
            # if not config["use_nosie"]:
            #     config["use_KEM"] = False
            # 创建唯一标识符
            config_id = tuple(sorted(config.items()))
            # 检查是否重复
            if config_id not in seen:
                seen.add(config_id)
                configs.append(config)

        return configs

    def generate_for_comparison(self, scenario):
        """生成特定场景的配置"""
        scenarios = {
            "base_performance": {
                "method": ["DBKN"],
                "traj_name": ["Fig8", "FigStar"],
                "use_nosie": [False],
                "use_KEM": [False],
                "use_eso": [False],
                "use_payload": [0],
            },
            "noise_robustness": {
                "method": ["DBKN"],
                "traj_name": ["Fig8", "FigStar"],
                "use_nosie": [True],
                "use_KEM": [True, False],
                "use_eso": [True, False],
                "use_payload": [0],
            },
            "payload_robustness": {
                "method": ["DBKN"],
                "traj_name": ["Fig8", "FigStar"],
                "use_nosie": [False],
                "use_KEM": [True, False],
                "use_eso": [True, False],
                "use_payload": [2],
            },
            "both_robustness": {
                "method": ["DBKN"],
                "traj_name": ["Fig8", "FigStar"],
                "use_nosie": [True],
                "use_KEM": [True, False],
                "use_eso": [True, False],
                "use_payload": [2],
            },
        }

        if scenario not in scenarios:
            raise ValueError(f"未知场景: {scenario}. 可用场景: {list(scenarios.keys())}")

        return self.generate_configs(scenarios[scenario])

    def get_result(self, scenario):
        """按场景批量 load_result()，按 use_KEM/use_eso 组合归类成 none/KEM/ESO/both 四档鲁棒性类型。"""
        if scenario == "base_performance":
            base_performance_list = self.generate_for_comparison(scenario)
            results = {}
            for config_dict in base_performance_list:
                config = Config(**config_dict)
                data_dict = config.load_result()
                traj_name = config_dict["traj_name"]
                method = config_dict["method"]
                # 初始化数据结构
                if traj_name not in results:
                    results[traj_name] = {"ref_traj": data_dict["ref_traj"]}
                # 存储方法特定的预测轨迹
                results[traj_name][method] = {
                    "pre_traj": data_dict["traj"],
                    "error_dis": data_dict["error_dis"],
                    "error_angle": data_dict["error_angle"],
                }

        else:
            robustness_performance_list = self.generate_for_comparison(scenario)
            results = {}
            for config_dict in robustness_performance_list:
                config = Config(**config_dict)
                data_dict = config.load_result()
                traj_name = config_dict["traj_name"]
                method = config_dict["method"]
                # 确定鲁棒性方法类型
                robustness_type = "none"
                # payload 相关场景下，"既不开 KEM 也不开 ESO"这一档没有意义
                # (没有任何补偿手段却要承受负载扰动，通常直接失控/误差过大，
                # 不具备对比价值)，直接跳过不汇总。
                if (
                    (scenario == "payload_robustness" or scenario == "both_robustness")
                    and not config_dict["use_KEM"]
                    and not config_dict["use_eso"]
                ):
                    continue
                if config_dict["use_KEM"] and config_dict["use_eso"]:
                    robustness_type = "both"
                elif config_dict["use_KEM"]:
                    robustness_type = "KEM"
                elif config_dict["use_eso"]:
                    robustness_type = "ESO"
                # 初始化数据结构
                if traj_name not in results:
                    results[traj_name] = {"ref_traj": data_dict["ref_traj"]}

                if method not in results[traj_name]:
                    results[traj_name][method] = {}

                # 存储结果
                results[traj_name][method][robustness_type] = {
                    "pre_traj": data_dict["traj"],
                    "error_dis": data_dict["error_dis"],
                    "error_angle": data_dict["error_angle"],
                }

        return results

    def save_table(self, results_dict, save_Fig_path):
        """把 noise/payload/both 三组鲁棒性场景的误差指标汇总成一张大表格图并保存。"""
        # 添加误差表格
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis("off")
        error_table_data = []

        for scenario in ["noise_robustness", "payload_robustness", "both_robustness"]:
            results = results_dict[scenario]
            for traj_name in ["Fig8", "FigStar"]:
                for method in self.method_config:
                    if method not in results[traj_name]:
                        continue
                    row_data = [f"{self.method_config[method]['name']}-{traj_name}-{scenario.split('_')[0]}"]

                    if scenario == "noise_robustness":
                        error_dis = results_dict["base_performance"][traj_name][method]["error_dis"]
                        error_angle = results_dict["base_performance"][traj_name][method]["error_angle"]
                        error_all = np.sqrt(error_dis**2 + error_angle**2)
                        row_data.append(f"{error_dis:.4f},{error_angle:.4f},{error_all:.4f}")
                    else:
                        # payload/both 场景没有单独的 base_performance 基线
                        # (baseline 本身就是 use_payload=0 的场景，和当前负载
                        # 等级不可直接比较)，这一列直接标 N/A。
                        row_data.append("N/A")
                    for rob_type in ["none", "KEM", "ESO", "both"]:
                        if rob_type in results[traj_name][method]:
                            error_dis = results[traj_name][method][rob_type]["error_dis"]
                            error_angle = results[traj_name][method][rob_type]["error_angle"]
                            error_all = np.sqrt(error_dis**2 + error_angle**2)
                            row_data.append(f"{error_dis:.4f},{error_angle:.4f},{error_all:.4f}")
                        else:
                            row_data.append("N/A")
                    error_table_data.append(row_data)

        # 创建表格
        table = ax.table(
            cellText=error_table_data,
            colLabels=["Scenario", "base", "No Filter", "KF Only", "ESO Only", "KF+ESO"],
            loc="center",
            # bbox=[0.0, -0.3, 1.0, 0.2],
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        # --- 给表格留空间 ---
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(
            os.path.join(save_Fig_path, f"table_{self.base['method']}.png"), format="png", dpi=500, bbox_inches="tight"
        )
        plt.show()
