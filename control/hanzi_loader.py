"""
把 hanzi-writer-data (makemeahanzi 笔画数据) 转换成 generate_multi_stroke() 需要的
strokes 格式: [[(x0,y0), (x1,y1), ...], [...], ...]，单位米，相对字符中心的偏移量。

数据来源: https://github.com/chanind/hanzi-writer-data
每个汉字一个 JSON 文件 (文件名就是这个字)，"medians" 字段是每一笔画的中线折线点，
坐标系是 makemeahanzi 的标准 1024 单位方格，Y 轴向上为正 (不是 SVG 常见的 Y 向下)。
"""

import json
import os

# 默认指向本机已经 clone 好的 hanzi-writer-data 仓库；换机器/换路径时改这里，
# 或者调用时传 data_dir 参数覆盖。
DEFAULT_HANZI_DATA_DIR = "/home/cym/ROS/Arms/isaac_so_arm101/hanzi-writer-data/data"

# makemeahanzi 标准坐标系: 字符大致画在 x:[0,1024] y:[-124,900] 的方格里，
# 用这个固定的参考框（而不是每个字自己的包围盒）做归一化，
# 这样不同汉字之间的相对大小关系和真实书法一致（笔画少的字看起来就应该更"疏朗"）。
_VIEWBOX_CENTER = (512.0, 388.0)
_VIEWBOX_SIZE = 1024.0


def load_hanzi_medians(char: str, data_dir: str = DEFAULT_HANZI_DATA_DIR) -> list:
    """加载单个汉字的笔画中线 (medians) 原始数据，未归一化。"""
    path = os.path.join(data_dir, f"{char}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到汉字 '{char}' 的笔画数据: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["medians"]


def hanzi_to_strokes(char: str, size_m: float = 0.12, data_dir: str = DEFAULT_HANZI_DATA_DIR) -> list:
    """
    把一个汉字转换成 strokes 列表 (单位: 米, 相对字符中心的偏移量)。

    Args:
        char: 单个汉字，比如 "十"。
        size_m: 字符大致占用的物理尺寸（对应 1024 单位的方格边长）。
        data_dir: hanzi-writer-data 的 data/ 目录路径。
    """
    medians = load_hanzi_medians(char, data_dir)
    scale = size_m / _VIEWBOX_SIZE
    cx, cy = _VIEWBOX_CENTER
    strokes = []
    for stroke in medians:
        pts = [((x - cx) * scale, (y - cy) * scale) for x, y in stroke]
        strokes.append(pts)
    return strokes


def text_to_strokes(
    text: str, size_m: float = 0.12, spacing_m: float | None = None, data_dir: str = DEFAULT_HANZI_DATA_DIR
) -> list:
    """
    把一串汉字横向排开 (沿 X 方向)，拼成一整套 strokes，自动加上字符间的偏移。
    多字符时手臂需要覆盖的范围会成倍增大，建议先用单字验证可行再逐步加长。

    Args:
        text: 要书写的一串汉字，比如 "十一"。
        size_m: 每个字的物理尺寸。
        spacing_m: 字符中心间距，默认 1.2 * size_m。
        data_dir: hanzi-writer-data 的 data/ 目录路径。
    """
    if spacing_m is None:
        spacing_m = size_m * 1.2
    all_strokes = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        offset_x = i * spacing_m
        for stroke in hanzi_to_strokes(ch, size_m=size_m, data_dir=data_dir):
            all_strokes.append([(x + offset_x, y) for x, y in stroke])
    return all_strokes
