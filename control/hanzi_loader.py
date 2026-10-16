"""
把 hanzi-writer-data (makemeahanzi 笔画数据) 转换成 generate_multi_stroke() 需要的
strokes 格式: [[(x0,y0), (x1,y1), ...], [...], ...]，单位米，相对字符中心的偏移量。

数据来源: https://github.com/chanind/hanzi-writer-data
每个汉字一个 JSON 文件 (文件名就是这个字)，"medians" 字段是每一笔画的中线折线点，
坐标系是 makemeahanzi 的标准 1024 单位方格，Y 轴向上为正 (不是 SVG 常见的 Y 向下)。

【在写字流程中的位置】
    write_demo.py: hanzi_to_strokes("二", size_m=0.12)
        -> 本文件: 汉字 -> 笔画中线折线 (米制, 以字中心为原点)
        -> TrajectoryGenerator.generate_multi_stroke(strokes, center=(0.35, 0.0))
           (把笔画平移到桌面上的 center 处, 补上 z 高度和抬笔/落笔动作, 再做逆运动学)

【数据格式】
    hanzi-writer-data 里每个字一个 JSON, 本文件只用其中的 "medians" 字段:
        medians = [ 第1笔的中线, 第2笔的中线, ... ]       # 按书写顺序排列
        每一笔的中线 = [[x0, y0], [x1, y1], ...]            # 一般只有几个到十几个关键点, 是粗折线
    返回给调用方的 strokes 形状完全相同, 只是坐标被换算成"米"并且移到了字中心:
        strokes = [ [(x, y), ...], [(x, y), ...], ... ]

【坐标约定 (读代码时容易晕的地方)】
    - 字形坐标: x 向右, y 向上 (makemeahanzi 约定, 与 SVG 的 y 向下相反)。
    - 本文件不做任何翻转/旋转: 字的"横向"原样对应世界系 X, 字的"竖向"原样对应世界系 Y
      (见 generate_multi_stroke: world_x = center_x + p[0], world_y = center_y + p[1])。
      所以写出来的字朝向, 取决于你从哪个方向看桌面。
    - 多字排版 (text_to_strokes) 沿 +X 方向展开, 即沿机械臂"前后"方向(取决于底座摆放), 很容易超出可达范围。
"""

# 标准库: json 解析笔画数据文件; os 拼接路径/检查文件是否存在。
# 本文件不依赖 numpy 等第三方库, 返回的都是纯 Python 的 list/tuple。
import json
import os

# 默认指向本机已经 clone 好的 hanzi-writer-data 仓库；换机器/换路径时改这里，
# 或者调用时传 data_dir 参数覆盖。
DEFAULT_HANZI_DATA_DIR = "/home/cym/ROS/Arms/isaac_so_arm101/hanzi-writer-data/data"

# makemeahanzi 标准坐标系: 字符大致画在 x:[0,1024] y:[-124,900] 的方格里，
# 用这个固定的参考框（而不是每个字自己的包围盒）做归一化，
# 这样不同汉字之间的相对大小关系和真实书法一致（笔画少的字看起来就应该更"疏朗"）。
# 参考框中心 (512, 388):
#     x 方向: 范围 [0, 1024]        -> 中点 = 512
#     y 方向: 范围 [-124, 900]      -> 中点 = (-124 + 900) / 2 = 388
# 把这个中点平移到原点, 字的"整体居中"就成立了。
# 注意是整个参考框的中心, 而不是单个字的包围盒中心:
#     "一"这种扁字, 它的笔画就会落在原点附近偏下/偏上的位置, 与书写时的真实字位一致。
_VIEWBOX_CENTER = (512.0, 388.0)
# 参考框边长 (1024 个字形单位)。物理缩放比例 scale = size_m / 1024:
#     size_m=0.12 时, 1 个字形单位 ≈ 0.117 mm, 整个字占 12cm x 12cm。
_VIEWBOX_SIZE = 1024.0


def load_hanzi_medians(char: str, data_dir: str = DEFAULT_HANZI_DATA_DIR) -> list:
    """加载单个汉字的笔画中线 (medians) 原始数据，未归一化。"""
    # 文件名就是汉字本身, 例如 data_dir/二.json。
    # (只支持单个字符; 传入多个字符会找不到文件, 多字请用 text_to_strokes。)
    path = os.path.join(data_dir, f"{char}.json")
    # 找不到文件时给出明确报错: 常见原因是 data_dir 路径写错(默认路径是作者本机的), 或该字不在 9574 字的数据集里。
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到汉字 '{char}' 的笔画数据: {path}")
    with open(path, encoding="utf-8") as f:
        # data 是一个 dict, 常见字段有 strokes(笔画轮廓)、medians(笔画中线)、radStrokes 等; 这里只用 medians。
        data = json.load(f)
    # 返回原始中线数据, 未做任何缩放/平移:
    #     [[[x, y], [x, y], ...],     <- 第1笔
    #      [[x, y], ...],             <- 第2笔
    #      ...]
    # 列表顺序就是笔顺, 后面生成轨迹时会按这个顺序依次书写。
    return data["medians"]


def hanzi_to_strokes(char: str, size_m: float = 0.12, data_dir: str = DEFAULT_HANZI_DATA_DIR) -> list:
    """
    把一个汉字转换成 strokes 列表 (单位: 米, 相对字符中心的偏移量)。

    Args:
        char: 单个汉字，比如 "十"。
        size_m: 字符大致占用的物理尺寸（对应 1024 单位的方格边长）。
        data_dir: hanzi-writer-data 的 data/ 目录路径。
    """
    # 第1步: 读原始笔画中线 (1024 单位的字形坐标)。
    medians = load_hanzi_medians(char, data_dir)
    # 第2步: 计算 字形单位 -> 米 的缩放比例。
    scale = size_m / _VIEWBOX_SIZE
    # 取参考框中心, 作为平移用的原点 (这里的 cx, cy 是字形坐标, 与 TrajectoryGenerator 里桌面上的 center 不是一回事)。
    cx, cy = _VIEWBOX_CENTER
    # strokes[i] 对应第 i 笔, 里面是该笔换算成米之后的 (x, y) 点列。
    strokes = []
    # 逐笔处理: 每一笔都是一串 [x, y] 字形坐标。
    for stroke in medians:
        # 核心换算, 先平移再缩放:
        #     x_m = (x - 512) * scale        y_m = (y - 388) * scale
        # 例 (size_m=0.12, scale≈1.17e-4):
        #     字形点 (512, 388) -> (0, 0)          字的中心
        #     字形点 (0, 388)   -> (-0.06, 0)      字的最左端
        #     字形点 (1024, 388)-> (+0.06, 0)      字的最右端
        # 只做了平移+等比缩放, 没有翻转 y 轴, 也没有旋转。
        pts = [((x - cx) * scale, (y - cy) * scale) for x, y in stroke]
        # 保持笔顺; 每笔之间的抬笔/落笔动作不在这里处理, 由 generate_multi_stroke 负责。
        strokes.append(pts)
    # 返回值可以直接喂给 CartesianTrajectoryGenerator.generate_multi_stroke(strokes, center=(cx, cy)):
    #     那里会把每个点加上 center 得到桌面世界系坐标, 再补充 z 高度。
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
    # 字间距默认取 1.2 倍字宽, 即相邻字的外框之间留 20% 的空隙。
    if spacing_m is None:
        spacing_m = size_m * 1.2
    # 把所有字的所有笔画按顺序放进同一个列表: 先写完第1个字的全部笔画, 再写第2个字, 依此类推。
    all_strokes = []
    # i 是字符在 text 中的下标(包括空白字符), 排版位置由它决定。
    for i, ch in enumerate(text):
        # 空白字符不产生笔画。但因为 i 仍然在计数, 空格会在行里留出一个字的空位:
        #     "你 好" 里"好"的 i=2, 与"你"之间隔了一个字宽 (而不是紧挨着)。
        if ch.isspace():
            continue
        # 第 i 个字沿 +X 平移 i * spacing_m:
        #     第 0 个字的中心就在原点(也就是调用时 center 所在位置), 后面的字往 +X 方向依次排开。
        #     注意整串文字并没有再整体居中; 如果想让整行字以 center 为中心, 需要调用方自己再减去 (n-1)*spacing/2。
        offset_x = i * spacing_m
        # 复用单字转换, 得到该字在自己中心坐标系下的笔画。
        for stroke in hanzi_to_strokes(ch, size_m=size_m, data_dir=data_dir):
            # 只改 x(加上字的偏移量), y 不变, 所以多字是横向一字排开。
            all_strokes.append([(x + offset_x, y) for x, y in stroke])
    # 多字时轨迹点数和所需关节范围都随字数线性/成倍增长; 建议先用单字验证可达性 (write_demo.py 的 check_reachability)。
    return all_strokes