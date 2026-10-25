"""统一配色：取自参考色卡（冷色为主的临床图表色系）。

约定与用户既有偏好一致：**填充用浅色，描边用中等深度，文字才用深色**。

色卡原始值（自上而下）：
    #E9F5F1  #E2F4F3  #DAF0F4  #D3E8F1  #A5CAB5  #B5CADC  #4D77A7  #D4D4D4
    #FEF1E8  #FDE4E4  #F7DFE5  #F5CED0  #DCE6EE  #DEE3EF  #CED6F6  #8D9ECF  #9DDFE4

其中只有 #4D77A7 是足够深、可直接用作线条的。其余作描边时按同色相压暗，
并在下方以 “derived from” 注明来源，保证四组序列既同源又可区分。
"""
from __future__ import annotations

# 文字与网格
INK = "#1E2A32"          # 正文字色（比纯黑柔和，配冷色系）
MUTED = "#6B7C8A"        # 次级文字 / 参考线
GRID = "#E3E8ED"         # 网格线

STYLE = {
    # 主序列：色卡里的浅蓝 + 唯一的深蓝
    "blue": {"fill": "#D3E8F1", "edge": "#4D77A7"},
    # 对照序列：淡紫蓝，描边 derived from #8D9ECF
    "slate": {"fill": "#CED6F6", "edge": "#5F6FA8"},
    # 第三序列：浅青，描边 derived from #9DDFE4
    "aqua": {"fill": "#DAF0F4", "edge": "#3D8E97"},
    # 暖色强调（原 orange 位）：浅粉，描边 derived from #F5CED0
    "coral": {"fill": "#FDE4E4", "edge": "#C97F86"},
    # 第五序列：极浅薄荷，描边 derived from #A5CAB5
    "sage": {"fill": "#E9F5F1", "edge": "#6D9A83"},
    # 中性：浅灰蓝，描边 derived from #B5CADC
    "steel": {"fill": "#DCE6EE", "edge": "#7794AC"},
}

# 兼容旧脚本里的语义键，避免逐个改写绘图代码
STYLE["orange"] = STYLE["coral"]
STYLE["green"] = STYLE["aqua"]
STYLE["purple"] = STYLE["slate"]

# 误差预算图的三个分带
FILL_ORACLE = "#D4D4D4"
EDGE_ORACLE = "#8A8A8A"
FILL_FLOOR = STYLE["coral"]["fill"]
EDGE_FLOOR = STYLE["coral"]["edge"]
FILL_RANK = STYLE["blue"]["fill"]
EDGE_RANK = STYLE["blue"]["edge"]

# 流程图的三个纵向区块
PIPE_FILL = STYLE["blue"]["fill"]
PIPE_EDGE = STYLE["blue"]["edge"]
AUDIT_FILL = "#FEF1E8"
AUDIT_EDGE = "#C0906A"
OUT_FILL = STYLE["sage"]["fill"]
OUT_EDGE = STYLE["sage"]["edge"]

# 顺序调色（用于多类别柱状/散点，例如架构对比）
CYCLE = [
    STYLE["blue"],
    STYLE["coral"],
    STYLE["aqua"],
    STYLE["slate"],
    STYLE["sage"],
    STYLE["steel"],
]
