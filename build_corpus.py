# -*- coding: utf-8 -*-
r"""
build_corpus.py  ·  产品资料 → AI 知识库语料生成器

【它干什么】
读一张产品信息表（Excel）→ 清洗整理 → 自动生成一份结构化 Markdown 语料库。
输出可以直接当作 AI 知识库（RAG）的语料使用。

【怎么用】
1. 装依赖：    pip install -r requirements.txt
2. 造示例数据：python make_sample_data.py
3. 生成语料库：python build_corpus.py

【输入】data/sample_products.xlsx     （可以换成你自己的表，列名对上就行）
【输出】output/product_corpus.md

【面试要点：我为什么这么设计】
  · 为什么用 header=1  →  源表第 1 行是标题行，真正的列名在第 2 行
  · 为什么空值要填「无」→  留空会变成 "nan" 混进知识库，检索到就污染答案
  · 为什么输出 Markdown →  # / ## 标题层级天然对应知识库的切块边界
  · 为什么卖点要提取核心词 → 客服问的是"功能"不是"型号"，
                            同一功能有多种说法，归组后才能被检索到
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")   # 让中文能正常打印

import re
from collections import defaultdict

import pandas as pd

# ============================================================
# 0. 路径：用本文件所在位置做基准，从任何地方运行都能找到文件
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))      # __file__ = 本脚本自己的路径
EXCEL_PATH = os.path.join(BASE, "data", "sample_products.xlsx")
OUTPUT_PATH = os.path.join(BASE, "output", "product_corpus.md")
OUTPUT_DIR = os.path.join(BASE, "output")

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)          # 输出目录不存在就建一个


# ============================================================
# 1. 读取 Excel —— header=1 让第 2 行作为列名
# ============================================================
df = pd.read_excel(EXCEL_PATH, header=1, engine="openpyxl")

print("读取到的数据形状：" + str(df.shape[0]) + " 行 × " + str(df.shape[1]) + " 列")

# 去掉全空的行
df = df.dropna(how="all").reset_index(drop=True)
print("有效产品数：" + str(len(df)))

# 列名映射：短名字 → Excel 里的真实列名
# （真实列名带单位后缀，比如「主机重量（单位G）」，用短名字引用更省事、也不容易写错）
COL = {
    "产品图片": "产品图片",
    "产品名称": "产品名称",
    "蓝牙版本": "蓝牙版本",
    "是否串联": "是否串联",
    "AI助手": "AI助手",
    "语音通话": "语音通话",
    "查看电量": "查看电量",
    "翻译": "翻译",
    "收音机功能": "收音机功能",
    "瓦数": "瓦数",
    "无线传输范围": "无线传输范围（M）",
    "电池容量": "电池容量（MAH）",
    "主机重量": "主机重量（单位G）",
    "使用时长": "使用时长（小时）",
    "充电时长": "充电时长（小时）",
    "喇叭数量": "喇叭数量（个）",
    "配件列表": "配件列表",
    "产品尺寸": "产品尺寸（长*宽*高）",
    "产品配色": "产品配色",
    "播放模式": "播放模式",
    "卖点": "卖点",
}


# ============================================================
# 2. 辅助函数
# ============================================================

def safe_val(row, col_key, default="无"):
    """从一行里取出某列的值；如果是空值，就返回默认值「无」

    为什么要有它？
      真实数据里一定有空格子。直接拿来拼字符串会拼出 "nan"，
      进了知识库就会变成一条垃圾信息。
    """
    col_name = COL[col_key]
    val = row.get(col_name)

    if pd.isna(val):            # pd.isna 判断"是不是空值"
        return default

    s = str(val).strip()        # str() 转成文字，strip() 去掉两边空格
    if s in ["", "nan", "None", "Nan", "NaN"]:
        return default
    return s


def num_val(row, col_key, default="无"):
    """取"数值"列：先按 safe_val 兜底，再抹掉多余的 .0

    为什么要抹掉 .0？
      只要这一列里有空值，pandas 就会把整列当成小数，
      原本的 1200 会被读成 1200.0，输出里就变成「电池容量：1200.0」。
      这里判断：结尾是 .0 就把最后两个字符切掉（s[:-2] 是切片）。
    """
    s = safe_val(row, col_key, default)
    if s.endswith(".0"):
        return s[:-2]
    return s


def split_selling_points(text):
    """按分隔符把卖点拆成条目 —— 但**括号里的逗号不算分隔符**

    为什么不能直接写 re.split(r'[，,;；]', text)？
      因为「蓝牙5.3高速连接（低延迟，游戏可用）」里那个逗号在括号内，
      直接按逗号切会把它切成两半：
        「蓝牙5.3高速连接（低延迟」 和 「游戏可用）」
      这种半截条目进了知识库就是垃圾。

    这里的做法：
      一个字一个字地走一遍，用 depth 记住「现在在不在括号里」。
      只有 depth == 0（在括号外面）时，逗号才当作分隔符。
    """
    items = []
    buf = ""        # buf：正在攒的这一条
    depth = 0       # depth：括号深度。0 = 在括号外，1 = 在括号里

    for ch in text:
        if ch in "（(":
            depth = depth + 1            # 遇到左括号 → 进括号
        elif ch in "）)":
            if depth > 0:
                depth = depth - 1        # 遇到右括号 → 出括号

        if ch in "，,;；" and depth == 0:
            items.append(buf.strip())     # 括号外的分隔符 → 在这里切一刀
            buf = ""
        else:
            buf = buf + ch

    items.append(buf.strip())             # 最后一条别忘了收尾

    # 去掉切出来的空条目
    result = []
    for it in items:
        if it != "":
            result.append(it)
    return result


def extract_main_selling_point(item):
    """提取卖点的核心关键词（括号前的那部分）

    例：「续航12小时（支持快充）」 → 「续航12小时」
    这样「同一功能的不同说法」才能归到一组。
    """
    main = re.split(r"[（(]", item)[0].strip()
    return main


# ============================================================
# 3. 逐产品生成文本 + 同时收集卖点汇总
# ============================================================

selling_point_summary = defaultdict(set)   # {核心词: {完整表述1, 完整表述2, ...}}
all_products_text = []

for idx, row in df.iterrows():
    name = safe_val(row, "产品名称")
    if name == "无":
        continue                            # 没名字的行跳过

    lines = []
    lines.append("# 产品：" + name)
    lines.append("")

    # -------- 基本参数 --------
    params = []

    bt = safe_val(row, "蓝牙版本")
    if bt != "无":
        params.append("蓝牙版本：" + bt)

    params.append("功率：" + num_val(row, "瓦数") + "W")

    trans_range = num_val(row, "无线传输范围")
    if trans_range != "无":
        params.append("无线传输范围：" + trans_range + "M")

    params.append("电池容量：" + num_val(row, "电池容量"))
    params.append("主机重量：" + num_val(row, "主机重量") + "g")
    params.append("使用时长：" + num_val(row, "使用时长") + "小时")
    params.append("充电时长：" + num_val(row, "充电时长") + "小时")
    params.append("喇叭数量：" + num_val(row, "喇叭数量") + "个")

    lines.append("## 基本参数")
    lines.append(" | ".join(params))
    lines.append("")

    # -------- 功能特性 --------
    features = [
        "串联功能：" + safe_val(row, "是否串联"),
        "AI助手：" + safe_val(row, "AI助手"),
        "语音通话：" + safe_val(row, "语音通话"),
        "查看电量：" + safe_val(row, "查看电量"),
        "翻译功能：" + safe_val(row, "翻译"),
        "收音机：" + safe_val(row, "收音机功能"),
    ]
    lines.append("## 功能特性")
    for f in features:
        lines.append("- " + f)
    lines.append("")

    # -------- 尺寸 / 配色 / 播放模式 / 配件 --------
    lines.append("## 产品尺寸\n" + safe_val(row, "产品尺寸"))
    lines.append("")
    lines.append("## 产品配色\n" + safe_val(row, "产品配色"))
    lines.append("")
    lines.append("## 播放模式\n" + safe_val(row, "播放模式"))
    lines.append("")
    lines.append("## 配件列表\n" + safe_val(row, "配件列表"))
    lines.append("")

    # -------- 卖点：逐条列出，同时收集进汇总 --------
    selling_raw = safe_val(row, "卖点")
    if selling_raw != "无":
        selling_items = split_selling_points(selling_raw)
        lines.append("## 卖点")
        for item in selling_items:
            lines.append("- " + item)
            main_key = extract_main_selling_point(item)
            selling_point_summary[main_key].add(item)
        lines.append("")
    else:
        lines.append("## 卖点\n（暂无标注）")
        lines.append("")

    lines.append("---\n")
    all_products_text.append("\n".join(lines))


# ============================================================
# 4. 卖点汇总表：按核心词分组，自动去重
# ============================================================

summary_lines = ["# 卖点汇总（所有产品）\n"]
summary_lines.append("> 把全部产品的卖点按核心功能分组；同一功能的不同说法会列在一起。\n")

for main_key in sorted(selling_point_summary.keys()):
    full_items = sorted(selling_point_summary[main_key])
    if len(full_items) == 1:
        summary_lines.append("- **" + main_key + "**：" + full_items[0])
    else:
        summary_lines.append("- **" + main_key + "**（共" + str(len(full_items)) + "种表述）：")
        for item in full_items:
            summary_lines.append("  - " + item)

summary_text = "\n".join(summary_lines)


# ============================================================
# 5. 写入文件
# ============================================================

header = (
    "# 产品信息语料库\n\n"
    "> 本文件由 Excel 产品数据自动生成，共包含 " + str(len(df)) + " 个产品。\n"
    "> 空值自动填充为「无」，卖点已按条拆分，文末附有全产品卖点汇总。\n\n"
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(header)
    f.write("\n".join(all_products_text))
    f.write("\n\n")
    f.write(summary_text)

print("")
print("语料库已生成：" + OUTPUT_PATH)
print("  产品数：" + str(len(df)))
print("  卖点类别数：" + str(len(selling_point_summary)))
