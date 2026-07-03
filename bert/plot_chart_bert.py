"""
BERT v2 情感分析结果可视化
读取 BERT v2 评分结果，生成带中文的图表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import platform
from datetime import datetime

# ── 中文显示 ──
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── 配置 ──
INPUT_CSV = 'D:/xueqiu/data/xueqiu_300750_bert_v2_scored.csv'
OUTPUT_CHART = 'D:/xueqiu/bert/sentiment_bert_v2.png'

# ── 读取数据 ──
df = pd.read_csv(INPUT_CSV)
total = len(df)

# BERT 情绪分布
dist = df['bert_label'].value_counts()

# 将 bert_label 映射为数值用于计算平均
label_to_score = {'负面': -1, '中性': 0, '正面': 1}
df['bert_score'] = df['bert_label'].map(label_to_score)
avg_sentiment = df['bert_score'].mean()

# 加权平均（用 bert_prob 作为权重，置信度越高权重越大）
weighted_avg = np.average(df['bert_score'], weights=df['bert_prob'])
weighted_avg_label = '正面' if weighted_avg > 0.1 else ('负面' if weighted_avg < -0.1 else '中性')

def parse_publish_time(time_str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(str(time_str).strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None

# ── 绘图 ──
print(f"[绘图] 生成图表 ...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
fig.suptitle("宁德时代 (300750.SZ) 雪球帖子情绪分析（BERT v2）", fontsize=16, fontweight="bold")

# ──── 图1：情绪分布柱状图 ────
labels = ["负面\n(-1)", "中性\n(0)", "正面\n(+1)"]
colors_dist = ["#d73027", "#f0f0f0", "#4575b4"]

counts = [dist.get(l, 0) for l in ['负面', '中性', '正面']]
bars = ax1.bar(labels, counts, color=colors_dist, edgecolor="grey", linewidth=0.8)

for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
             str(count), ha="center", va="bottom", fontsize=11)

ax1.set_title("BERT 情绪分布", fontsize=13, pad=10)
ax1.set_ylabel("帖子数量", fontsize=11)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 10)

for bar, count in zip(bars, counts):
    pct = count / total * 100
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
             f"{pct:.1f}%", ha="center", va="center", fontsize=10,
             color="black" if count < max(counts) * 0.5 else "white",
             fontweight="bold")

# ──── 图2：情绪时间序列 ────
df["parsed_time"] = df["publish_time"].apply(parse_publish_time)
ts_df = df.dropna(subset=["parsed_time"]).sort_values("parsed_time")

if len(ts_df) > 0:
    ts_df["date"] = ts_df["parsed_time"].dt.date
    daily = ts_df.groupby("date").agg(
        avg_score=("bert_score", "mean"),
        count=("bert_score", "count")
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    daily["weighted_score"] = ts_df.groupby("date").apply(
        lambda g: np.average(g["bert_score"], weights=g["bert_prob"])
    ).values

    ax2.plot(daily["date"], daily["avg_score"], color="#fc8d59", linewidth=1.5,
             marker="o", markersize=3, label="简单平均情绪", alpha=0.8)
    ax2.plot(daily["date"], daily["weighted_score"], color="#4575b4", linewidth=1.5,
             marker="s", markersize=3, label="加权平均情绪（按置信度）", alpha=0.8)

    ax2.axhline(y=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(y=avg_sentiment, color="#fc8d59", linestyle=":", linewidth=0.8,
                alpha=0.6, label=f"简单均值 ({avg_sentiment:+.3f})")

    ax2.set_title("BERT 情绪时间序列（按日）", fontsize=13, pad=10)
    ax2.set_xlabel("日期", fontsize=11)
    ax2.set_ylabel("平均情绪分数 (-1 ~ +1)", fontsize=11)
    ax2.legend(loc="best", frameon=True, facecolor="white", edgecolor="grey")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2.fill_between(daily["date"], daily["avg_score"], 0,
                     where=daily["avg_score"] >= 0,
                     color="#4575b4", alpha=0.15)
    ax2.fill_between(daily["date"], daily["avg_score"], 0,
                     where=daily["avg_score"] < 0,
                     color="#d73027", alpha=0.15)
else:
    ax2.text(0.5, 0.5, "无法解析时间数据", ha="center", va="center",
             transform=ax2.transAxes, fontsize=14, color="grey")
    ax2.set_title("BERT 情绪时间序列", fontsize=13)

# 底部信息
fig.text(0.5, 0.01,
         f"统计时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  "
         f"帖子数：{total}  "
         f"简单平均情绪：{avg_sentiment:+.4f}  "
         f"加权平均情绪：{weighted_avg:+.4f}（{weighted_avg_label}）",
         ha="center", fontsize=9, color="grey")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
print(f"[绘图] 图表已保存到 {OUTPUT_CHART}")
plt.close()
