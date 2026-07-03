# 雪球评论情感分析 (Xueqiu Sentiment Analysis)

基于 **BERT（bert-base-chinese）** 微调的雪球（xueqiu.com）股票评论情感分析模型。

## 项目结构

```
├── .gitignore
├── README.md
├── bert/
│   ├── albert_finetune.ipynb          # BERT 微调训练脚本（Colab）
│   ├── run_bert_inference.py          # BERT 推理评分脚本
│   ├── plot_chart_bert.py             # 情感分析可视化
│   └── sentiment_bert_v2.png          # BERT v2 结果图
└── data/
    ├── merge.csv                      # 三只股票合并训练数据（比亚迪+亿纬锂能+国轩高科）
    └── xueqiu_300750_bert_v2_scored.csv  # BERT v2 对宁德时代评论的评分结果
```

## 方法

1. **DeepSeek API** — 对评论进行 5 分法情感打分（对照组）
2. **BERT 微调** — 用 DeepSeek 的评分结果训练 bert-base-chinese，三分类（负面/中性/正面）

## 代码说明

| 文件 | 作用 |
|:--|:--|
| `bert/albert_finetune.ipynb` | **训练模型。** 在 Google Colab 上运行，用 merge.csv 训练 BERT，输出模型文件 |
| `bert/run_bert_inference.py` | **用模型评分。** 加载训练好的 BERT 模型，对新的雪球评论逐条打分，输出 CSV |
| `bert/plot_chart_bert.py` | **画图。** 读取评分结果，生成情感分布柱状图和时间序列图 |

## 结果

BERT v2 在宁德时代 989 条评论上的表现：
- 正面/负面一致率 vs DeepSeek：**77% / 74%**
- 训练数据：比亚迪 + 亿纬锂能 + 国轩高科（约 3000 条）

## 模型

模型文件较大（~390MB），未上传至本仓库。
如需使用，可在 Colab 上运行 `bert/albert_finetune.ipynb` 自行训练。
