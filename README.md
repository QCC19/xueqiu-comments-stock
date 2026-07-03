# 雪球评论情感分析 (Xueqiu Sentiment Analysis)

基于 **BERT（bert-base-chinese）** 微调的雪球（xueqiu.com）股票评论情感分析模型。

## 项目结构

```
├── albert_finetune.ipynb          # BERT 微调训练脚本（Google Colab）
├── run_bert_inference.py          # BERT 推理脚本（对新评论打分）
├── deepseek/
│   ├── sentiment_analysis_deepseek.py  # DeepSeek API 打分脚本
│   └── merge.csv                      # 三只股票合并训练数据（比亚迪+亿纬锂能+国轩高科）
├── bert/
│   ├── plot_chart_bert.py         # 情感分析可视化
│   └── sentiment_bert_v2.png      # BERT v2 结果图
└── data/
    └── xueqiu_300750_bert_v2_scored.csv  # BERT v2 对宁德时代评论的打分结果
```

## 方法

1. **DeepSeek API** — 对评论进行 5 分法情感打分（对照组）
2. **BERT 微调** — 用 DeepSeek 的打标结果训练 bert-base-chinese，三分类（负面/中性/正面）

## 结果

BERT v2 在宁德时代 989 条评论上的表现：
- 正面/负面一致率 vs DeepSeek：**77% / 74%**
- 训练数据：比亚迪 + 亿纬锂能 + 国轩高科（约 3000 条）

## 模型

模型文件较大（~390MB），未上传至本仓库。
如需使用，可自行在 Colab 上运行 `albert_finetune.ipynb` 训练。
