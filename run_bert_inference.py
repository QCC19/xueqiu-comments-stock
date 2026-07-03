import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

model = AutoModelForSequenceClassification.from_pretrained('D:/xueqiu/bert_model_v2').to(device)
tokenizer = AutoTokenizer.from_pretrained('D:/xueqiu/bert_model_v2')
model.eval()

# 2. 读取宁德时代评论
df = pd.read_csv('D:/xueqiu/deepseek/xueqiu_300750_full_sentiment_deepseek.csv', encoding='utf-8-sig')
df = df[df['text'].notna() & (df['text'].str.len() >= 3)].copy()
print(f'宁德时代评论数: {len(df)}')

# 3. 逐批打分
labels_map = {0: '负面', 1: '中性', 2: '正面'}
bert_labels = []
bert_probs = []

batch_size = 32

for i in range(0, len(df), batch_size):
    batch_texts = df['text'].iloc[i:i+batch_size].tolist()
    inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    for prob in probs:
        pred = int(prob.argmax())
        bert_labels.append(labels_map[pred])
        bert_probs.append(prob.max())

# 4. 写入新列
df['bert_label'] = bert_labels
df['bert_prob'] = bert_probs

# 5. 保存
output_path = 'D:/xueqiu/data/xueqiu_300750_bert_v2_scored.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'\n完成！已保存到 {output_path}')
print(f'\nBERT 评分分布:')
print(df['bert_label'].value_counts())
