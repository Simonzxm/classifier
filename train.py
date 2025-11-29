import pandas as pd
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    pipeline,
    # 引入早停回调函数
    EarlyStoppingCallback 
)
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 1. 数据清洗与预处理函数 ---

def sanitize_text(text):
    """
    使用占位符替换文本中的URL，以减少噪声并保留“链接存在”的信号。
    """
    if not isinstance(text, str):
        return ""
    
    # 匹配常见的URL模式
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # 替换所有找到的URL为占位符 [URL_PLACEHOLDER]
    sanitized_text = re.sub(url_pattern, '[URL_PLACEHOLDER]', text)

    # 匹配常见邮箱地址 (包含 mailto: 前缀的情况)
    # 示例匹配：user@example.com, first.last+tag@sub.domain.co
    email_pattern = r"(?:mailto:)?\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    # 将邮箱替换为占位符 [EMAIL_PLACEHOLDER]
    sanitized_text = re.sub(email_pattern, '[EMAIL_PLACEHOLDER]', sanitized_text)

    return sanitized_text

# --- 2. 准备数据：从 CSV 文件加载 ---

DATA_FILE = "data.csv"

try:
    # 从 CSV 文件加载数据
    df = pd.read_csv(DATA_FILE)
    print(f"✅ 成功从 {DATA_FILE} 加载 {len(df)} 条样本。")
except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {DATA_FILE}。请确保文件已创建并放在同一目录下。")
    exit()

# 应用清洗函数
df['text'] = df['text'].apply(sanitize_text)

# 划分训练集和测试集
# 注意：如果样本总量增加，这里的 test_size=0.3 应该仍然合理
X_train, X_eval, y_train, y_eval = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

# 重新组合成 DataFrame 供 Dataset 使用
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
eval_df = pd.DataFrame({'text': X_eval, 'label': y_eval})

# 转换为 Hugging Face Datasets 格式
train_dataset = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
eval_dataset = Dataset.from_pandas(eval_df).remove_columns(["__index_level_0__"])


# --- 3. 模型加载与 Tokenization ---

MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
NUM_LABELS = 2 

# 加载分词器和模型 (num_labels=2 for binary classification)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)

# 定义分词函数
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# 应用分词函数到数据集
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# --- 4. 定义评估指标 ---

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1) 
    labels = p.label_ids
    
    # 关注 F1-score (二元分类) 和 Accuracy
    acc = accuracy_score(labels, preds)
    # 使用 'binary' f1
    f1 = f1_score(labels, preds, average='binary') 
    
    return {"accuracy": acc, "f1_score": f1}

# --- 5. 配置训练参数和 Trainer ---

# **Early Stopping 参数**
EARLY_STOPPING_PATIENCE = 5 # 连续 5 轮评估指标不改善就停止
EARLY_STOPPING_THRESHOLD = 0.001 # 容忍的最小变化阈值

training_args = TrainingArguments(
    output_dir="./results",
    local_rank=0,
    num_train_epochs=25,                  # 最大轮次，早停机制会提前终止
    per_device_train_batch_size=4,        
    per_device_eval_batch_size=4,
    warmup_steps=50,
    learning_rate=1e-5,
    weight_decay=0.05,                   
    max_grad_norm=1.0,                   
    logging_dir='./notification_logs',
    logging_steps=10,
    eval_strategy="epoch",         
    save_strategy="epoch",               
    load_best_model_at_end=True,         
    metric_for_best_model="f1_score",    # 监控 F1-score
    greater_is_better=True,              # F1-score 越高越好
)

# 初始化 Trainer 并添加 EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE, 
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD
    )]
)

# --- 6. 开始训练、评估和保存 ---

print("--- 开始模型微调 ---")
# 训练将由 EarlyStoppingCallback 控制停止
trainer.train() 
print("--- 微调完成 ---")

# 评估最终模型 (即最佳模型检查点)
eval_results = trainer.evaluate()
print("\n最终评估结果:", eval_results)

# 保存微调后的模型和分词器
model_save_path = "./model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"\n✅ 模型已保存到 {model_save_path}")

# --- 7. 加载和测试保存的模型 (已修复 Tokenizer 警告) ---

# 修复 Tokenizer 警告：加载时添加 fix_mistral_regex=True
final_tokenizer = AutoTokenizer.from_pretrained(model_save_path, fix_mistral_regex=True)

final_classifier = pipeline(
    "text-classification", 
    model=model_save_path, 
    tokenizer=final_tokenizer # 使用修复后的 tokenizer 实例
)

test_texts = [
    "恭喜，您的积分已达到领取标准，请点击 [URL_PLACEHOLDER] 兑换。", # 新通知 (已清洗)
    "据当地媒体报道，昨晚的地震未造成人员伤亡。", # 新闻
    "尊敬的用户，您的账户余额将于 24 小时内到期，请及时续费以避免服务中断。", # 通知类
    "本研究提出了一种基于注意力机制的模型压缩方法，在多个基准上实现近似原模型的性能。", # 学术
    "学校通知：期末考试安排已发布，请登录教务系统查看具体座位和考场信息。", # 通知
    "统计数据显示，第三产业在本季度对 GDP 增长的贡献率继续上升，消费驱动效果显著。", # 新闻/报告
    "系统提醒：您有一条新的消息，请在 App 中查看详情。", # 通知（短）
    "研究团队通过长期跟踪样本，发现该变异体的传播路径具有区域性差异性。", # 学术/研究
    "本次艺术展展出的作品主要探讨记忆与影像之间的关系，吸引了大量观众参观。", # 文化/新闻
    "教务处公告：下周一开始的选课系统将按学号分批开放，请务必在规定时间内完成。", # 通知
    "公司季度财报显示，净利润较上年同期增长，管理层计划将更多资源投入产品研发。", # 报告
    "安全提示：检测到您的账号存在异常登录风险，请立即修改密码并开启双因素认证。", # 通知/安全
    "天文学家在新的光谱数据中识别出可能的行星形成区，后续观测正在安排中。", # 科学新闻
    "温馨提示：为了保证考试公平，考场禁止携带任何电子设备。", # 通知
    "一篇综述文章总结了近年来在小样本学习领域的主要进展与挑战。", # 学术综述
    "您的快递（单号 SF123456）已到达，请及时取件。", # 通知/物流
    "白皮书建议加强基础设施韧性，以应对未来极端天气带来的风险。", # 政策/报告
    "请于本周五前提交年度绩效自评表，逾期视为自动放弃。", # 通知/行政
    "近日研究表明，城市绿地的增加有助于降低局部热岛效应且提升居民幸福感。", # 科研/新闻
    "作品评论指出，该小说以史实为底，艺术化处理了时代人物的心理描写。", # 文学/书评
]

print("\n--- 微调模型测试结果 ---")
for text in test_texts:
    cleaned_text = sanitize_text(text) 
    
    prediction = final_classifier(cleaned_text)[0]
    
    label_map = {'LABEL_0': '非通知', 'LABEL_1': '通知'}
    
    print(f"文本: '{cleaned_text[:20]}...'")
    print(f"  -> 预测类别: {label_map.get(prediction['label'], '未知')} (置信度: {prediction['score']:.4f})")