from transformers import AutoTokenizer, pipeline
import re

# Simple sanitizer (same logic as in train.py)
def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '[URL_PLACEHOLDER]', text)

model_path = "./model"

# Load tokenizer (fix_mistral_regex for compatibility) and pipeline
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model_path, tokenizer=tokenizer)

# Expanded test_texts (copied from train.py)
test_texts = [
    "恭喜，您的积分已达到领取标准，请点击 [URL_PLACEHOLDER] 兑换。",
    "据当地媒体报道，昨晚的地震未造成人员伤亡。",
    "尊敬的用户，您的账户余额将于 24 小时内到期，请及时续费以避免服务中断。",
    "本研究提出了一种基于注意力机制的模型压缩方法，在多个基准上实现近似原模型的性能。",
    "学校通知：期末考试安排已发布，请登录教务系统查看具体座位和考场信息。",
    "统计数据显示，第三产业在本季度对 GDP 增长的贡献率继续上升，消费驱动效果显著。",
    "系统提醒：您有一条新的消息，请在 App 中查看详情。",
    "研究团队通过长期跟踪样本，发现该变异体的传播路径具有区域性差异性。",
    "本次艺术展展出的作品主要探讨记忆与影像之间的关系，吸引了大量观众参观。",
    "教务处公告：下周一开始的选课系统将按学号分批开放，请务必在规定时间内完成。",
    "公司季度财报显示，净利润较上年同期增长，管理层计划将更多资源投入产品研发。",
    "安全提示：检测到您的账号存在异常登录风险，请立即修改密码并开启双因素认证。",
    "天文学家在新的光谱数据中识别出可能的行星形成区，后续观测正在安排中。",
    "温馨提示：为了保证考试公平，考场禁止携带任何电子设备。",
    "一篇综述文章总结了近年来在小样本学习领域的主要进展与挑战。",
    "您的快递（单号 SF123456）已到达，请及时取件。",
    "白皮书建议加强基础设施韧性，以应对未来极端天气带来的风险。",
    "请于本周五前提交年度绩效自评表，逾期视为自动放弃。",
    "近日研究表明，城市绿地的增加有助于降低局部热岛效应且提升居民幸福感。",
    "作品评论指出，该小说以史实为底，艺术化处理了时代人物的心理描写。",
]

label_map = {'LABEL_0': '非通知', 'LABEL_1': '通知'}

print("--- 开始对样例文本进行快速预测 ---\n")
for text in test_texts:
    cleaned = sanitize_text(text)
    pred = classifier(cleaned)[0]
    print(f"文本: {cleaned}")
    print(f"  -> 预测: {label_map.get(pred['label'], pred['label'])} (score={pred['score']:.4f})\n")
