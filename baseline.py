from transformers import pipeline
import re

def sanitize_text(text):
    """
    使用占位符替换文本中的URL。
    """
    # 匹配常见的URL模式 (http/https, www.开头, 或以域名结尾)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # 替换所有找到的URL为占位符 [URL_PLACEHOLDER]
    # 这个占位符将作为一个单独的、有意义的 Token 输入模型
    sanitized_text = re.sub(url_pattern, '[URL_PLACEHOLDER]', text)
    
    return sanitized_text

# ----------------------------------------------------
# 步骤 1: 加载 Zero-Shot 分类器 (与之前相同)
# ----------------------------------------------------

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=0 # 尽量使用 GPU
)

# ----------------------------------------------------
# 步骤 2: 定义待分类的文本和二元标签
# ----------------------------------------------------

texts_to_classify = [
    "各位同学，请查收关于组织2025级本科生开展新生入学教育 线上测试的通知，在10月17日之前完成，每位学生有3次作答机会，测试成绩90分及以上为合 格，3次测试成绩均不合格的学生须参加集中补考。测试连接https://njuxsxy.mh.chaoxing.com",
    "【重要通知】 各位同学，本次运动会乙组所有后续赛事（大一）将在期中考试后择一周末在鼓楼校区举行，具体安排另行通知。",
    "为全面复盘2025年本科招生工作成效，系统谋划2026年本科招生政策优化方向，本科生院于8月25日至29日召开2025年本科招生工作总结专题会议，会议由本科生院副院长、本科招生办公室主任陈琳主持。会上，本科生院院长王骏首先向全国各省份招生组致以诚挚感谢。他指出，2025年本科招生工作面临诸多挑战，各招生组工作人员始终坚守一线岗位，在高强度的工作节奏下攻坚克难，高效完成了各项招生任务，为学校生源质量的稳定提供了坚实保障。随后，本科生院与各省份招生组围绕招生工作实际展开深度研讨，针对招生宣传、生源对接、政策执行等环节中遇到的难点问题进行了全面梳理，共同分析问题根源并探讨解决方案。最后，陈琳主任对后续招生工作提出了展望。她强调，各招生组需进一步深化与重点生源高中的常态化交流合作，通过开展教授进中学、南星梦想计划、校园开放日等多样化活动，持续扩大南京大学品牌影响力，吸引更多优质生源报考，从而推动学校本科招生质量实现全面提升。全国各省份招生组组长及工作人员、本科生院本科招生办公室全体成员参会，共同为学校招生工作的持续优化建言献策。"
]

# 核心：定义二元标签
candidate_labels = ["通知消息", "非通知消息"]

# ----------------------------------------------------
# 步骤 3: 执行分类并判断
# ----------------------------------------------------

print("\n--- Zero-Shot 二元分类结果 (通知 vs. 非通知) ---")

for text in texts_to_classify:
    # 执行分类
    text = sanitize_text(text)
    result = classifier(text, candidate_labels)
    
    # 提取 "通知消息" 的分数
    
    # 获取标签列表中的 "通知消息" 的索引
    notification_index = result['labels'].index("通知消息")
    
    # 获取对应的置信度分数
    notification_score = result['scores'][notification_index]
    
    # 设定一个阈值（例如 0.8），高于阈值即认为是通知
    is_notification = notification_score >= 0.8 

    print(f"文本: '{text[:20]}...'")
    print(f"  -> 通知分数: {notification_score:.4f}")
    print(f"  -> 最终判定: {'✅ 是通知' if is_notification else '❌ 不是通知'}")
    
    if notification_score < 0.8 and notification_score > 0.4:
        print("  -> (提示：分数接近阈值，模型可能存在犹豫。)")