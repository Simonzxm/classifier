Notification Classifier — Quick server

Files added:
- `main.py`: FastAPI app that loads `./model` and serves a `/predict` endpoint.
- `requirements.txt`: Python dependencies to run the server.

Quick start (run locally):

1) Create and activate your environment, then install deps:

```bash
cd /home/nova/classifier
pip install -r requirements.txt
```

2) Start the FastAPI server (uses GPU if available):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

3) Test the server with `curl`:

```bash
curl -X POST "http://120.55.180.172:7004/predict" -H "Content-Type: application/json" -d '{"texts": ["各学院、各单位：2025年度国家双一流——拔尖创新人才培养专项资金（项目号为“149112”）、江苏省双一流——拔尖创新人才培养专项资金（项目号为“1480602”）、江苏高校品牌专业建设工程专项经费（项目号为“14805”）即将使用完毕。其中，国家双一流——拔尖创新人才培养（项目号为“149112”）仅能预约“设备费”科目。11月27日（周四）17：00起不再受理以上各项专项资金的网上预约业务，剩余额度将自动清零。12月1日（下周一）下班前，已在财务处网上系统完成预约的单据请务必提交至财务处，逾期不予受理。未核销的暂付款请按财务处相关规定抓紧时间办理冲账报销手续。特此通知。","1月12日，南京大学本科教育教学改革推进会在仙林校区举行。本次会议旨在全面学习贯彻习近平总书记重要讲话和全国教育大会精神，总结回顾本科人才培养改革经验成效，分析研判当前形势及存在问题，以关键问题为导向，启动本科教育教学审核评估整改工作并开展新一轮本科教育教学改革大讨论，全力推动“奋进行动”《本科拔尖创新人才培养行动方案》进一步落地见效，全面提升拔尖创新人才自主培养质量。南京大学党委书记、中国科学院院士谭铁牛，校长、中国科学院院士谈哲敏出席会议并讲话。"]}'
```

Usage:

```bash
curl -X POST "http://120.55.180.172:7004/predict" -H "Content-Type: application/json" -d '{"texts": ["YOUR TEXT", "YOUR TEXT"]}'
```
