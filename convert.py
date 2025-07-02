import json
from pathlib import Path

with open("dataset/ko_wiki_v1_squad.json", encoding="utf-8") as f:
    data = json.load(f)

stage1_lines = []
stage2_lines = []

for item in data["data"] :
    title = item.get("title", "")
    for para in item["paragraphs"]:
        context = para["context"].replace("\n", " ").strip()
        for qa in para["qas"]:
            question = qa["question"].strip()
            for answer in qa["answers"]:
                answer_text = answer["text"].strip()

                # stage1: Q&A
                stage1_lines.append(f"사용자: {question}\n시스템: {answer_text}\n")

                # stage2: Context + Q&A
                stage2_lines.append(f"문서: {context}\n 질문: {question}\n정답: {answer_text}\n")

# 저장
Path("data").mkdir(exist_ok=True)
with open("data/train_stage1.txt", "w", encoding="utf-8") as f:
    f.writelines("\n".join(stage1_lines))

with open("data/train_stage2.txt", "w", encoding="utf-8") as f:
    f.writelines("\n".join(stage2_lines))