import torch
import pickle, json
from transformers import PreTrainedTokenizerFast
from model import GPT2Lite  # gpt2lite.py 또는 model.py 내에 정의되어 있어야 함

# 경로 설정
MODEL_PATH = "./checkpoints/model-1000.pkl"  # 원하는 체크포인트
TOKENIZER_PATH = "./tokenizer/sk_base_tokenizer.json"

# JSON에서 config 불러오기
with open("config.json", "r") as f:
    config = json.load(f)

# 토크나이저 로드
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

# 모델 구성
model = GPT2Lite(
    vocab_size=tokenizer.vocab_size,
    block_size=config["block_size"],
    n_layer=config["n_layer"],
    n_head=config["n_head"],
    n_embd=config["n_embd"]
)

# ✅ state_dict 로드 (pkl 방식)
with open(MODEL_PATH, "rb") as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ 생성 함수
@torch.no_grad()
def generate_text(prompt, max_new_tokens=60):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ✅ 인터랙티브 사용
print("🧠 GPT2Lite 대화 시작 (종료: 'exit')\n")
while True:
    prompt = input("📝 입력: ")
    if prompt.strip().lower() in ["exit", "quit"]:
        print("👋 종료합니다.")
        break
    result = generate_text(prompt)
    print("🤖 GPT2Lite:", result)