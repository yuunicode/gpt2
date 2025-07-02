#############################################################################
#                           모델 테스트용 ㅎㅎ                                 #
#############################################################################
import torch
from transformers import PreTrainedTokenizerFast
from model import GPT2Lite
import pickle

# config와 tokenizer 불러오기
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/sk_base_tokenizer.json")
vocab_size = tokenizer.vocab_size
block_size = 128

# 테스트 문장
prompt = "질문: 다테 기미코가 최초로 은퇴한 날짜는?\n정답:"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 공통 모델 설정
def load_model_from_pkl(pkl_path):
    model = GPT2Lite(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=256
    ).to("cpu")
    with open(pkl_path, "rb") as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 비교용 함수
def generate_and_print(model, label):
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=30)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n[{label}]")
    print(output_text)

# 두 모델 비교
model1 = load_model_from_pkl("checkpoints/model-epoch1-step-1000.pkl")
generate_and_print(model1, "STEP 1000")

model2 = load_model_from_pkl("checkpoints/model-epoch1-step-7000.pkl")
generate_and_print(model2, "STEP 7000")

model3 = load_model_from_pkl("checkpoints/model-epoch1-step-25000.pkl")
generate_and_print(model2, "STEP 25000")