import os, pickle, wandb, json
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from model import GPT2Lite
from tqdm import tqdm

import torch
import sys
print("Torch version:", torch.__version__) #2.5.1+cu121
print("CUDA version in torch:", torch.version.cuda) #12.1

# JSON에서 config 불러오기
with open("config.json", "r") as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("❌ CUDA device not available. Training requires a GPU.")

# wandb를 이용해서 파라미터/로스 체킹
wandb.init(entity = "llmtesting",
           project="gpt2lite", 
           config=config)

# 토크나이저 로딩 (학습 안하고 SK 토크나이저 씀)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=config["tokenizer_path"])
vocab_size = tokenizer.vocab_size

class TxtDataset(Dataset):
    def __init__(self, text_file, debug_all=False):
        with open(text_file, encoding='utf-8') as f:
            lines = f.readlines()
        text = "".join(lines)
        ids = tokenizer.encode(text)

        self.data = [ids[i:i + config["block_size"] + 1] for i in range(len(ids) - config["block_size"])]
        # 디버깅: 블럭이 얼마나 자르는지 궁금해서 넣음
        with open("debug_blocks.txt", "w", encoding="utf-8") as f:
            max_blocks = len(self.data) if debug_all else min(len(self.data), 100)
            for i, chunk in enumerate(self.data[:max_blocks]):
                input_text = tokenizer.decode(chunk[:-1])
                f.write(f"[Block {i+1}]\n{input_text}\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])
    
# 모델 정의
model = GPT2Lite(vocab_size, 
                 block_size = config["block_size"],
                 n_layer = config["n_layer"],
                 n_head = config["n_head"],
                 n_embd = config["n_embd"]
                 ).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
step = 0

os.makedirs(config["checkpoint_dir"], exist_ok=True)

# 훈련 루프
for stage in config["curriculum"]:
    dataset = TxtDataset(stage)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(config["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"[{stage}] Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            wandb.log({"loss": loss.item(), "step": step})
            pbar.set_postfix(loss=loss.item())

            if step % config["save_every"] == 0:
                path = f"{config['checkpoint_dir']}/model-epoch{epoch+1}-step-{step}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(model.state_dict(), f)
                print(f"[✓] Saved {path}")
