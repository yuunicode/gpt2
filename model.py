import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """ 현재 토큰 이전까지만 정보를 가지고 있는 셀프어텐션, 토치의 모듈 상속 """

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_dim = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.block_size = block_size
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size() # x 선형변환
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (C // self.n_head) ** 0.5
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = attn @ v # 어텐션 합산 결과
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.dropout(self.proj(y))


class Block(nn.Module):
    """ 트랜스포머 블록 """

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT2Lite(nn.Module):
    """ GPT2 라이트버전 """
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd) #각 단어의 임베딩
        self.pos_emb = nn.Embedding(block_size, n_embd) # 포지션임베딩
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]) # 트랜스포머 블록을 N_LAYER개 쌓음
        self.ln_f = nn.LayerNorm(n_embd) # 마지막 레이어 정규화
        self.lm_head = nn.Linear(n_embd, vocab_size) # 어휘 분포 예측

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    def generate(self, idx, max_new_tokens):
        """ 프롬프트를 받아 다음 토큰들을 생성해나가는 AR 방식의 루프.

        Args:
            idx (Tensor): 초기 입력 토큰들
            max_new_tokens (_type_):생성할 새 토큰 수

        Returns:
            idx : 원래 입력 + 생성된 토큰들이 붙은 최종 시퀀스
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self.forward(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            
        return idx
