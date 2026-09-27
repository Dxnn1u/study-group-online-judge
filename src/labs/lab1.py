import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5

class Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head #12
        self.head_dim = config.n_embd // config.n_head # 768 / 12 = 64
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, attention_mask=None):
        Batch, Time, Channels = x.shape

        # 1. calculate q, k, v at a time
        q, k, v = self.c_attn(x).split(Channels, dim=2) 

        # 2. include multi-head (split channels in to n_head * head_dim)
        q = q.view(Batch, Time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(Batch, Time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(Batch, Time, self.n_head, self.head_dim).transpose(1, 2)

        # 3. attention score
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. mask
        mask = torch.tril(torch.ones(Time, Time, dtype=torch.bool, device=x.device))
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()
        att = att.masked_fill(~mask, torch.finfo(att.dtype).min)

        # 5. softmax
        att = F.softmax(att, dim=-1)

        # 6. average
        y = att @ v

        # 7. combine head (change back to original shape)
        y = y.transpose(1, 2).contiguous().view(Batch, Time, Channels)

        return self.c_proj(y)
    

class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # 768 -> 3072
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) # 3072 -> 768

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config:GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT2(nn.Module):
    def __init__(self, config:GPT2Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        B, T = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.wte(input_ids) + self.wpe(position_ids)

        for block in self.h:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    

def load_weights(model: GPT2) -> GPT2:
    hf = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    hf_sd = hf.state_dict()
    my_sd = model.state_dict()

    transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

    loaded  = set()

    with torch.no_grad():
        for hf_key, hf_tensor in hf_sd.items():
            if hf_key == "lm_head.weight":
                continue
            my_key = hf_key.replace("transformer.", "")
            if my_key.endswith(transposed):
                hf_tensor = hf_tensor.t()

            assert my_sd[my_key].shape == hf_tensor.shape, \
                f"{my_key}: {my_sd[my_key].shape} vs {hf_tensor.shape}"
            my_sd[my_key].copy_(hf_tensor)
            loaded.add(my_key)

    missing = set(my_sd.keys()) - loaded - {"lm_head.weight"}
    assert not missing, f"Missing keys: {missing}"
    
    return model


_model = None
_tokenizer = None

def _load():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        _model = load_weights(GPT2(GPT2Config())).eval()
        _tokenizer.pad_token = _tokenizer.eos_token # only executed once when _load()
        _tokenizer.padding_side = "left"
    return _model, _tokenizer

@torch.no_grad()
def gpt2_complete(
        input: list[str],
        max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:

    model, tokenizer = _load()
    max_seq_length = min(max_seq_length, model.config.n_positions)
    eos_id = tokenizer.eos_token_id

    enc = tokenizer(input, return_tensors="pt", padding=True) # enc includes input_ids and attention_mask
    ids = enc.input_ids
    batch_size = ids.shape[0]
    prompt_len = ids.shape[1]
    attention_mask = enc.attention_mask

    lengths = attention_mask.sum(dim=1)
    finished = lengths >= max_seq_length

    step_logits = []
    
    while not finished.all():
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids = position_ids.masked_fill(~attention_mask.bool(), 0)
        logits = model(ids, attention_mask=attention_mask, position_ids=position_ids) # (B, T, vocab_size)
        next_logits = logits[:, -1, :] # (1, vocab_size) : (1, 50257)

        active = ~finished
        next_logits = next_logits.masked_fill(~active[:, None], 0) # judge restriction, 0 for finished sequences

        next_token = next_logits.argmax(dim=-1) 
        next_token = torch.where(active, next_token, eos_id)

        ids = torch.cat([ids, next_token[:, None]], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device=ids.device)], dim=1)
        step_logits.append(next_logits)

        lengths = lengths + active.long()
        finished = finished | (next_token == tokenizer.eos_token_id)
    
    if len(step_logits) == 0:
        logits = torch.empty((batch_size, 0, model.config.vocab_size), device=ids.device)
    else:
        logits = torch.stack(step_logits, dim=1) # (1, T, vocab_size) 
    generated = ids[:, prompt_len:] # (B, T)
    completions = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return completions, logits