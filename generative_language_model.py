import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dataclasses import dataclass
from heinsen_attention import LogAttention


@dataclass
class ModelConfig:
    vocab_sz: int = 50304   # vocab size of 50257, padded up for efficiency
    d_emb: int = 768        # number of embedding features
    n_layers: int = 24      # number of residual layers
    n_heads: int = 24       # number of heads per token
    d_key: int = 32         # number of key features per head
    d_val: int = 32         # number of value features per head


class ResidualLayer(nn.Module):
    """
    A simple causal (autoregressive) residual layer.
    
    Input shapes:
        tokens: [..., n_tok, d_emb].

    Output shapes:
        tokens: [..., n_tok, d_emb].
    """
    def __init__(self, d_emb, n_heads, d_key, d_val):
        super().__init__()
        self.d_emb, self.n_heads, self.d_key, self.d_val = (d_emb, n_heads, d_key, d_val)
        self.feedforward1 = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, n_heads * (d_key + d_key + d_val)),
        )
        self.log_attention = LogAttention(is_causal=True)
        self.feedforward2 = nn.Sequential(
            nn.Linear(n_heads * d_val, d_emb * 2),
            nn.GLU(dim=-1),
            nn.Linear(d_emb, d_emb, bias=False),
        )

    def extra_repr(self):
        return ', '.join('{}={}'.format(s, getattr(self, s)) for s in 'd_emb n_heads d_key d_val'.split(' '))

    def forward(self, inp, using_prev_context=False):
        x = self.feedforward1(inp)                                 # [..., n_toks, n_heads * (d_key + d_key + d_val)]
        x = x.view(*x.shape[:-1], self.n_heads, -1)                # [..., n_toks, n_heads, d_key + d_key + d_val]
        x = x.transpose(-3, -2)                                    # [..., n_heads, n_toks, d_key + d_key + d_val]
        x = x.split([self.d_key, self.d_key, self.d_val], dim=-1)  # tuple of three tensors
        x = self.log_attention(*x, using_prev_context)             # [..., n_heads, n_toks, d_val]
        x = x.transpose(-3, -2).flatten(-2)                        # [..., n_toks, n_heads * d_val]
        x = self.feedforward2(x)                                   # [..., n_toks, d_emb]
        return inp + x 


class EmbedPosition(nn.Module):
    """
    As proposed by Franz A. Heinsen, March 2024.
    
    Input shapes:
        tokens: [..., n_tok, d_emb].

    Output shapes:
        tokens: [..., n_tok, d_emb].
    """
    def __init__(self, d_emb):
        super().__init__()
        self.d_emb = d_emb
        self.dense = nn.Linear(d_emb, d_emb * 2)

    def extra_repr(self):
        return 'd_emb={}'.format(d_emb)

    def _log_linear_recurrence(self, log_coeffs, prepended_logits):
        "Applies method proposed in https://arxiv.org/abs/2311.06281."
        a_star = F.pad(log_coeffs.cumsum(dim=-2), (0,0, 1,0), value=0)              # [..., 1 + n_tok, d_emb]
        logit0_plus_b_star = torch.logcumsumexp(prepended_logits - a_star, dim=-2)  # [..., 1 + n_tok, d_emb]
        log_linear_recurrence = a_star + logit0_plus_b_star                         # [..., 1 + n_tok, d_emb]
        return log_linear_recurrence[..., 1:, :]                                    # [..., n_tok, d_emb]

    def forward(self, tokens, using_prev_context):
        tup = self.dense(tokens).split(self.d_emb, dim=-1)                          # [..., n_tok, d_emb] x 2
        log_coeffs, logits = (F.logsigmoid(tup[0]), tup[1])                         # [..., n_tok, d_emb] x 2
        if using_prev_context:
            prepended_logits = torch.cat([self.prev_context, logits], dim=-2)       # [..., 1 + n_tok, d_emb]
        else: 
            prepended_logits = F.pad(logits, (0,0, 1,0), value=0)                   # [..., 1 + n_tok, d]
        pos_embs = self._log_linear_recurrence(log_coeffs, prepended_logits)        # [..., n_tok, d_emb]
        self.prev_context = pos_embs[..., -1:, :].detach()                          # [..., 1, d_emb]
        return tokens + pos_embs                                                    # [..., n_tok, d_emb]


class GenerativeLanguageModel(nn.Module):
    """
    Given a sequence of token ids, predict each next token id.

    Input shape:
        token_ids: [..., n_toks], sequence of token ids.

    Output shape:
        predicted logits [..., n_toks, vocab_sz].
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        _initial_embs = torch.empty(config.vocab_sz, config.d_emb).uniform_(-1, 1) / sqrt(config.d_emb)
        self.embed = nn.Embedding(*_initial_embs.shape, _weight=_initial_embs)
        self.embed_pos = EmbedPosition(config.d_emb)
        self.layers = nn.Sequential(*[
            ResidualLayer(config.d_emb, config.n_heads, config.d_key, config.d_val)
            for _ in range(config.n_layers)
        ])
        self.lnorm = nn.LayerNorm(config.d_emb)
        self.config = config

    def extra_repr(self):
        return 'config={}'.format(self.config)

    def body(self, token_ids, using_prev_context=False):
        x = self.embed(token_ids)
        x = self.embed_pos(x, using_prev_context)
        for layer in self.layers:
            x = layer(x, using_prev_context)
        x = self.lnorm(x)
        return x

    def head(self, x):
        return x @ self.embed.weight.T

    def forward(self, token_ids, using_prev_context=False):
        x = self.body(token_ids, using_prev_context)
        x = self.head(x)
        return x

    # Convenience methods:

    def get_param_groups(self, weight_decay):
        decay_attrs = { nn.Embedding: ['weight'], nn.Linear: ['weight'], }
        decay_modules = set(m for m in self.modules() if type(m) in decay_attrs.keys())
        decay_ids = set(id(getattr(m, attr)) for m in decay_modules for attr in decay_attrs[type(m)])
        return [
            { 'params': [p for p in self.parameters() if id(p) in decay_ids], 'weight_decay': weight_decay, },
            { 'params': [p for p in self.parameters() if id(p) not in decay_ids], 'weight_decay': 0.0, },
        ]

    @torch.no_grad()
    def generate(self, token_ids, n_new, temp=1.0, topk=None, using_prev_context=False, show_progress=False):
        assert self.training is False, "Model should be in eval mode."
        generated_ids = []
        upc_states = [using_prev_context] + [True] * (n_new - 1)
        for upc_state in (tqdm(upc_states) if show_progress else upc_states):
            hidden_states = self.body(token_ids, using_prev_context=upc_state)
            logits = self.head(hidden_states[..., -1, :]) / temp
            if topk is not None:
                min_of_topk = logits.topk(topk, dim=-1).values.min(dim=-1, keepdim=True).values
                logits[logits < min_of_topk] = float('-inf')
            token_ids = torch.multinomial(logits.softmax(dim=-1), num_samples=1) 
            generated_ids.append(token_ids)
        return torch.cat(generated_ids, dim=-1)


def build_model(**model_config_kwds):
    return GenerativeLanguageModel(ModelConfig(**model_config_kwds))
