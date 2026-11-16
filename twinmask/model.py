"""TwinMask-S: joint absorbing-state (masked) diffusion over node text tokens and edge variables.

Backbone: a pretrained masked LM (RoBERTa-base) processes each node's tokens independently
(intra-node attention). After selected layers, a GraphMixer lets the per-node summary tokens
(<s>, position 0) attend to each other with a per-head bias that depends on the *current, partially
masked* state of every edge variable (absent / present / [M]) -> dynamic topology attention.
Text head = the LM head on masked token positions; edge head = pair MLP on final node states.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from .common import TimeEmb

ABSENT, PRESENT, MASKED, SELF = 0, 1, 2, 3
import os as _os
_SB = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "hf_models", "all-distilroberta-v1")
SBERT_DIR = _os.environ.get("TWINMASK_ENCODER", _SB)
if SBERT_DIR == "none":
    SBERT_DIR = None


class GraphMixer(nn.Module):
    def __init__(self, d, heads=12, topo="dynamic"):
        super().__init__()
        self.h, self.topo = heads, topo
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.state_bias = nn.Embedding(4, heads)
        self.cond = nn.Linear(d, d)
        for m in (self.out, self.ff[-1]):  # start as identity so the pretrained LM is untouched
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        with torch.no_grad():  # start as attention over the *visible* graph (GAT-like message passing);
            # unknown [M] edges sit in between absent and present. A zero init leaves the mixer structure-blind
            # for the whole (short, early-stopped) training window.
            self.state_bias.weight[ABSENT] = -4.0
            self.state_bias.weight[MASKED] = -2.0
            self.state_bias.weight[PRESENT] = 0.0
            self.state_bias.weight[SELF] = 0.0

    def forward(self, x, state, valid, temb):
        B, k, d = x.shape
        if self.topo == "static":  # unknown edges are treated as absent (the "edge dropout" view)
            state = torch.where(state == MASKED, torch.full_like(state, ABSENT), state)
        elif self.topo == "none":  # structure-blind: every pair looks the same
            state = torch.where(state == SELF, state, torch.full_like(state, MASKED))
        bias = self.state_bias(state).permute(0, 3, 1, 2).float()  # B,H,k,k
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        hdn = self.ln1(x) + self.cond(temb)[:, None, :]
        q, kk, v = self.qkv(hdn).view(B, k, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, kk, v, attn_mask=bias.to(q.dtype))
        y = x + self.out(a.transpose(1, 2).reshape(B, k, d))
        y = y + self.ff(self.ln2(y))
        return y - x  # residual update (zero at init)


class TwinMask(nn.Module):
    def __init__(self, base="distilroberta-base", mix_after=(2, 4, 6), topo="dynamic", encoder_init=SBERT_DIR):
        super().__init__()
        self.lm = AutoModelForMaskedLM.from_pretrained(base, attn_implementation="sdpa")
        if encoder_init:  # contrastively pre-trained encoder (same arch/tokenizer); MLM head stays from `base`
            from transformers import AutoModel
            enc = AutoModel.from_pretrained(encoder_init)
            missing, unexpected = self.lm.roberta.load_state_dict(
                {k: v for k, v in enc.state_dict().items() if not k.startswith("pooler.")}, strict=False)
            assert not unexpected, unexpected
            del enc
        cfg = self.lm.config
        d = cfg.hidden_size
        self.mask_id = None
        self.mix_after = tuple(mix_after)
        self.mixers = nn.ModuleList([GraphMixer(d, cfg.num_attention_heads, topo) for _ in mix_after])
        self.tx, self.ta = TimeEmb(d), TimeEmb(d)
        self.final_mixer = GraphMixer(d, cfg.num_attention_heads, topo)
        self.pair_struct = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64))
        self.edge_head = nn.Sequential(nn.Linear(2 * d + 64, d), nn.GELU(), nn.Linear(d, 1))
        self.node_proj = nn.LayerNorm(d)
        self.deg_emb = nn.Linear(1, d)
        nn.init.normal_(self.deg_emb.weight, std=0.02); nn.init.zeros_(self.deg_emb.bias)

    def forward(self, ids, attn, valid, state, t_x, t_a, need_text=True):
        """ids/attn: B,k,L ; valid: B,k ; state: B,k,k ; t_x,t_a: B.
        Returns node states (B,k,d), token hidden (B*k,L,d) and edge logits (B,k,k)."""
        B, k, L = ids.shape
        enc = self.lm.roberta
        x = enc.embeddings(input_ids=ids.view(B * k, L).long())
        temb = self.tx(t_x) + self.ta(t_a)  # B,d
        x0 = x[:, 0] + temb.repeat_interleave(k, 0).to(x.dtype)
        x = torch.cat([x0[:, None], x[:, 1:]], 1)
        ext = torch.zeros(B * k, 1, 1, L, device=ids.device, dtype=x.dtype)
        ext = ext.masked_fill(~attn.view(B * k, 1, 1, L).bool(), torch.finfo(x.dtype).min)
        am = attn.view(B * k, L, 1)

        def pool(z):  # mean over the node's (possibly masked) tokens -> one summary vector per node
            m = am.to(z.dtype)
            return ((z * m).sum(1) / m.sum(1).clamp(min=1)).view(B, k, -1)

        # structural encodings from the *visible* graph only (masked edges contribute nothing):
        # softmax attention is a weighted mean and cannot count neighbours when node inputs coincide
        # (e.g. fully masked text), so degree and common-neighbour / path counts are given explicitly.
        A = (state == PRESENT).float()
        A2 = A @ A
        A3 = A2 @ A
        deg = A.sum(-1, keepdim=True)  # B,k,1
        degemb = self.deg_emb(torch.log1p(deg))
        mi = 0
        for li, layer in enumerate(enc.encoder.layer, 1):
            x = layer(x, attention_mask=ext)[0]
            if li in self.mix_after:
                delta = self.mixers[mi](pool(x).float() + degemb, state, valid, temb)
                x = x + delta.reshape(B * k, 1, -1).to(x.dtype)  # broadcast neighbour message to every token
                mi += 1
        node = pool(x).float() + degemb
        node = node + self.final_mixer(node, state, valid, temb)
        h = self.node_proj(node)
        a, b = h[:, :, None, :], h[:, None, :, :]
        ps = torch.stack([torch.log1p(A2), torch.log1p(A3), torch.log1p(deg * deg.transpose(1, 2))], -1)
        pair = torch.cat([a * b, (a - b).abs(), self.pair_struct(ps)], -1)
        e_logit = self.edge_head(pair).squeeze(-1)
        e_logit = 0.5 * (e_logit + e_logit.transpose(1, 2))
        return h, x, e_logit

    def token_logits(self, x_sel):
        return self.lm.lm_head(x_sel)
