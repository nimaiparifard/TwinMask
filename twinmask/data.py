"""Data preparation: extract TAGs from the DIVERGE codebase, build leakage-free link splits,
cache MiniLM sentence embeddings (phase 0) and RoBERTa token ids (phase 1+)."""
import os
import argparse
import numpy as np
import torch

DIVERGE_DIR = r"C:/Users/Nima/Desktop/Python Project/DIVERGE/datasets"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")


def undirected_unique(ei, n):
    ei = ei[:, ei[0] != ei[1]]
    a = torch.minimum(ei[0], ei[1]); b = torch.maximum(ei[0], ei[1])
    key = torch.unique(a * n + b)
    return torch.stack([key // n, key % n])  # (2, E) with u < v


def sample_negatives(n, num, forbid_keys, rng, allowed=None):
    """Uniform random non-edges (u<v). forbid_keys: set of u*n+v of all true edges."""
    out = []
    pool = allowed if allowed is not None else np.arange(n)
    seen = set()
    while len(out) < num:
        u = rng.choice(pool, 2 * num); v = rng.choice(pool, 2 * num)
        for a, b in zip(u, v):
            if a == b:
                continue
            a, b = (a, b) if a < b else (b, a)
            k = int(a) * n + int(b)
            if k in forbid_keys or k in seen:
                continue
            seen.add(k); out.append((a, b))
            if len(out) == num:
                break
    return torch.tensor(out, dtype=torch.long).t()


def build_split(name, seed=0, cold_frac=0.05, val_frac=0.05, test_frac=0.10):
    g = torch.load(os.path.join(DIVERGE_DIR, f"{name}.pt"), weights_only=False, map_location="cpu")
    n = g.num_nodes
    y = g.y.view(-1)
    E = undirected_unique(g.edge_index, n)
    rng = np.random.default_rng(seed)
    all_keys = set((E[0] * n + E[1]).tolist())

    # cold-start nodes: chosen among nodes with degree >= 1; all their edges are hidden from training
    deg = torch.bincount(torch.cat([E[0], E[1]]), minlength=n)
    cand = torch.nonzero(deg > 0).view(-1).numpy()
    cold = torch.from_numpy(np.sort(rng.choice(cand, int(cold_frac * n), replace=False)))
    is_cold = torch.zeros(n, dtype=torch.bool); is_cold[cold] = True
    touch_cold = is_cold[E[0]] | is_cold[E[1]]
    both_cold = is_cold[E[0]] & is_cold[E[1]]
    cold_pos = E[:, touch_cold & ~both_cold]
    rest = E[:, ~touch_cold]

    perm = torch.from_numpy(rng.permutation(rest.shape[1]))
    nt, nv = int(test_frac * rest.shape[1]), int(val_frac * rest.shape[1])
    test_pos = rest[:, perm[:nt]]; val_pos = rest[:, perm[nt:nt + nv]]; train_pos = rest[:, perm[nt + nv:]]

    warm = torch.nonzero(~is_cold).view(-1).numpy()
    test_neg = sample_negatives(n, test_pos.shape[1], all_keys, rng, warm)
    val_neg = sample_negatives(n, val_pos.shape[1], all_keys, rng, warm)
    # cold negatives: one endpoint is the cold node of each positive, other endpoint random warm node
    cn = []
    cold_endpoint = torch.where(is_cold[cold_pos[0]], cold_pos[0], cold_pos[1])
    for c in cold_endpoint.tolist():
        while True:
            v = int(rng.choice(warm)); a, b = min(c, v), max(c, v)
            if a * n + b not in all_keys:
                cn.append((a, b)); break
    cold_neg = torch.tensor(cn).t()

    texts = [t.replace("\n", " ").strip() for t in g.raw_texts]
    masks = {}
    for k in ("train_mask", "val_mask", "test_mask"):
        m = getattr(g, k, None)
        masks[k] = m.view(-1) if m is not None else None
    if masks["train_mask"] is None:  # 60/20/20 random split for datasets without public masks
        p = rng.permutation(n); m = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
        m[0][p[:int(.6 * n)]] = True; m[1][p[int(.6 * n):int(.8 * n)]] = True; m[2][p[int(.8 * n):]] = True
        masks = dict(zip(("train_mask", "val_mask", "test_mask"), m))
    hom = (y[E[0]] == y[E[1]]).float().mean().item()
    return dict(name=name, num_nodes=n, y=y, texts=texts, label_names=list(g.label_name),
                train_pos=train_pos, val_pos=val_pos, val_neg=val_neg, test_pos=test_pos, test_neg=test_neg,
                cold_nodes=cold, cold_pos=cold_pos, cold_neg=cold_neg, homophily=hom, **masks)


@torch.no_grad()
def embed_minilm(texts, bs=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    return torch.from_numpy(m.encode(texts, batch_size=bs, show_progress_bar=False, convert_to_numpy=True,
                                     normalize_embeddings=True)).float()


def tokenize_roberta(texts, L=64):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("roberta-base")
    ids = torch.full((len(texts), L), tok.pad_token_id, dtype=torch.int32)
    for s in range(0, len(texts), 4096):
        enc = tok(texts[s:s + 4096], truncation=True, max_length=L, padding="max_length")["input_ids"]
        ids[s:s + len(enc)] = torch.tensor(enc, dtype=torch.int32)
    return ids


def prepare(name, L=64):
    os.makedirs(PROC, exist_ok=True)
    out = os.path.join(PROC, f"{name}.pt")
    if os.path.exists(out):
        return
    d = build_split(name)
    d["emb"] = embed_minilm(d["texts"])
    d["tokens"] = tokenize_roberta(d["texts"], L)
    torch.save(d, out)
    print(f"[{name}] n={d['num_nodes']} train_e={d['train_pos'].shape[1]} test_e={d['test_pos'].shape[1]} "
          f"cold_nodes={len(d['cold_nodes'])} cold_e={d['cold_pos'].shape[1]} hom={d['homophily']:.3f}")


def load(name):
    return torch.load(os.path.join(PROC, f"{name}.pt"), weights_only=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "wikics", "instagram", "reddit"])
    a = ap.parse_args()
    for nm in a.datasets:
        prepare(nm)
