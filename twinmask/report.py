"""Aggregate results/*.json into markdown tables + figures (results/REPORT.md, results/fig_*.png)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .common import RESULTS


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    return f"{100*np.mean(xs):.2f}" + (f" ± {100*np.std(xs):.2f}" if len(xs) > 1 else "")


def phase0_tables():
    p = os.path.join(RESULTS, "phase0", "phase0_all.json")
    if not os.path.exists(p):
        return "_phase 0 not finished_\n"
    R = json.load(open(p))
    g = defaultdict(list)
    for r in R:
        g[(r["dataset"], r["variant"])].append(r)
    dsets = list(dict.fromkeys(r["dataset"] for r in R))
    vars_ = ["text_cosine", "gae", "mlp_lp", "edge_only", "text_only", "tied", "joint"]
    out = ["### Phase 0 — Link prediction (random negatives: AUC; HeaRT hard negatives: MRR, Hits@20), cold-start AUC, "
           "node classification (low-label public split / supervised 60-20-20); all in %, mean ± std over seeds\n",
           "| dataset | variant | LP AUC | HeaRT MRR | HeaRT H@20 | cold AUC | NC low-label | NC 60/20/20 |",
           "|---|---|---|---|---|---|---|---|"]
    for d in dsets:
        for v in vars_:
            rs = g.get((d, v), [])
            if not rs:
                continue
            out.append(f"| {d} | {v} | {ms([r.get('lp',{}).get('auc') for r in rs])} | {ms([r.get('heart',{}).get('mrr') for r in rs])} | "
                       f"{ms([r.get('heart',{}).get('hits@20') for r in rs])} | {ms([r.get('cold',{}).get('auc') for r in rs])} | "
                       f"{ms([r['nc']['acc'] for r in rs])} | {ms([r['nc'].get('sup_acc') for r in rs])} |")
    out += ["\n### Phase 0 — Text-embedding denoising: cosine(x̂0, x0) of test nodes, graph visible vs hidden\n",
            "| dataset | variant | t | noisy input | no graph | with graph | Δ (graph − no graph) |", "|---|---|---|---|---|---|---|"]
    for d in dsets:
        for v in ("text_only", "joint", "tied"):
            rs = g.get((d, v), [])
            if not rs or "denoise_cos" not in rs[0]:
                continue
            for t in ("0.5", "0.9"):
                ni = np.mean([r["denoise_cos"][t]["noisy_input"] for r in rs])
                ng = np.mean([r["denoise_cos"][t]["nograph"] for r in rs])
                gr = np.mean([r["denoise_cos"][t]["graph"] for r in rs])
                out.append(f"| {d} | {v} | {t} | {ni:.3f} | {ng:.3f} | {gr:.3f} | {gr-ng:+.3f} |")
    return "\n".join(out) + "\n"


def load_dir(tag):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, tag, "*.json")))]


def tm_tables(tag, title):
    R = load_dir(tag)
    if not R:
        return f"_{title}: no results yet_\n"
    out = [f"### {title} — link prediction / cold-start / node classification\n",
           "| dataset | variant | seed | LP AUC (full) | LP AUC text masked | LP AUC pair only | HeaRT MRR | HeaRT H@20 | cold AUC | NC low-label | NC 60/20/20 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r, k, m="auc": f"{100*r[k][m]:.2f}" if k in r and m in r[k] else "–"
    for r in R:
        out.append(f"| {r['dataset']} | {r['variant']} | {r['seed']} | {f(r,'lp_full')} | {f(r,'lp_textmasked')} | "
                   f"{f(r,'lp_paironly')} | {f(r,'heart','mrr')} | {f(r,'heart','hits@20')} | {f(r,'cold')} | "
                   f"{f(r,'nc','acc')} | {f(r,'nc','sup_acc')} |")
    out += [f"\n### {title} — text completion of a test node (masked-token NLL ↓ / top-1 acc ↑)\n",
            "| dataset | variant | mask ratio | alone | subgraph, edges [M] | subgraph + edges | NLL gain of context |",
            "|---|---|---|---|---|---|---|"]
    for r in R:
        if "text" not in r:
            continue
        for rt in ("0.5", "1.0"):
            a, e, c = r["text"][f"alone@{rt}"], r["text"][f"edges_masked@{rt}"], r["text"][f"ctx@{rt}"]
            out.append(f"| {r['dataset']} | {r['variant']} | {rt} | {a['nll']:.3f} / {100*a['acc']:.1f}% | "
                       f"{e['nll']:.3f} / {100*e['acc']:.1f}% | {c['nll']:.3f} / {100*c['acc']:.1f}% | {a['nll']-c['nll']:+.3f} |")
    return "\n".join(out) + "\n"


def phase2_section():
    R = load_dir("phase2")
    if not R:
        return "_phase 2 analysis not finished_\n"
    out = ["### Phase 2 — TSMI estimates (nats per item; NELBO differences, higher = stronger text↔structure dependence)\n",
           "| dataset | homophily | TSMI_A (text→edges) | TSMI_X topology (edges→text) | CTX_X (neighbourhood→text) |", "|---|---|---|---|---|"]
    for r in sorted(R, key=lambda r: -r["homophily"]):
        t = r["tsmi"]
        out.append(f"| {r['dataset']} | {r['homophily']:.3f} | {t['TSMI_A']:.4f} | {t['TSMI_X_topology']:.4f} | {t['CTX_X']:.4f} |")
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for r in R:
            if "heatmap" not in r:
                continue
            H = r["heatmap"]
            grid = sorted({float(k.split("|")[0]) for k in H})
            A = np.full((len(grid), len(grid)), np.nan); T = np.full_like(A, np.nan)
            for i, tx in enumerate(grid):
                for j, ta in enumerate(grid):
                    c = H[f"{tx}|{ta}"]
                    if c["edge_auc"] is not None: A[i, j] = c["edge_auc"]
                    if c["text_nll"] is not None: T[i, j] = c["text_nll"]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            for a_, M, ttl, cm in ((ax[0], A, "edge AUC on masked pairs", "viridis"), (ax[1], T, "text NLL on masked tokens", "magma_r")):
                im = a_.imshow(M, origin="lower", cmap=cm)
                a_.set_xticks(range(len(grid)), grid); a_.set_yticks(range(len(grid)), grid)
                a_.set_xlabel("t_a (edge mask rate)"); a_.set_ylabel("t_x (text mask rate)"); a_.set_title(f"{r['dataset']}: {ttl}")
                for i in range(len(grid)):
                    for j in range(len(grid)):
                        if not np.isnan(M[i, j]):
                            a_.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7, color="w")
                fig.colorbar(im, ax=a_)
            fig.tight_layout()
            fn = f"fig_heatmap_{r['dataset']}.png"
            fig.savefig(os.path.join(RESULTS, fn), dpi=130); plt.close(fig)
            figs.append(fn)
        # TSMI vs homophily
        fig, ax = plt.subplots(figsize=(5, 4))
        for key, mk in (("TSMI_A", "o"), ("CTX_X", "s")):
            xs = [r["homophily"] for r in R]; ys = [r["tsmi"][key] for r in R]
            ax.scatter(xs, ys, marker=mk, label=key)
            for r, x, y in zip(R, xs, ys):
                ax.annotate(r["dataset"], (x, y), fontsize=7)
        ax.set_xlabel("edge homophily"); ax.set_ylabel("nats / item"); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "fig_tsmi_vs_homophily.png"), dpi=130); plt.close(fig)
        figs.append("fig_tsmi_vs_homophily.png")
    except Exception as e:  # plotting is optional
        out.append(f"\n_(figures skipped: {e})_")
    out += [f"\n![{f}]({f})" for f in figs]
    return "\n".join(out) + "\n"


def main():
    md = ["# TwinMask — experimental results (auto-generated)\n", phase0_tables(),
          tm_tables("phase1", "Phase 1/2 (TwinMask-S, DistilRoBERTa backbone)"), phase2_section(),
          tm_tables("phase3", "Phase 3 (ogbn-arxiv)")]
    p = os.path.join(RESULTS, "REPORT.md")
    open(p, "w", encoding="utf-8").write("\n".join(md))
    print(p)


if __name__ == "__main__":
    main()
