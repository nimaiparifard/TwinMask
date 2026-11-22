"""Summarise results/path_rl/*.json as markdown (written to results/path_rl/REPORT.md and printed)."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "results", "path_rl")
ORDER = ["diagonal", "diagonal_random", "text_first", "edge_first", "text_lead2", "text_lead1",
         "edge_lead1", "edge_lead2", "joint_conf", "policy_grpo"]


def d(m, kk):
    v = m.get("vs_diagonal", {}).get(kk)
    return "" if v is None else f"{v['mean']:+.3f}±{v['ci95']:.3f}"


def main():
    out = []
    for f in sorted(glob.glob(os.path.join(DIR, "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        out.append(f"\n### {r['dataset']} (homophily {r['homophily']:.2f}) — rho={r['rho']}, N={r['steps']}, "
                   f"mode={r['mode']}, test cold nodes={r['n_test']} x {r['reps']} resamplings")
        out.append(f"tuned fixed path (selected on disjoint fit nodes): **{r['tuned_path']}**\n")
        out.append("| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for nm in ORDER:
            m = r["paths"].get(nm)
            if m is None:
                continue
            out.append(f"| {nm} | {m['text_nll']:.3f} | {d(m, 'text_nll')} | {m['text_acc']:.3f} | {m['edge_nll']:.4f} | "
                       f"{d(m, 'edge_nll')} | {m['edge_auc']:.3f} | {m['edge_f1']:.3f} |")
        p = r["paths"].get("policy_grpo")
        if p:
            out.append(f"\npolicy text share per step: {[round(x, 2) for x in p['text_share_per_step']]}")
        if r.get("sweep"):
            out.append("\n| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |")
            out.append("|---|---|---|---|---|---|---|")
            Ns = sorted({int(kk.split("@")[1]) for kk in r["sweep"]})
            for N in Ns:
                a, b = r["sweep"][f"diagonal@{N}"], r["sweep"][f"joint_conf@{N}"]
                out.append(f"| {N} | {a['text_nll']:.3f} | {a['edge_nll']:.4f} | {a['edge_auc']:.3f} | "
                           f"{b['text_nll']:.3f} | {b['edge_nll']:.4f} | {b['edge_auc']:.3f} |")
    txt = "# TwinMask-Path results\n\nΔ = paired mean difference to the diagonal path on the same episodes (95% CI); " \
          "negative NLL Δ = better.\n" + "\n".join(out) + "\n"
    with open(os.path.join(DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
