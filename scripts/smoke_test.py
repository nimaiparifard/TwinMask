"""Quick end-to-end check: train 100 steps on Cora, evaluate, run phase-2 analysis on a tiny budget."""
from twinmask import phase2, tm

model, hist, sec = tm.train("cora", "joint", steps=100, bs=16, k=12, log_every=50)
model.eval()
res = tm.evaluate(model, "cora", "joint", 0, k=12, max_pairs=200, n_text=100)
r = phase2.analyse("cora", "joint", n_sub=64, k=12, grid=(0.0, 0.5, 1.0))
print("TSMI", r["tsmi"])
print("NELBO", r["nelbo"])
