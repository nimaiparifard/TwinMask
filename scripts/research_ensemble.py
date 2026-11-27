"""Single-denoiser TAG ensemble pilot on the cached MiniLM representations."""
import argparse
import time
from pathlib import Path
from itertools import combinations

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from twinmask.common import TimeEmb, save_json, set_seed
from twinmask.data import load


class SAGE(nn.Module):
    def __init__(self, width, output, conditional=False, *, hidden=128, dropout=.2):
        super().__init__()
        self.first = SAGEConv(width, hidden)
        self.last = SAGEConv(hidden, output)
        self.condition = nn.Sequential(TimeEmb(hidden)) if conditional else None
        self.edge_condition = nn.Linear(1, hidden) if conditional else None
        self.dropout = dropout

    def forward(self, x, edges, t=None, rate=0):
        h = self.first(x, edges)
        if self.condition is not None:
            h = h + self.condition(t / 100) + self.edge_condition(h.new_tensor([[rate]]))
        return self.last(F.dropout(F.gelu(h), self.dropout, self.training), edges)


def schedule(device):
    grid = torch.arange(101, device=device, dtype=torch.float64) / 100
    alpha = ((grid + .008) / 1.008 * torch.pi / 2).cos().square()
    return (alpha / alpha[0]).clamp(min=1e-5).float()


def noise_view(x, alpha, noise):
    return alpha.sqrt() * x + (1 - alpha).sqrt() * noise


def tick(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def generate(model, x, pairs, alpha, start, steps, rate, seed):
    rng = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, device=x.device, generator=rng)
    kept = pairs[:, torch.rand(pairs.shape[1], device=x.device, generator=rng) >= rate]
    edges = torch.cat((kept, kept.flip(0)), dim=1)
    grid = torch.linspace(start, 0, steps + 1).round().long().tolist()
    state = noise_view(x, alpha[start], noise)
    for current, following in zip(grid, grid[1:]):
        eps = model(state, edges, x.new_tensor(current), rate)
        estimate = (state - (1 - alpha[current]).sqrt() * eps) / alpha[current].sqrt()
        state = noise_view(estimate, alpha[following], eps)
    return state, noise


def metrics(probs, y):
    confidence, pred = probs.max(1)
    correct = pred == y
    ece = sum((confidence[ids].mean() - correct[ids].float().mean()).abs() * ids.float().mean()
              for ids in ((confidence * 15).long().clamp(max=14) == k for k in range(15)) if ids.any())
    return dict(accuracy=correct.float().mean().item(),
                macro_f1=f1_score(y.cpu(), pred.cpu(), average="macro"),
                nll=F.nll_loss(probs.clamp_min(1e-9).log(), y).item(),
                brier=(probs - F.one_hot(y, probs.shape[1])).square().sum(1).mean().item(),
                ece=float(ece))


def diversity(probs, y):
    labels = probs.argmax(-1)
    errors = labels != y
    disagreements, double_faults, correlations = [], [], []
    for i, j in combinations(range(len(probs)), 2):
        disagreements.append((labels[i] != labels[j]).float().mean())
        double_faults.append((errors[i] & errors[j]).float().mean())
        left, right = errors[i].float(), errors[j].float()
        if 0 < left.mean() < 1 and 0 < right.mean() < 1:
            correlations.append(F.cosine_similarity(left-left.mean(), right-right.mean(), dim=0))
    return dict(disagreement=float(torch.stack(disagreements).mean()),
                double_fault=float(torch.stack(double_faults).mean()),
                error_correlation=float(torch.stack(correlations).mean()) if correlations else None,
                oracle_accuracy=float((~errors).any(0).float().mean()))


def run(name, seed, epochs, device, out):
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data = load(name)
    x, y, pairs = (data[k].to(device) for k in ("emb", "y", "train_pos"))
    order = torch.randperm(len(x), device=device)
    tr, va, te = order[:int(.6*len(x))], order[int(.6*len(x)):int(.8*len(x))], order[int(.8*len(x)):]
    fit, select = va.chunk(2)
    x = (x - x[tr].mean(0)) / x[tr].std(0).clamp_min(1e-6)
    edges = torch.cat((pairs, pairs.flip(0)), 1)
    alpha = schedule(device)
    def fit_denoiser(fixed_time=None):
        start_time = tick(device)
        denoiser = SAGE(x.shape[1], x.shape[1], conditional=True).to(device)
        opt = torch.optim.AdamW(denoiser.parameters(), lr=.001, weight_decay=.0001)
        for _ in range(epochs):
            t = int(torch.randint(1, 101, ()).item()) if fixed_time is None else fixed_time
            rate = (0, .1, .3, .5)[torch.randint(4, ()).item()] if fixed_time is None else 0
            observed = pairs[:, torch.rand(pairs.shape[1], device=device) >= rate]
            eps = torch.randn_like(x)
            prediction = denoiser(noise_view(x, alpha[t], eps), torch.cat((observed, observed.flip(0)), 1),
                                  x.new_tensor(t), rate)
            target = eps if fixed_time is None else x
            loss = F.mse_loss(prediction[tr], target[tr])
            opt.zero_grad()
            loss.backward()
            opt.step()
        denoiser.eval()
        return denoiser, tick(device) - start_time

    model, denoiser_seconds = fit_denoiser()
    views, gaussian, fidelity, generation = [], [], [], []
    policies = [(10, 1, 0), (10, 4, .1), (30, 1, 0), (30, 4, .1), (50, 8, .3)]
    for k, (start, steps, rate) in enumerate(policies):
        start_time = tick(device)
        view, eps = generate(model, x, pairs, alpha, start, steps, rate, seed*100+k)
        generation.append(tick(device) - start_time)
        cosine = F.cosine_similarity(view, x).mean().item()
        low, high = 0., 32.
        for _ in range(16):
            scale = (low + high) / 2
            if F.cosine_similarity(x + scale*eps, x).mean() > cosine:
                low = scale
            else:
                high = scale
        gaussian.append(x + (low + high)/2*eps)
        views.append(view)
        normalized = F.normalize(x)
        retrieval = torch.cat([(F.normalize(chunk) @ normalized.T).argmax(1)
                               for chunk in view.split(256)])
        fidelity.append(dict(cosine=cosine, gaussian_cosine=F.cosine_similarity(gaussian[-1], x).mean().item(),
                             identity=(retrieval == torch.arange(len(x), device=device)).float().mean().item()))

    def fit_head(inputs, head_seed):
        set_seed(head_seed)
        head = SAGE(x.shape[1], int(y.max())+1).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.005)
        best = float("inf")
        for epoch in range(epochs):
            head.train()
            logits = head(inputs[epoch % len(inputs)], edges)
            optimizer.zero_grad()
            F.cross_entropy(logits[tr], y[tr]).backward()
            optimizer.step()
            head.eval()
            with torch.no_grad():
                probabilities = torch.stack([head(v, edges)[fit].softmax(1) for v in inputs]).mean(0)
                loss = F.nll_loss(probabilities.clamp_min(1e-9).log(), y[fit]).item()
                if loss < best:
                    best = loss
                    state = {key: value.clone() for key, value in head.state_dict().items()}
        head.load_state_dict(state)
        return head

    start_time = tick(device)
    raw = fit_head([x], seed)
    shared = fit_head([x, *views], seed)
    extra = [fit_head([x], seed+100+k) for k in range(4)]
    heads_seconds = tick(device) - start_time
    with torch.no_grad():
        raw_probs = raw(x, edges).softmax(1)
        samples = [shared(v, edges).softmax(1) for v in views]
        gaussian_probs = [shared(v, edges).softmax(1) for v in gaussian]
        raw.train()
        dropout = [raw(x, edges).softmax(1) for _ in views]
        candidates = [i for i, row in enumerate(fidelity) if row["cosine"] >= .8 and row["identity"] >= .9]
        chosen = []
        base = shared(x, edges).softmax(1)

        def validation(indices):
            p = torch.stack([base, *(samples[i] for i in indices)]).mean(0)
            return F.nll_loss(p[select].log(), y[select]).item()

        remaining_calls = 9
        while candidates:
            feasible = [i for i in candidates if policies[i][1] <= remaining_calls]
            if not feasible:
                break
            current = validation(chosen)
            best = max(feasible, key=lambda i: (current-validation([*chosen, i])) / policies[i][1])
            if validation([*chosen, best]) >= current:
                break
            chosen.append(best)
            candidates.remove(best)
            remaining_calls -= policies[best][1]
        arms = dict(raw=[raw_probs], shared_raw=[base], diffusion_1=samples[:1],
                    diffusion_2=samples[:2], diffusion_5=samples, gaussian_5=gaussian_probs,
                    dropout_5=dropout, head_ensemble_5=[raw_probs, *(h(x, edges).softmax(1) for h in extra)],
                    selected=[base, *(samples[i] for i in chosen)],
                    diffusion_raw_plus_4=[base, *samples[:4]],
                    mean_embedding=[shared(torch.stack(views).mean(0), edges).softmax(1)])
    independent, independent_seconds, independent_views = [base], [], []
    for k, (start, steps, rate) in enumerate(policies[:4]):
        set_seed(seed + 1000 + k)
        fitted, seconds = fit_denoiser()
        independent_seconds.append(seconds)
        view, _ = generate(fitted, x, pairs, alpha, start, steps, rate, seed*100+k)
        independent_views.append(view)
        with torch.no_grad():
            independent.append(shared(view, edges).softmax(1))
    set_seed(seed + 2000)
    autoencoder, autoencoder_seconds = fit_denoiser(fixed_time=30)
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(seed*100+2)
        eps = torch.randn(x.shape, device=device, generator=g)
        view = autoencoder(noise_view(x, alpha[30], eps), edges, x.new_tensor(30), 0)
        arms["fixed_noise_autoencoder"] = [shared(view, edges).softmax(1)]
        arms["diffusion_one_at_30"] = samples[2:3]
        arms["independent_raw_plus_4"] = independent
    representation_arms = dict(diffusion_5=views, gaussian_5=gaussian,
                               diffusion_raw_plus_4=[x, *views[:4]],
                               independent_raw_plus_4=[x, *independent_views],
                               fixed_noise_autoencoder=[view], diffusion_one_at_30=views[2:3])
    control_start = tick(device)
    raw.eval()
    for key, inputs in representation_arms.items():
        method = fit_head(inputs, seed)
        with torch.no_grad():
            arms[f"raw_head_{key}"] = [raw(v, edges).softmax(1) for v in inputs]
            arms[f"matched_head_{key}"] = [method(v, edges).softmax(1) for v in inputs]
    control_seconds = tick(device)-control_start
    with torch.no_grad():
        members = {key: torch.stack(values)[:, te] for key, values in arms.items()}
        results = {key: metrics(values.mean(0), y[te]) for key, values in members.items()}
        diversity_results = {key: diversity(values, y[te]) for key, values in members.items() if len(values) > 1}
        predictions = {key: values.mean(0).cpu() for key, values in members.items()}
    directory = Path("checkpoints") / out
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(denoiser=model.state_dict(), train=tr.cpu(), fit=fit.cpu(), select=select.cpu(),
                    test=te.cpu(), labels=y[te].cpu(), predictions=predictions,
                    members={key: value.cpu() for key, value in members.items()}), directory / f"{name}_{seed}.pt")
    return dict(seed=seed, epochs=epochs, metrics=results, diversity=diversity_results, policies=policies, fidelity=fidelity,
                selection=chosen, denoisers=1, selected_calls=9-remaining_calls,
                denoiser_seconds=denoiser_seconds, generation_seconds=generation, heads_seconds=heads_seconds,
                matched_head_controls_seconds=control_seconds,
                independent_denoiser_seconds=independent_seconds, autoencoder_seconds=autoencoder_seconds,
                peak_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in args.datasets:
        rows = []
        for seed in args.seeds:
            rows.append(run(name, seed, args.epochs, torch.device(args.device), "research_ensemble_v2"))
            save_json(rows, "research_ensemble_v2", f"{name}.json")
            print(name, seed, {k: round(v["accuracy"], 4) for k, v in rows[-1]["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
