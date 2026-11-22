"""Build the ref_selected augmented graph and frozen 5 LoRA + 8 diffusion views."""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
from torch_geometric.utils import to_undirected
from twinmask.data import ROOT


DATASETS = ("cora", "citeseer")


def prepare_augmented(name, diverge, cache, device="cuda", epochs=1000):
    diverge = diverge.resolve()
    selection = diverge / "tags_data_augmentation/semi_study/results"
    for path in (selection / "edge_selection.json", selection / "mode_selection.json",
                 diverge / "datasets" / f"{name}.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"DIVERGE input missing: {path}")
    sys.path.insert(0, str(diverge))
    from scripts import semi_aug_bridge as bridge
    from tags_data_augmentation.semi_study import semi_task

    llm = "llama_3.2_1B"
    for init in semi_task.INITS:
        for path in (semi_task.base_cache_path(name, llm, init),
                     semi_task.aug_cache_path(name, llm, init, "curated_raw")):
            if not Path(path).is_file():
                raise FileNotFoundError(f"DIVERGE LoRA cache missing: {path}")
    print(f"{name}: loading DIVERGE ref_selected graph and five LoRA views", flush=True)
    task = bridge.make_task(name, "augFull", 0, selection_dir=selection)
    if task.inits != list(semi_task.INITS) or len(task.inits) != 5:
        raise RuntimeError(f"Expected five LoRA views, found {task.inits}")
    edge = to_undirected(task.und, num_nodes=task.n).to(device)

    implementation = [Path(bridge.__file__), Path(bridge.phase0.__file__),
                      Path(ROOT, "twinmask", "common.py")]
    device_name = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    digest = hashlib.sha256(f"{epochs}/{torch.__version__}/{device_name}".encode())
    for path in implementation:
        digest.update(path.read_bytes())
    for tensor in (task.und, task.y, *task.masks, *(task.embs[i] for i in task.inits)):
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
    signature = digest.hexdigest()
    folder = cache / name / signature
    folder.mkdir(parents=True, exist_ok=True)
    views = {f"lora_{init}": task.embs[init] for init in task.inits}
    for init in task.inits:
        for kind in bridge.VARIANTS if init == "pissa" else ("joint",):
            key = f"diff_{init}_{kind}"
            path = folder / f"{key}_s0.pt"
            if not path.exists():
                print(f"{name}: training {key} ({epochs} epochs on {device})", flush=True)
                features = bridge.diffusion_view(task, task.embs[init], edge, kind, 0,
                                                 (256, epochs, 1.0, 0.1), device=device)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(features.cpu(), tmp)
                tmp.replace(path)
            views[key] = torch.load(path, map_location="cpu", weights_only=True)
    return task, views, signature


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--diverge", type=Path, default=Path(ROOT).parent / "DIVERGE")
    parser.add_argument("--cache", type=Path, default=Path(ROOT, "checkpoints/cf_dive_ref_selected"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    args = arguments().parse_args()
    for ds in args.datasets:
        prepare_augmented(ds, args.diverge, args.cache, args.device)
