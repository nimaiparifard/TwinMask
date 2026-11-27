"""Frozen contextual DistilRoBERTa comparison with cached masked-input encodings."""
import argparse
from hashlib import sha256
from pathlib import Path

import torch
from torch import nn

from scripts.research_twinmask import PAD
from scripts.research_twinmask_v5 import IsolatedTwin
from scripts.research_twinmask_v6 import (Config, LOCKED, SEARCH_SPLIT, locked_protocol, measure,
                                          protocol, stage_cases, train)
from twinmask.common import save_json


class FrozenText:
    def __init__(self, device):
        from transformers import AutoModel

        self.device = device
        path = Path("hf_models/all-distilroberta-v1")
        self.path = path
        self.model = AutoModel.from_pretrained(path, local_files_only=True).requires_grad_(False).eval().to(device)
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.model.bfloat16()
        self.cache = {}

    @torch.no_grad()
    def get(self, ids, vocabulary):
        tokens = vocabulary[ids]
        key = sha256(tokens.cpu().contiguous().numpy().tobytes()).digest()
        if key not in self.cache:
            chunks = []
            for block in tokens.split(64):
                result = self.model(input_ids=block, attention_mask=block != PAD).last_hidden_state
                chunks.append(result.cpu().to(torch.float16))
            self.cache[key] = torch.cat(chunks)
        return self.cache[key].to(ids.device, dtype=torch.float32)


class PretrainedTwin(IsolatedTwin):
    def __init__(self, frozen, vocabulary, vocab, config, family):
        super().__init__(vocab, config.width, family, fusion=config.fusion)
        self.frozen = frozen
        self.pretrained_projection = nn.Linear(frozen.model.config.hidden_size, config.width)
        self.register_buffer("lexical_ids", torch.cat((vocabulary.new_tensor([50264, 1, 3]), vocabulary)))

    def encode_text(self, ids):
        return self.pretrained_projection(self.frozen.get(ids, self.lexical_ids))


def run(device):
    frozen = FrozenText(device)

    def factory(make, vocab, config, family):
        return PretrainedTwin(frozen, make.vocabulary, vocab, config, family).to(device)

    shared = protocol("cora", SEARCH_SPLIT)
    outcomes = {}
    for family in ("mlm", "sage"):
        trials = []
        for lr, fusion in ((.001, .5), (.003, .5), (.001, 1.), (.003, 1.)):
            config = Config(lr=lr, fusion=fusion)
            _, _, trace = train(shared, SEARCH_SPLIT, 71, family, config, device, 64, model_factory=factory, mask_bank=4)
            trials.append(dict(config=vars(config), steps=trace["best_step"], history=trace,
                               dev=min(v["score"] for v in trace["curve"])))
        best = min(trials, key=lambda row: row["dev"])
        locked = locked_protocol(shared, LOCKED[0])
        model, (make, nodes, _), trace = train(locked, LOCKED[0], 97, family, Config(**best["config"]),
                                              device, best["steps"], fixed_steps=True,
                                              model_factory=factory, mask_bank=4)
        outcomes[family] = dict(winner=best, trials=trials, test=measure(model, stage_cases(make, nodes, LOCKED[0], device, 2)),
                                test_history=trace)
        print("frozen DistilRoBERTa", family, flush=True)
    return dict(model=str(frozen.path), cached_masked_inputs=len(frozen.cache), arms=outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    save_json(run(torch.device(args.device)), "research_twinmask_v6", "cora_pretrained.json")


if __name__ == "__main__":
    main()
