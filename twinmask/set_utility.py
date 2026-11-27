"""Target-conditioned subset utility and label-free budgeted selection."""

import torch
from torch import nn
from torch.nn import functional as F


def enumerate_subsets(valid):
    width = valid.shape[1]
    bits = torch.arange(2**width, device=valid.device)[:, None] >> torch.arange(width, device=valid.device)
    bits = bits.remainder(2).bool()[:, None, :]
    return (bits & valid).float(), (~bits | valid).all(-1)


def choose(scores, masks, allowed, budget, *, exact=False, cost=0.):
    counts = masks.sum(-1)
    feasible = counts == counts.amax(0).clamp(max=budget) if exact else counts <= budget
    return (scores - cost*counts).masked_fill(~(feasible & allowed), -torch.inf).argmax(0)


def shapley_values(loss, width):
    from math import comb

    scores = loss.new_zeros(loss.shape[1], width)
    for j in range(width):
        bit = 1 << j
        for subset in range(2**width):
            if subset & bit == 0:
                scores[:, j] += (loss[subset] - loss[subset | bit]) / (width*comb(width-1, subset.bit_count()))
    return scores


class SetUtility(nn.Module):
    def __init__(self, dimension, hidden=48, *, attention=False, dropout=0.):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2*hidden+1, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.dropout = dropout
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True) if attention else None

    def forward(self, features, masks, valid, *, sample_dropout=False):
        h = self.phi(features)
        if self.attention is not None:
            # A dummy valid position for empty panels keeps attention finite.
            visible = valid.clone()
            visible[:, 0] |= ~valid.any(1)
            h = h+self.attention(h, h, h, key_padding_mask=~visible, need_weights=False)[0]
        h = F.dropout(h, self.dropout, self.training or sample_dropout)
        selected = torch.einsum("sni,nih->snh", masks, h)
        full = (h * valid[..., None]).sum(1).expand_as(selected)
        values = self.rho(torch.cat((selected, full, masks.sum(-1, keepdim=True)), -1))
        mean, spread = values.unbind(-1)
        return mean, F.softplus(spread) + .01


def subset_objective(mean, spread, utility, masks, allowed, decision_weight=0., uncertainty=False):
    error = F.smooth_l1_loss(mean[allowed], utility[allowed])
    delta = utility - utility.roll(1, 0)
    pairs = allowed & allowed.roll(1, 0) & (delta != 0)
    if pairs.any():
        error = error + .15*F.softplus(-delta[pairs].sign()*(mean - mean.roll(1, 0))[pairs]).mean()
    if decision_weight:
        count = masks.sum(-1)
        for budget in (1, 2, 3):
            probabilities = (mean / .35).masked_fill(~(allowed & (count <= budget)), -torch.inf).softmax(0)
            error = error - decision_weight*(probabilities*utility).sum(0).mean()/3
    if uncertainty:
        residual = (mean.detach() - utility)[allowed]
        sigma = spread[allowed]
        error = error + .2*(sigma.log()+.5*(residual/sigma).square()).mean()
    return error
