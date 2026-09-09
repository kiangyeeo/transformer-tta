"""Faithful standalone port of the official NCTTA adaptation objective."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class NCTTAObjectiveError(RuntimeError):
    """An invalid official-objective state with machine-readable diagnostics."""

    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"{message}: {diagnostics}")


@dataclass(frozen=True)
class NCTTALossResult:
    loss: torch.Tensor
    diagnostics: dict


def softmax_entropy(logits):
    """Per-sample entropy used by TTAB's adaptation utility."""

    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def compute_nc_loss(
    weight,
    features,
    y_hat,
    top_k=3,
    type="infonce",
    metric="cos",
    tau_align=0.07,
    margin=0.2,
    mix_prob_weight=0.0,
    y_hat_is_logits=False,
    reduce="mean",
    safe_eps=1.0e-8,
):
    """Port of official ``compute_nc_loss`` without importing TTAB."""

    if weight.dim() != 2 or features.dim() != 2 or y_hat.dim() != 2:
        raise ValueError("weight, features, and y_hat must all be rank two")
    batch_size, feature_dim = features.shape
    class_count, weight_dim = weight.shape
    if weight_dim != feature_dim or y_hat.shape != (batch_size, class_count):
        raise ValueError(
            "NCTTA geometry mismatch: "
            f"features={tuple(features.shape)}, weight={tuple(weight.shape)}, "
            f"y_hat={tuple(y_hat.shape)}"
        )

    w_hat = F.normalize(weight, p=2, dim=1)
    h_hat = F.normalize(features, p=2, dim=1)
    if y_hat_is_logits:
        probs = F.softmax(y_hat, dim=1)
    else:
        probs = (y_hat + safe_eps).clamp_min(safe_eps)
        probs = probs / probs.sum(dim=1, keepdim=True)

    sims = h_hat @ w_hat.t()
    if metric == "cos":
        dists = 1.0 - sims.clamp(-1.0, 1.0)
    elif metric == "euclid":
        dists = (2.0 - 2.0 * sims.clamp(-1.0, 1.0)).clamp_min(0.0).sqrt()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    topk_prob, topk_idx = torch.topk(
        probs, k=min(int(top_k), class_count), dim=1, largest=True
    )
    topk_dists = torch.gather(dists, dim=1, index=topk_idx)
    topk_sims = torch.gather(sims, dim=1, index=topk_idx)

    d_mean = topk_dists.mean(dim=1, keepdim=True)
    d_std = topk_dists.std(dim=1, keepdim=True, unbiased=False) + safe_eps
    norm_d = (topk_dists - d_mean) / d_std
    q_dist = F.softmax(-norm_d, dim=1)
    q_prob = F.softmax(torch.log(topk_prob + safe_eps), dim=1)
    alpha = float(mix_prob_weight)
    if alpha <= 0:
        q = q_dist
    elif alpha >= 1:
        q = q_prob
    else:
        q = (1 - alpha) * q_dist + alpha * q_prob
        q = q + safe_eps
        q = q / q.sum(dim=1, keepdim=True)

    sims_sorted, _ = sims.sort(dim=1, descending=True)
    primary_margin = sims_sorted[:, 0] - sims_sorted[:, 1].clamp_max(1.0e9)
    if type == "infonce":
        exp_logits = torch.exp(sims / max(float(tau_align), safe_eps))
        exp_probs_topk = torch.gather(exp_logits, dim=1, index=topk_idx)
        loss_per = -torch.log(
            (q * exp_probs_topk).sum(dim=1) / exp_logits.sum(dim=1)
        )
    elif type == "cosine_l2":
        w_soft = torch.einsum("bk,bkd->bd", q, w_hat[topk_idx])
        w_soft = F.normalize(w_soft, dim=1)
        diff = h_hat - w_soft
        loss_per = (diff * diff).sum(dim=1)
    elif type == "margin_triplet":
        w_pos = torch.einsum("bk,bkd->bd", q, w_hat[topk_idx])
        w_pos = F.normalize(w_pos, dim=1)
        sim_pos = (h_hat * w_pos).sum(dim=1, keepdim=True)
        mask_all = torch.ones_like(sims, dtype=torch.bool)
        mask_topk = torch.zeros_like(mask_all)
        mask_topk.scatter_(
            dim=1,
            index=topk_idx,
            src=torch.ones_like(topk_idx, dtype=torch.bool),
        )
        sims_neg = sims.masked_fill(~(mask_all & (~mask_topk)), -1.0e9)
        sim_neg_hard, _ = sims_neg.max(dim=1, keepdim=True)
        loss_per = F.relu(margin - (sim_pos - sim_neg_hard)).squeeze(1)
    elif type == "hinge_cos":
        s_pos = (q * topk_sims).sum(dim=1, keepdim=True)
        mask_all = torch.ones_like(sims, dtype=torch.bool)
        mask_topk = torch.zeros_like(mask_all)
        mask_topk.scatter_(
            dim=1,
            index=topk_idx,
            src=torch.ones_like(topk_idx, dtype=torch.bool),
        )
        sims_neg = sims.masked_fill(~(mask_all & (~mask_topk)), -1.0e9)
        s_neg, _ = sims_neg.max(dim=1, keepdim=True)
        loss_per = F.relu(margin - (s_pos - s_neg)).squeeze(1)
    else:
        raise ValueError(f"Unknown type: {type}")

    if reduce == "mean":
        loss = loss_per.mean()
    elif reduce == "sum":
        loss = loss_per.sum()
    elif reduce == "none":
        loss = loss_per
    else:
        raise ValueError(f"Unknown reduce: {reduce}")
    return loss, {
        "q": q.detach(),
        "q_dist": q_dist.detach(),
        "q_prob": q_prob.detach(),
        "topk_idx": topk_idx.detach(),
        "topk_dists": topk_dists.detach(),
        "topk_sims": topk_sims.detach(),
        "sims": sims.detach(),
        "dists": dists.detach(),
        "primary_margin": primary_margin.detach(),
    }


def effective_classifier_weight(net_c, feature_dim):
    """Read the effective frozen classifier weight after normal forward hooks."""

    classifier = getattr(net_c, "fc", None)
    if classifier is None or not hasattr(classifier, "weight"):
        raise RuntimeError("NCTTA requires netC.fc.weight")
    weight = classifier.weight.detach()
    if weight.dim() != 2 or int(weight.shape[1]) != int(feature_dim):
        raise RuntimeError(
            "NCTTA requires post-netB feature dim == classifier dim; "
            f"got feature_dim={feature_dim}, weight={tuple(weight.shape)}"
        )
    return weight


def nctta_loss_from_outputs(features, logits, classifier_weight, config):
    """Official NCTTA entropy/filter/reweight path for explicit F/B/C output."""

    params = config["nctta"] if "nctta" in config else config
    h_normalized = F.normalize(features, p=2, dim=1)
    w_normalized = F.normalize(classifier_weight, p=2, dim=1)
    predicted_classes = torch.argmax(logits, dim=1)
    loss_ent = softmax_entropy(logits)
    loss_nc, aux = compute_nc_loss(
        weight=classifier_weight,
        features=features,
        y_hat=logits,
        top_k=params["top_k"],
        type="infonce",
        metric="cos",
        tau_align=1.0,
        margin=0.2,
        mix_prob_weight=params["mix_prob_weight"],
        y_hat_is_logits=True,
        reduce="none",
    )
    selected_weights = w_normalized[predicted_classes]
    distance = torch.norm(h_normalized - selected_weights, p=2, dim=1)
    selected = loss_ent < float(params["thre_ent"])
    selected_count = int(selected.sum().item())
    diagnostics = {
        "batch_size": int(features.shape[0]),
        "selected_count": selected_count,
        "filtered_count": int(features.shape[0]) - selected_count,
        "feature_dim": int(features.shape[1]),
        "classifier_dim": int(classifier_weight.shape[1]),
        "classifier_weight_detached": not classifier_weight.requires_grad,
        "effective_top_k": min(int(params["top_k"]), int(logits.shape[1])),
        "entropy_min": float(loss_ent.detach().min().item()),
        "entropy_max": float(loss_ent.detach().max().item()),
        "entropy_mean": float(loss_ent.detach().mean().item()),
        "nc_loss_mean": float(loss_nc.detach().mean().item()),
        "predicted_fca_distance_mean": float(distance.detach().mean().item()),
        "topk_index_checksum": int(aux["topk_idx"].sum().item()),
        "q_dist_l2": float(torch.linalg.vector_norm(aux["q_dist"]).item()),
        "q_prob_l2": float(torch.linalg.vector_norm(aux["q_prob"]).item()),
        "hybrid_q_min": float(aux["q"].min().item()),
        "hybrid_q_max": float(aux["q"].max().item()),
        "hybrid_q_l2": float(torch.linalg.vector_norm(aux["q"]).item()),
        "entropy_selected_index_checksum": int(
            torch.nonzero(selected, as_tuple=False).sum().item()
        ),
    }
    if selected_count == 0:
        raise NCTTAObjectiveError(
            "NCTTA entropy filter selected zero samples", diagnostics
        )

    coeff_ent = float(params["reweight_ent"]) * (
        1 / torch.exp(loss_ent.detach() - float(params["margin_ent"]))
    )
    coeff_exp = float(params["nu"]) / (
        1 + float(params["eta"]) * distance.detach()
    )
    coeff = coeff_ent + coeff_exp
    weighted = (loss_ent + float(params["scale"]) * loss_nc).mul(coeff)
    loss = weighted[selected].mean(0)
    diagnostics.update(
        {
            "entropy_coefficient_mean": float(coeff_ent.mean().item()),
            "fca_distance_coefficient_mean": float(coeff_exp.mean().item()),
            "combined_coefficient_mean": float(coeff.mean().item()),
            "loss": float(loss.detach().item()),
        }
    )
    if not bool(torch.isfinite(loss).item()):
        raise NCTTAObjectiveError("NCTTA loss is non-finite", diagnostics)
    diagnostics["topk_index_min"] = int(aux["topk_idx"].min().item())
    diagnostics["topk_index_max"] = int(aux["topk_idx"].max().item())
    return NCTTALossResult(loss=loss, diagnostics=diagnostics)


def nctta_loss(inputs, net_f, net_b, net_c, config):
    """Explicit ``netF -> netB -> netC`` NCTTA forward and objective."""

    backbone_features = net_f(inputs)
    features = net_b(backbone_features)
    logits = net_c(features)
    classifier_weight = effective_classifier_weight(net_c, features.shape[1])
    return nctta_loss_from_outputs(features, logits, classifier_weight, config)
