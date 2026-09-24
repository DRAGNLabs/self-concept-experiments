"""Content-token residual-stream probes; train, validate, and freeze before eval."""
from contextlib import contextmanager
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

from selfconcept.soo.activations import get_decoder_layers
from .roles import ROLES, content_indices, render_probe


@contextmanager
def capture(model, layers, project=None):
    """Capture residual output (not attention o_proj); project on-device in generation.

    Generation consumes the prompt then one token at a time. Its final sampled
    token is unprocessed; callers explicitly exclude it from token projections.
    """
    values = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        def hook(module, inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().float()[0]
            if project is not None:
                weight = torch.as_tensor(project[layer]["weight"], device=hidden.device)
                bias = torch.as_tensor(project[layer]["bias"], device=hidden.device)
                hidden = torch.softmax(hidden @ weight.T + bias, dim=-1)[:, ROLES.index("cot")]
            values[layer].append(hidden.cpu())
        handles.append(get_decoder_layers(model)[layer].register_forward_hook(hook))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def probabilities(x, probe):
    logits = x @ probe["weight"].T + probe["bias"]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_linear(x, y, c=0.005):
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=1729)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf.fit(scaler.transform(x), y)
    assert list(clf.classes_) == list(range(len(ROLES)))
    weight = clf.coef_ / scaler.scale_[None, :]
    return {"weight": weight.astype(np.float32),
            "bias": (clf.intercept_ - weight @ scaler.mean_).astype(np.float32)}


def metrics(x, y, probe):
    p = probabilities(x, probe)
    return {"accuracy": float(accuracy_score(y, p.argmax(1))),
            "cot_auc": float(roc_auc_score(y == ROLES.index("cot"), p[:, ROLES.index("cot")])),
            "log_loss": float(log_loss(y, p)),
            "confusion_matrix": confusion_matrix(y, p.argmax(1), labels=range(len(ROLES))).tolist(),
            "n_tokens": len(y)}


def train(model, tokenizer, family, corpus, out: Path, layers, seed=1729):
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    if len({r["id"] for r in rows}) != len(rows) or len({r["article"] for r in rows}) != len(rows):
        raise ValueError("Probe documents/articles must be unique and split-disjoint")
    xs = {s: {l: [] for l in layers} for s in ("train", "validation", "test")}
    ys = {s: [] for s in xs}
    positions = {s: [] for s in xs}
    device = model.get_input_embeddings().weight.device
    for row in rows:
        text = tokenizer.decode(tokenizer(row["text"], add_special_tokens=False)["input_ids"][:192])
        # Vary preceding context independently of role, shared by all copies.
        context = "A neutral document follows. " * (1 + int(row["id"][-2:], 16) % 12)
        rendered = [render_probe(tokenizer, family, text, role, context) for role in ROLES]
        encoded = [tokenizer(t, add_special_tokens=False, return_offsets_mapping=True) for t, _ in rendered]
        # Match by content-relative offsets, so every class sees identical tokens.
        maps = []
        for enc, (_, (start, end)) in zip(encoded, rendered):
            indices = content_indices(enc["offset_mapping"], [(start, end)], tokenizer.all_special_ids, enc["input_ids"])
            maps.append({(enc["offset_mapping"][i][0]-start, enc["offset_mapping"][i][1]-start, enc["input_ids"][i]): i for i in indices})
        common = sorted(set(maps[0]).intersection(*[set(m) for m in maps[1:]]))[2:-2]
        if len(common) < 16:
            raise ValueError(f"Insufficient matched content tokens: {row['id']}")
        chosen = [common[i] for i in np.linspace(0, len(common)-1, min(48, len(common)), dtype=int)]
        split = row["split"]
        for label, (enc, mapping) in enumerate(zip(encoded, maps)):
            indices = [mapping[k] for k in chosen]
            inputs = torch.tensor([enc["input_ids"]], device=device)
            with torch.inference_mode(), capture(model, layers) as values:
                model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1)
            for layer in layers:
                xs[split][layer].append(values[layer][0][indices].numpy())
            ys[split].extend([label] * len(indices))
            positions[split].extend([[i, k[0]] for i, k in zip(indices, chosen)])
        print(f"probe {row['id']} {split}", flush=True)
    ys = {s: np.array(v) for s, v in ys.items()}
    xs = {s: {l: np.concatenate(v) for l, v in ls.items()} for s, ls in xs.items()}
    result, report = {}, {"roles": ROLES, "seed": seed, "site": "post_decoder_layer_residual", "layers": {}}
    for layer in layers:
        fitted = fit_linear(xs["train"][layer], ys["train"])
        result[layer] = fitted
        rng = np.random.default_rng(seed)
        shuffled = fit_linear(xs["train"][layer], rng.permutation(ys["train"]))
        report["layers"][str(layer)] = {s: metrics(xs[s][layer], ys[s], fitted) for s in ("validation", "test")}
        report["layers"][str(layer)]["shuffled_test"] = metrics(xs["test"][layer], ys["test"], shuffled)
        np.savez(out / f"layer-{layer}.npz", **fitted)
    positional = fit_linear(np.asarray(positions["train"]), ys["train"])
    report["position_only_test"] = metrics(np.asarray(positions["test"]), ys["test"], positional)
    primary = layers[1]
    valid = report["layers"][str(primary)]["validation"]
    report["primary_layer"] = primary
    report["usable"] = valid["accuracy"] >= .8 and valid["cot_auc"] >= .9
    report["note"] = "Fixed midpoint; no selection on deception labels. WikiText adaptation, three-role softmax; probabilities are model-specific."
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    return result, report


def load(out):
    report = json.loads((out / "validation.json").read_text())
    return {int(layer): dict(np.load(out / f"layer-{layer}.npz")) for layer in report["layers"]}, report
