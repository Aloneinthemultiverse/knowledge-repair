"""Correct agent confidence before the gate trusts it.

Measured on 80 outcome-labelled conflicts, agent confidence does not merely fail
to predict correctness — it INVERTS at the top:

    stated 0.00-0.50  ->  actually right 56%
    stated 0.50-0.70  ->  actually right 58%
    stated 0.70-0.85  ->  actually right 70%
    stated 0.85-0.95  ->  actually right 46%
    stated 0.95-1.00  ->  actually right 37%      <- most confident, most wrong

Thirty-four percent of claims landed in that last band. So the confidence floor
this project already ships is filtering in the wrong direction: QUARANTINED is
applied to low-confidence claims, which are the ones more likely to be right.
That is a live defect in shipped behaviour, independent of any model.

HISTOGRAM BINNING, not isotonic and not Platt. Both of those assume the mapping
is monotone, and this one is not — it rises to 0.70 and then falls to 0.37.
Fitting isotonic here collapsed every bin into a single constant 0.50 (the best
non-decreasing approximation of a hump is a flat line) and took the AUC from
0.584 down to 0.500. The monotone assumption did not smooth the signal, it
deleted it. Plain per-bin observed accuracy keeps the inversion intact.

WHAT THIS IS NOT: a fix for the model. It corrects one model's confidence on one
task, and it must be refitted whenever either changes. `fit()` records the model
id and sample count so a stale calibrator can be detected rather than trusted.

    cal = Calibrator.load()
    true_p = cal.correct(0.95)     # -> ~0.4, not 0.95
"""
import json
import os
from bisect import bisect_right

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration.json")

# Below this many samples the mapping is noise dressed as a correction. Refusing
# is better than shipping a curve fitted on a handful of conflicts.
MIN_SAMPLES = 40


class Calibrator:
    """Maps stated confidence to observed accuracy."""

    def __init__(self, xs=None, ys=None, meta=None):
        self.xs = xs or []          # bin upper edges, ascending
        self.ys = ys or []          # observed accuracy per bin
        self.meta = meta or {}

    # ── fitting ──────────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, rows, conf_key="confidence", label_key="won", n_bins=8,
            model_id=None):
        """Fit from outcome-labelled rows: [{confidence, won}, ...]."""
        rows = [r for r in rows if r.get(conf_key) is not None]
        if len(rows) < MIN_SAMPLES:
            raise ValueError(
                f"{len(rows)} samples is below MIN_SAMPLES={MIN_SAMPLES}; "
                "a mapping fitted here would be noise")

        rows = sorted(rows, key=lambda r: r[conf_key])
        step = max(1, len(rows) // n_bins)
        binned = []
        for i in range(0, len(rows), step):
            chunk = rows[i:i + step]
            if not chunk:
                continue
            binned.append((max(r[conf_key] for r in chunk),
                           sum(float(bool(r[label_key])) for r in chunk) / len(chunk)))

        xs = [b[0] for b in binned]
        ys = [round(b[1], 4) for b in binned]
        meta = {"n": len(rows), "model_id": model_id, "n_bins": len(binned),
                "method": "histogram"}
        return cls(xs, ys, meta)

    # ── use ──────────────────────────────────────────────────────────────────
    def correct(self, confidence: float) -> float:
        """Stated confidence -> probability of actually being right.

        Uncalibrated input passes through unchanged when no fit is loaded, so an
        install without calibration behaves exactly as before rather than
        silently applying a default curve.
        """
        if not self.xs:
            return float(confidence)
        i = bisect_right(self.xs, float(confidence))
        return float(self.ys[min(i, len(self.ys) - 1)])

    def is_stale(self, model_id: str) -> bool:
        """A calibrator fitted on another model is worse than none: it applies a
        correction learned from behaviour this model does not have."""
        fitted = self.meta.get("model_id")
        return bool(fitted and model_id and fitted != model_id)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path=DEFAULT_PATH):
        json.dump({"xs": self.xs, "ys": self.ys, "meta": self.meta},
                  open(path, "w"), indent=1)
        return path

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        if not os.path.exists(path):
            return cls()                 # identity mapping
        d = json.load(open(path))
        return cls(d.get("xs"), d.get("ys"), d.get("meta"))

    def report(self) -> str:
        if not self.xs:
            return "no calibration fitted — confidence passes through unchanged"
        lines = [f"fitted on n={self.meta.get('n')} "
                 f"model={self.meta.get('model_id') or 'unspecified'}"]
        for x, y in zip(self.xs, self.ys):
            lines.append(f"  stated <= {x:.2f}  ->  actual {y:.2f}")
        return "\n".join(lines)


def calibrated_confidence(raw: float, model_id: str = None) -> float:
    """Convenience used by the gate. Never raises: a calibration problem must not
    stop a claim being recorded, only stop it being trusted more than it earns."""
    try:
        cal = Calibrator.load()
        if model_id and cal.is_stale(model_id):
            return float(raw)
        return cal.correct(raw)
    except Exception:
        return float(raw)
