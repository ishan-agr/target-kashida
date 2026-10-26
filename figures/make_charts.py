"""Regenerate the matplotlib charts used in TECHNICAL_WRITEUP.md.

    python figures/make_charts.py      -> figures/*.svg

Every value is either computed from a formula (score granularity) or copied from
the project's validated benchmark notes; nothing is simulated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
C1, C2, C3, C4, C5 = "#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "svg.fonttype": "path",  # embed glyphs; no font dependency in the PDF renderer
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. VRAM budget
def vram_budget() -> None:
    parts = [("AWQ weights (LLM)", 5.0, C1), ("Vision tower FP16", 0.8, "#60a5fa"),
             ("KV-cache pool", 10.7, C3), ("CUDA ctx / activations", 1.5, MUTED),
             ("Sidecar headroom", 3.5, C2), ("Safety margin", 2.5, "#cbd2d9")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    left = 0.0
    for label, gb, c in parts:
        ax.barh([0], [gb], left=left, color=c, edgecolor="white", height=0.55)
        if gb >= 1.4:
            ax.text(left + gb / 2, 0, f"{gb:g} GB", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
        left += gb
    ax.axvline(24, color=INK, lw=1.2, ls="--")
    ax.text(23.8, -0.42, "A10 = 24 GB", ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlim(0, 25)
    ax.set_yticks([])
    ax.set_xlabel("GPU memory (GB)")
    ax.grid(axis="y", visible=False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], [p[0] for p in parts],
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8.5)
    ax.set_title("Where 24 GB goes: the frozen ~18 GB base leaves ~3–3.5 GB for everything else")
    save(fig, "fig_vram_budget")


# --------------------------------------------- 5. data mix + catastrophic forgetting
def forgetting() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1, 1.7]})
    runs = ["V14", "V15", "V16 (plan)"]
    cheque = np.array([5.6, 100.0, 75.0])
    a1.barh(runs, cheque, color=C1, label="in-domain cheque")
    a1.barh(runs, 100 - cheque, left=cheque, color=C2, label="generic / handwritten Arabic OCR")
    for i, v in enumerate(cheque):
        a1.text(min(v, 92) + 2 if v < 50 else v / 2, i, f"{v:g}%", va="center",
                color="white" if v >= 50 else INK, fontsize=9, fontweight="bold")
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel("share of training samples (%)")
    a1.set_title("Training mix", fontsize=10)
    a1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    a1.grid(axis="y", visible=False)

    fields = ["bank", "drawer", "IBAN", "amount", "payee"]
    base = np.array([97.6, 92.8, 79.0, 89.2, 66.3])
    v14 = np.array([27.7, 27.7, 14.5, 53.0, 72.3]) - base
    v15 = np.array([90.4, 89.2, 67.5, 84.3, 62.7]) - base
    xi = np.arange(len(fields))
    a2.bar(xi - 0.2, v14, 0.38, color=C2, label="V14 (Arabic-heavy)")
    a2.bar(xi + 0.2, v15, 0.38, color=C1, label="V15 (cheque-only)")
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xticks(xi, fields)
    a2.set_ylabel("Δ vs untouched base (pp)")
    a2.set_title("Change per field after merge + re-quantize", fontsize=10)
    a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle("The mix is a hyperparameter: V14 forgot the form, V15 forgot how to read",
                 fontsize=11, fontweight="bold", color=INK)
    save(fig, "fig_forgetting")


# ------------------------------------------------- 7. projections vs measurements
def projections() -> None:
    items = ["Image preprocessing", "Multi-crop voting", "Printed-OCR sidecar"]
    lo, hi = np.array([5, 15, 10]), np.array([15, 25, 20])
    actual = np.array([-12.5, 0.0, -12.5])
    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.barh(y, hi - lo, left=lo, color="#bfdbfe", height=0.5, label="projected range (literature-based)")
    ax.plot(actual, y, "D", color=C4, ms=8, label="measured")
    for yi, a in zip(y, actual):
        ax.text(a - 1.2, yi, f"{a:+.1f}", va="center", ha="right", color=C4, fontsize=9)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, items)
    ax.invert_yaxis()
    ax.set_xlim(-20, 30)
    ax.set_xlabel("effect on handwritten-payee reading (pp)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_title("Speculative gains don't stack, and these were never gains")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_projections")


# ---------------------------------------------------- 8. small-N granularity
def granularity() -> None:
    n = np.arange(5, 201)
    se = 100 * np.sqrt(0.5 * 0.5 / n)  # worst-case binomial standard error at p = 0.5
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.plot(n, 100 / n, color=C1, lw=2, label="resolution: one item flips = 100/N pp")
    ax.plot(n, 1.96 * se, color=C2, lw=2, ls="--", label="95% CI half-width at p = 0.5")
    for k, lbl in [(8, "8 field-bearing\ncheques"), (18, "first set\n(18)"), (31, "Arabic\npayees (31)"),
                   (83, "full set\n(83)")]:
        ax.plot([k], [100 / k], "o", color=INK, ms=4)
        ax.annotate(lbl, (k, 100 / k), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
    ax.set_ylim(0, 30)
    ax.set_xlabel("evaluation items N (log scale)")
    ax.set_ylabel("percentage points")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Compute your granularity before you interpret a delta")
    save(fig, "fig_granularity")


# ------------------------------------------------------ 9. failure taxonomy
def failure_taxonomy() -> None:
    rows = [("Completely wrong read", "fluent, plausible name that is not on the cheque", 64, C4,
             "rules out post-OCR correction, voting, prompt optimisation"),
            ("Character-level corruption", "right name, dotted letters swapped", 22, C2,
             "addressable by fine-tuning the reader on handwritten pixels"),
            ("Template text read as payee", "printed 'pay to the order of' phrase", 14, C1,
             "addressable by crop cleanup, template filters, DPO negatives")]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for i, (name, desc, pct, c, fix) in enumerate(rows):
        ax.barh(i, pct, color=c, height=0.55)
        ax.text(pct + 1.5, i - 0.08, f"{pct}%", va="center", fontweight="bold", color=INK, fontsize=10)
        ax.text(pct + 1.5, i + 0.22, fix, va="center", fontsize=8, color=MUTED, style="italic")
    ax.set_yticks(range(3), [f"{n}\n({d})" for n, d, *_ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("share of handwritten-Arabic payee misses (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Classify failures before choosing techniques")
    save(fig, "fig_failure_taxonomy")


if __name__ == "__main__":
    for f in (vram_budget, forgetting, projections, granularity, failure_taxonomy):
        f()
        print("ok", f.__name__)
