import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

## Values from experiment

index_sizes = {
    "Mice (Plaid 4bits)": {
        "splade": 9868.8,
        "plaid": 90242.8,
    },
    "LateOn (Plaid 2bits)": {
        "splade": 0,
        "plaid": 51768.945,
    }
}

FIG_SIZE = (3.6, 7)
FONT_SIZE = 14

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": [""],
        "font.size": FONT_SIZE,
        "axes.labelsize": 11,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": FONT_SIZE - 3,
    }
)
# plt.style.use("ggplot")

models = list(index_sizes)
mb_to_gb = 1_000
splade_sizes = [index_sizes[model]["splade"] / mb_to_gb for model in models]
plaid_sizes = [index_sizes[model]["plaid"] / mb_to_gb for model in models]
totals = [splade + plaid for splade, plaid in zip(splade_sizes, plaid_sizes)]

fig, ax = plt.subplots(figsize=FIG_SIZE)
model_positions = range(len(models))
ax.bar(
    model_positions,
    plaid_sizes,
    width=0.55,
    color="#e76f51",
    label="PLAID",
)
ax.bar(
    model_positions,
    splade_sizes,
    bottom=plaid_sizes,
    width=0.55,
    color="#2a9d8f",
    label="SPLADE",
)

for x, splade, plaid, total in zip(model_positions, splade_sizes, plaid_sizes, totals):
    if plaid > 0:
        ax.text(
            x,
            plaid / 2,
            f"{plaid:,.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=FONT_SIZE - 3,
        )
    if splade > 0:
        ax.text(
            x,
            plaid + splade / 2,
            f"{splade:,.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=FONT_SIZE - 3,
        )
    ax.text(
        x,
        total + max(totals) * 0.015,
        f"Total\n{total:,.1f} Gb",
        va="bottom",
        ha="center",
        fontsize=FONT_SIZE,
        fontweight="bold",
    )

ax.set_xticks(
    list(model_positions),
    labels=[model.replace(" (", "\n(") for model in models],
)
ax.set_ylabel("Index size (GB)", fontsize=FONT_SIZE)
ax.set_title(f"Index size")
ax.legend(loc="upper right", frameon=True)
ax.set_ylim(0, max(totals) * 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()

output_path = Path("figures/ecir_index_size.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved index-size plot to {output_path}")
# plt.show()
