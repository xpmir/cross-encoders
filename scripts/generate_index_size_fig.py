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
        "Plaid Store": 90242.8,
        "plaid index": 0,
    },
    "LateOn (Plaid 2bits)": {
        "splade": 0,
        "Plaid Store":48536.281,
        "plaid index": 3232.664,
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
plaid_store_sizes = [
    index_sizes[model]["Plaid Store"] / mb_to_gb for model in models
]
plaid_index_sizes = [
    index_sizes[model]["plaid index"] / mb_to_gb for model in models
]
totals = [
    store + idx + splade
    for store, idx, splade in zip(
        plaid_store_sizes, plaid_index_sizes, splade_sizes
    )
]

fig, ax = plt.subplots(figsize=FIG_SIZE)
model_positions = range(len(models))
ax.bar(
    model_positions,
    plaid_store_sizes,
    width=0.55,
    color="#e76f51",
    label="PLAID-Store",
)
ax.bar(
    model_positions,
    plaid_index_sizes,
    bottom=plaid_store_sizes,
    width=0.55,
    color="#f4a261",
    label="PLAID-index",
)
ax.bar(
    model_positions,
    splade_sizes,
    bottom=[
        store + idx
        for store, idx in zip(plaid_store_sizes, plaid_index_sizes)
    ],
    width=0.55,
    color="#2a9d8f",
    label="SPLADE",
)

for x, splade, store, idx, total in zip(
    model_positions, splade_sizes, plaid_store_sizes, plaid_index_sizes, totals
):
    if store > 0:
        ax.text(
            x,
            store / 2,
            f"{store:,.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=FONT_SIZE - 3,
        )
    if idx > 0:
        ax.text(
            x,
            store + idx / 2,
            f"{idx:,.1f}",
            ha="center",
            va="center",
            color="#264653",
            fontweight="bold",
            fontsize=FONT_SIZE - 3,
        )
    if splade > 0:
        ax.text(
            x,
            store + idx + splade / 2,
            f"{splade:,.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=FONT_SIZE - 3,
        )
    ax.text(
        x,
        total + max(totals) * 0.015,
        f"Total\n{total:,.1f} GB",
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
ax.set_title("Index size")
ax.legend(loc="upper right", frameon=True)
ax.set_ylim(0, max(totals) * 1.3)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()

output_path = Path("figures/ecir_index_size.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved index-size plot to {output_path}")
# plt.show()
