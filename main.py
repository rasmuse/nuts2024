# %%

import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.cm
import matplotlib.pyplot as plt

# %%

all_data = (
    pd.read_csv("peanuts-2016-data.csv")
    .set_index(["TBF", "Nöt", "Timestamp"])
    .sort_index()
    .groupby(["TBF", "Nöt"])
    .last()
)
all_data

# %%

PROP_VARS = [
    "Flottig",
    "Knaprig",
    "Rostad",
    "Salt",
]
SCORE_VAR = "Betyg"

d = all_data.reindex(all_data.index.unique("TBF")[:10], level="TBF").reindex(
    all_data.index.unique("Nöt")[:5], level="Nöt"
)[[SCORE_VAR, *PROP_VARS]]

d

# %%

nut_means = d.groupby("Nöt").mean()
nut_means

# %%

nut_stderrs = d.groupby("Nöt").std() / d.groupby("Nöt").count() ** 0.5
nut_stderrs

# %%

nut_styles = [
    dict(marker=marker, color=color)
    for marker, color in zip("vsdpo>", matplotlib.cm.get_cmap("tab10").colors)
]

fig, axs = plt.subplots(ncols=len(PROP_VARS), sharey=True, figsize=(8, 3))

for ax, prop_var in zip(axs, PROP_VARS):
    ax_data = pd.DataFrame(
        {
            "x": nut_means[prop_var],
            "y": nut_means[SCORE_VAR],
            "yerr": nut_stderrs[prop_var] * 1.96,
        }
    )

    for nut, style in zip(ax_data.index.unique("Nöt"), nut_styles):
        ax.errorbar(
            **ax_data.xs(nut),
            lw=0,
            elinewidth=1,
            capsize=5,
            **style,
            label=nut,
        )

    ax.set_xlabel(prop_var)

    regression = scipy.stats.linregress(ax_data["x"], ax_data["y"])
    x_pred = ax_data["x"].sort_values()
    y_pred = regression.intercept + x_pred * regression.slope
    ax.plot(x_pred, y_pred, lw=1, ls="--", color="k")

    ax.text(
        0.05,
        0.98,
        f"p = {regression.pvalue:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )

axs[0].set_ylabel(SCORE_VAR)

axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.05))

# %%

raw_data = pd.DataFrame(
    [
        ["AAA", "J1", 5, 3, 4],
        ["AAA", "J2", 3, 2, 7],
        ["BBB", "J1", 6, 4, 5],
    ]
).set_axis(["TBF", "Nöt", "Prop1", "Prop2", "Score"], axis=1)

ALPHA_TO_NUM = {
    "JA": "J2",
    "JB": "J1",
}
NAME_TO_ALPHA = {
    "Jord1": "JA",
    "Jord2": "JB",
}

NUM_TO_NAME = {ALPHA_TO_NUM[alpha]: name for name, alpha in NAME_TO_ALPHA.items()}

raw_data.replace(NUM_TO_NAME)
