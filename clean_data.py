# %%

import pandas as pd
import pathlib

# %%

DATA_PATH = pathlib.Path(__file__).parent / "data.xlsx"

NAME = "Namn"
LETTER_CODE = "Bokstavskod"
NUM_CODE = "Sifferkod"
TESTER = "Testare"
NUT_TYPE = "Nöttyp"
PROP_VARS = [
    "Flottig",
    "Knaprig",
    "Rostad",
    "Salt",
]
SCORE_VAR = "Betyg"

translation_data = pd.read_excel(DATA_PATH, sheet_name="Sifferkoder").set_index(
    LETTER_CODE
)[NUM_CODE]
nut_data = (
    pd.read_excel(DATA_PATH, sheet_name="Nötter")
    .join(translation_data, on=LETTER_CODE)
    .set_index(NUM_CODE)
    .sort_index()
)
evaluation_data = pd.read_excel(DATA_PATH, sheet_name="Utvärdering").set_index(
    [TESTER, NUM_CODE]
)

num_to_name = nut_data[NAME]

num_to_name

# %%

nut_types = list(nut_data[NUT_TYPE].unique())
nut_types

# %%

nuts_by_type = {
    nut_type: list(nut_data.loc[lambda d: d[NUT_TYPE] == nut_type].index)
    for nut_type in nut_types
}
nuts_by_type

# %%

quant_data = evaluation_data[[*PROP_VARS, SCORE_VAR]]
quant_data

# %%

NUT_STYLES = [
    dict(marker=marker, color=color)
    for marker, color in zip("vsdpo>", matplotlib.colormaps["tab10"].colors)
]

# %%


def num_responses_fig(quant_data, nut_type):
    fig, ax = plt.subplots(figsize=(8, 6))

    counts = (
        quant_data.reindex(nuts_by_type[nut_type], level=NUM_CODE)
        .groupby(NUM_CODE)
        .count()
        .sort_index(ascending=False)
        .rename(lambda num_code: f"{num_to_name[num_code]} ({num_code})")
        .rename_axis("Namn")
    )
    counts.plot.barh(ax=ax)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Antal svar")

    fig.suptitle("Antal svar\n" + nut_type)

    return fig


figdir = pathlib.Path("figs")
figdir.mkdir(exist_ok=True)

for nt in nut_types:
    fig = num_responses_fig(quant_data, nt)
    fig.savefig(figdir / f"counts-{nt}.png", dpi=300, bbox_inches="tight")
# %%


def score_prediction_fig(quant_data, nut_type):
    quant_data = quant_data.reindex(nuts_by_type[nut_type], level=NUM_CODE)
    nut_means = quant_data.groupby(NUM_CODE).mean()
    nut_stderrs = (
        quant_data.groupby(NUM_CODE).std() / quant_data.groupby(NUM_CODE).count() ** 0.5
    )

    fig, axs = plt.subplots(ncols=len(PROP_VARS), sharey=True, figsize=(9, 4))

    for ax, prop_var in zip(axs, PROP_VARS):
        ax_data = pd.DataFrame(
            {
                "x": nut_means[prop_var],
                "y": nut_means[SCORE_VAR],
                "yerr": nut_stderrs[prop_var] * 1.96,
            }
        )

        for num_code, style in zip(ax_data.index.unique(NUM_CODE), NUT_STYLES):
            ax.errorbar(
                **ax_data.xs(num_code),
                lw=0,
                elinewidth=1,
                capsize=5,
                **style,
                label=f"{num_to_name[num_code]} ({num_code})",
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

    axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncols=3)

    fig.tight_layout()

    fig.suptitle(nut_type)

    return fig


figdir = pathlib.Path("figs")
figdir.mkdir(exist_ok=True)

for nt in nut_types:
    fig = score_prediction_fig(quant_data, nt)
    fig.savefig(figdir / f"predictors-{nt}.png", dpi=300, bbox_inches="tight")

# %%
