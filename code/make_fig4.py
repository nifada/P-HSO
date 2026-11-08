"""Fig. 4: P@10 / NDCG@10 of P-HSO by fusion strategy and similarity measure, with PCA reference."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "table_mean_ci.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "fig4.png"
d = pd.read_csv(src)
d = d[d.N == 10]

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
SC = ["ML0", "ML5", "ML10"]
FU = [("M", "(a) Mean"), ("X", "(b) Multiplicative"), ("W", "(c) Weighted"), ("PW", "(d) Personalized weighted")]
SS = [("Res", "Resnik", "#2878d4"), ("Lin", "Lin", "#e8683a"), ("JC", "Jiang-Conrath", "#1fae7b")]
MET = [("precision", "Precision@10", 1), ("ndcg", "NDCG@10", 1)]


def val(sc, m, met):
    r = d[(d.scenario == sc) & (d.method == m) & (d.metric == met)].iloc[0]
    return r["mean"], r.ci_low, r.ci_high


fig, axes = plt.subplots(2, 4, figsize=(10.5, 6.6), sharey="row")
w = 0.26
for row, (met, ylab, f) in enumerate(MET):
    for col, (fu, title) in enumerate(FU):
        ax = axes[row, col]
        for i, sc in enumerate(SC):
            for j, (s, _, c) in enumerate(SS):
                m, lo, hi = val(sc, f"P-HSO-{s}-{fu}", met)
                x = i + (j - 1) * w
                ax.bar(x, m * f, w * 0.92, color=c, zorder=2)
                ax.errorbar(x, m * f, yerr=[[(m - lo) * f], [(hi - m) * f]], fmt="none",
                            ecolor="#333", elinewidth=1.1, capsize=3, zorder=3)
            p = val(sc, "PCA", met)[0] * f
            # PCA reference drawn in front of the bars so it stays visible
            ax.hlines(p, i - 1.6 * w, i + 1.6 * w, colors="black", linestyles=(0, (4, 2)),
                      linewidth=1.8, zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels(SC)
        ax.set_ylim(0, 0.67)
        ax.grid(axis="y", color="#e5e5e5", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=12)
        if col == 0:
            ax.set_ylabel(ylab)

h = [plt.Line2D([], [], color="black", ls=(0, (4, 2)), lw=1.8)] + \
    [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SS]
fig.legend(h, ["PCA alone"] + [n for _, n, _ in SS], loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(out, dpi=200)
print("saved", out)
