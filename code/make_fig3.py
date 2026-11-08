"""Fig. 3: dot-and-whisker comparison of all methods at Top-10, with PCA reference line."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "table_mean_ci.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "fig3.png"
d = pd.read_csv(src)
d = d[d.N == 10]

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
SC = [("ML0", "(a) ML0"), ("ML5", "(b) ML5"), ("ML10", "(c) ML10")]
METHODS = [("P-HSO-Lin-PW", "P-HSO-Lin, personalized weighted"), ("P-HSO-Lin-W", "P-HSO-Lin, weighted"),
           ("P-HSO-Lin-X", "P-HSO-Lin, multiplicative"), ("P-HSO-Lin-M", "P-HSO-Lin, mean"),
           ("PCA", "PCA"), ("SVD", "SVD"), ("NMF", "NMF"), ("HSO-SVD", "HSO-SVD"),
           ("HSO-NMF", "HSO-NMF"), ("CBSO", "CBSO")]
MET = [("precision", "Precision@10", 1, 0.72), ("ndcg", "NDCG@10", 1, 0.72)]
BLUE, GREY = "#2878d4", "#4a4a4a"


def val(sc, m, met):
    r = d[(d.scenario == sc) & (d.method == m) & (d.metric == met)].iloc[0]
    return r["mean"], r.ci_low, r.ci_high


fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6), sharey=True)
ys = list(range(len(METHODS)))[::-1]
for row, (met, xlab, f, xmax) in enumerate(MET):
    for col, (sc, title) in enumerate(SC):
        ax = axes[row, col]
        pca = val(sc, "PCA", met)[0] * f
        ax.axvline(pca, color="black", ls=(0, (4, 2)), lw=1.2, zorder=1)
        for y, (m, _) in zip(ys, METHODS):
            mu, lo, hi = (v * f for v in val(sc, m, met))
            c = BLUE if m.startswith("P-HSO") else GREY
            ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt="o", color=c, ms=5,
                        elinewidth=1.4, capsize=3.5, zorder=3)
        ax.axhline(ys[3] - 0.5, color="#cccccc", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([n for _, n in METHODS])
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
        ax.grid(axis="x", color="#e5e5e5", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(xlab)
        if row == 0:
            ax.set_title(title, fontsize=12)

h = [plt.Line2D([], [], color="black", ls=(0, (4, 2)), lw=1.2),
     plt.Line2D([], [], color=BLUE, marker="o", ls=""),
     plt.Line2D([], [], color=GREY, marker="o", ls="")]
fig.legend(h, ["PCA mean (reference)", "P-HSO (proposed)", "Baselines"], loc="lower center",
           ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(out, dpi=200)
print("saved", out)
