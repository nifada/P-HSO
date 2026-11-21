"""extra_baseline_tests.py -- paired Wilcoxon tests (Holm) for TopPop and EM-PCA.

    cd results && python ../code/extra_baseline_tests.py results_per_user.csv

Compares PCA with TopPop and EM-PCA (and damped EM-PCA), and SVD with TopPop, per condition, list size
and metric, on the same users (metric averaged over the splits of each user, as in the paper).
Writes table_extra_baselines.csv: a, b, n, mean_diff_a_minus_b, p, r_rb (matched-pairs rank-biserial),
p_holm (Holm within condition, list size and metric). Needs numpy, scipy, pandas.
"""
import sys, numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1])
pu = d.groupby(["scenario", "method", "N", "user"])[["precision", "ndcg"]].mean().reset_index()   # per user, mean over folds (as in the paper)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj
print("MEANS (mean over users)")
m = pu.groupby(["scenario", "method", "N"])[["precision", "ndcg"]].mean().round(4).reset_index()
print(m[m.N == 10].pivot(index="method", columns="scenario", values="precision")[["ML0", "ML5", "ML10"]].to_string())
print("\nNDCG@10"); print(m[m.N == 10].pivot(index="method", columns="scenario", values="ndcg")[["ML0", "ML5", "ML10"]].to_string())
rows = []
pairs = [("PCA", "TopPop"), ("SVD", "TopPop"), ("PCA", "EM-PCA"), ("PCA", "EM-PCA-damped")]
for sc in ["ML0", "ML5", "ML10"]:
    for N in (10, 20):
        for met in ("precision", "ndcg"):
            grp = []
            for a, b in pairs:
                x = pu[(pu.scenario == sc) & (pu.N == N) & (pu.method == a)].set_index("user")[met]
                y = pu[(pu.scenario == sc) & (pu.N == N) & (pu.method == b)].set_index("user")[met]
                x, y = x.align(y, join="inner"); dd = x - y
                if (dd != 0).sum() == 0: continue
                w = stats.wilcoxon(x, y)
                nz = dd[dd != 0]; rk = stats.rankdata(nz.abs()); rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
                grp.append(dict(scenario=sc, N=N, metric=met, a=a, b=b, n=len(x), mean_diff_a_minus_b=dd.mean(), p=w.pvalue, r_rb=rb))
            if grp:
                p = holm([g["p"] for g in grp])
                for g, pp in zip(grp, p): g["p_holm"] = pp
                rows += grp
r = pd.DataFrame(rows); r.to_csv("table_extra_baselines.csv", index=False)
pd.set_option("display.width", 200)
print("\nPAIRED WILCOXON, Top-10 (a - b; Holm within scenario/N/metric)")
print(r[(r.N == 10)].round(4).drop(columns=["N"]).to_string(index=False))
