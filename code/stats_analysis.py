"""
stats_analysis.py  --  statistics for the revision, from results_per_user.csv.

    cd results && python ../code/stats_analysis.py results_per_user.csv

Writes:
  table_mean_ci.csv    mean, SD and bootstrap 95% CI per method x scenario x N   (Figs. 3-4, Tables IV-V)
  table_anova.csv      mixed-design (split-plot) ANOVA: Fusion (4) and SSM (3) within
                       users, Condition (3) between users, all interactions:
                       F, df1, df2, p, partial eta^2                           (Table VI)
  table_pairwise.csv   paired Wilcoxon signed-rank, best P-HSO vs each baseline,
                       Holm-corrected, with matched-pairs rank-biserial r        (Section III.A)

Unit of analysis = user within a condition (metric averaged over the folds in which
the user was a test user).  Every P-HSO configuration is evaluated on the same users
(within-subject), but ML0/ML5/ML10 draw their test users separately, so Condition is
a between-subject factor.  Unequal group sizes are handled with unweighted
(Type III) means.
Needs only numpy, scipy, pandas.
"""
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["precision", "ndcg"]
RNG = np.random.default_rng(7)


def boot_ci(x, B=2000):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def _helmert(k):
    """orthonormal contrasts for a k-level factor (k x k-1)"""
    H = np.zeros((k, k - 1))
    for j in range(1, k):
        H[:j, j - 1] = 1.0 / j
        H[j, j - 1] = -1.0
        H[:, j - 1] /= np.linalg.norm(H[:, j - 1])
    return H


def mixed_anova(groups, within_names, levels, between_name="Condition"):
    """groups: list of arrays, one per between-level, each (n_g, prod(levels)),
    cells ordered with the last within factor varying fastest.
    Returns F tests for the between factor, every within effect, and every
    within effect x between interaction (sphericity assumed)."""
    G = len(groups)
    ns = np.array([len(g) for g in groups])
    N = ns.sum()
    one = [np.ones((L, 1)) / np.sqrt(L) for L in levels]
    con = [_helmert(L) for L in levels]
    rows = []

    def oneway(scores):
        """scores: list per group of (n_g, q) arrays -> SS_intercept(III), SS_between, SS_error"""
        q = scores[0].shape[1]
        means = np.array([s.mean(0) for s in scores])                 # G x q
        sse = sum(((s - s.mean(0)) ** 2).sum() for s in scores)
        L = means.mean(0)
        ss_int = (L ** 2).sum() / ((1.0 / ns).sum() / G ** 2)
        gm = np.concatenate(scores).mean(0)
        ssb = (ns[:, None] * (means - gm) ** 2).sum()
        return ss_int, ssb, sse, q

    # between-subject effect: subject totals (constant contrast)
    C0 = one[0]
    for M in one[1:]:
        C0 = np.kron(C0, M)
    _, ssb, sse, q = oneway([g @ C0 for g in groups])
    d1, d2 = G - 1, N - G
    F = (ssb / d1) / (sse / d2)
    rows.append(dict(effect=between_name, df1=d1, df2=d2, F=F, p=stats.f.sf(F, d1, d2),
                     partial_eta2=ssb / (ssb + sse)))
    # within effects
    W = len(levels)
    for r in range(1, W + 1):
        for E in itertools.combinations(range(W), r):
            C = None
            for j in range(W):
                M = con[j] if j in E else one[j]
                C = M if C is None else np.kron(C, M)
            ss_int, ssb, sse, q = oneway([g @ C for g in groups])
            name = " x ".join(within_names[j] for j in E)
            d2 = q * (N - G)
            for label, ss, d1 in ((name, ss_int, q), (name + " x " + between_name, ssb, q * (G - 1))):
                F = (ss / d1) / (sse / d2)
                rows.append(dict(effect=label, df1=int(d1), df2=int(d2), F=F, p=stats.f.sf(F, d1, d2),
                                 partial_eta2=ss / (ss + sse)))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1, run)
    return adj


def main(path):
    df = pd.read_csv(path)
    per_user = (df.groupby(["scenario", "method", "impute", "ssm", "fusion", "N", "user"], dropna=False)
                [METRICS].mean().reset_index())

    # ---- 1. mean, SD, 95% CI
    rows = []
    for (sc, me, N), g in per_user.groupby(["scenario", "method", "N"]):
        for m in METRICS:
            lo, hi = boot_ci(g[m])
            rows.append(dict(scenario=sc, method=me, N=N, metric=m, mean=g[m].mean(),
                             sd=g[m].std(ddof=1), ci_low=lo, ci_high=hi, n_users=len(g)))
    pd.DataFrame(rows).to_csv("table_mean_ci.csv", index=False)

    # ---- 2. mixed-design ANOVA on the twelve P-HSO configurations (zero filling)
    ph = per_user[per_user.method.str.startswith("P-HSO") & (per_user.impute == "zero")]
    SS = ["Res", "Lin", "JC"]
    FU = [f for f in ["M", "X", "W", "PW"] if f in set(ph.fusion.dropna())]
    SC = [c for c in ["ML0", "ML5", "ML10"] if c in set(ph.scenario)]
    an = []
    for N in sorted(ph.N.unique()):
        for m in METRICS:
            groups = []
            for c in SC:
                w = (ph[(ph.N == N) & (ph.scenario == c)]
                     .pivot_table(index="user", columns=["fusion", "ssm"], values=m))
                w = w[[(f, s_) for f in FU for s_ in SS]].dropna()
                groups.append(w.values)
            r = mixed_anova(groups, ["Fusion", "SSM"], [len(FU), len(SS)])
            r.insert(0, "fusion_levels", "/".join(FU))
            r.insert(0, "metric", m)
            r.insert(0, "N", N)
            an.append(r)
    print("users per condition:", dict(zip(SC, [len(g) for g in groups])))
    pd.concat(an).to_csv("table_anova.csv", index=False)

    # ---- 3. best P-HSO vs baselines, paired Wilcoxon + Holm
    base = ["CBSO", "SVD", "NMF", "PCA", "HSO-SVD", "HSO-NMF"]
    pw = []
    for (sc, N), g in per_user.groupby(["scenario", "N"]):
        for m in METRICS:
            ph_means = g[g.method.str.startswith("P-HSO") & (g.impute == "zero")].groupby("method")[m].mean()
            best = ph_means.idxmax()
            a = g[g.method == best].set_index("user")[m]
            for b in base:
                bb = g[g.method == b].set_index("user")[m]
                if bb.empty:
                    continue
                x, y = a.align(bb, join="inner")
                d = x - y
                if (d != 0).sum() == 0:
                    continue
                res = stats.wilcoxon(x, y, zero_method="wilcox")
                rk = stats.rankdata(np.abs(d[d != 0]))
                rbc = (rk[d[d != 0] > 0].sum() - rk[d[d != 0] < 0].sum()) / rk.sum()
                pw.append(dict(scenario=sc, N=N, metric=m, best_phso=best, baseline=b,
                               mean_diff=d.mean(), W=res.statistic, p=res.pvalue, r_rb=rbc))
    pw = pd.DataFrame(pw)
    if len(pw):
        pw["p_holm"] = pw.groupby(["scenario", "N", "metric"])["p"].transform(holm)
    pw.to_csv("table_pairwise.csv", index=False)
    print("wrote table_mean_ci.csv, table_anova.csv, table_pairwise.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_per_user.csv")
