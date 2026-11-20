"""
repeated_users_sensitivity.py -- answers the reviewer's questions on repeated users and sphericity.

Run inside results/ (the folder that holds results_per_user.csv); stats_analysis.py is imported from
code/ via PYTHONPATH:

    cd results && PYTHONPATH=../code python ../code/repeated_users_sensitivity.py results_per_user.csv --N 10 --reps 300
    (the paper uses --reps 300 at Top-10 and --reps 100 at N = 5, 15, 20)

Needs numpy, scipy, pandas only. Writes:
  overlap_table.csv        users per condition, users in 1/2/3 conditions, pairwise overlaps
  anova_original.csv       the paper's mixed ANOVA + Greenhouse-Geisser epsilon and corrected p
  anova_single_condition.csv   sensitivity analysis: every user is kept in ONE random condition
                           (repeated `reps` times), so all observations are independent.
                           Reports median F, median partial eta^2 and the share of repetitions with p<.05.
"""
import sys, argparse, itertools
import numpy as np, pandas as pd
from scipy import stats
import stats_analysis as sa          # mixed_anova, _helmert from the paper's code

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--reps", type=int, default=500); ap.add_argument("--N", type=int, default=10)
a = ap.parse_args()

df = pd.read_csv(a.path)
per_user = (df.groupby(["scenario", "method", "impute", "ssm", "fusion", "N", "user"], dropna=False)
            [sa.METRICS].mean().reset_index())
ph = per_user[per_user.method.str.startswith("P-HSO") & (per_user.impute == "zero") & (per_user.N == a.N)]
SC = [c for c in ["ML0", "ML5", "ML10"] if c in set(ph.scenario)]
SS = ["Res", "Lin", "JC"]
FU = [f for f in ["M", "X", "W", "PW"] if f in set(ph.fusion.dropna())]

# ------------------------------------------------------------------ 1. overlap of users across conditions
sets = {c: set(ph[ph.scenario == c].user) for c in SC}
allusers = set().union(*sets.values())
cnt = pd.Series({u: sum(u in s for s in sets.values()) for u in allusers})
rows = [dict(item=f"users in {c}", value=len(s)) for c, s in sets.items()]
rows += [dict(item="user-condition units used in the ANOVA", value=sum(len(s) for s in sets.values())),
         dict(item="distinct users", value=len(allusers))]
rows += [dict(item=f"users appearing in {k} condition(s)", value=int((cnt == k).sum())) for k in (1, 2, 3)]
rows += [dict(item=f"overlap {x} & {y}", value=len(sets[x] & sets[y])) for x, y in itertools.combinations(SC, 2)]
pd.DataFrame(rows).to_csv("overlap_table.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))


def wide(metric, keep=None):
    groups = []
    for c in SC:
        w = (ph[ph.scenario == c].pivot_table(index="user", columns=["fusion", "ssm"], values=metric))
        w = w[[(f, s_) for f in FU for s_ in SS]].dropna()
        if keep is not None:
            w = w.loc[w.index.intersection(keep[c])]
        groups.append(w)
    return groups


# ------------------------------------------------------------------ 2. Greenhouse-Geisser epsilon of the paper's ANOVA
def gg_table(groups, names=("Fusion", "SSM"), levels=(len(FU), len(SS))):
    one = [np.ones((L, 1)) / np.sqrt(L) for L in levels]
    con = [sa._helmert(L) for L in levels]
    N = sum(len(g) for g in groups); G = len(groups)
    out = []
    for r in range(1, len(levels) + 1):
        for E in itertools.combinations(range(len(levels)), r):
            C = None
            for j in range(len(levels)):
                M = con[j] if j in E else one[j]
                C = M if C is None else np.kron(C, M)
            q = C.shape[1]
            if q == 1:
                eps = 1.0
            else:   # pooled within-condition covariance of the contrast scores
                resid = np.vstack([(g.values @ C) - (g.values @ C).mean(0) for g in groups])
                S = resid.T @ resid / (N - G)
                eps = np.trace(S) ** 2 / (q * np.trace(S @ S)) if np.trace(S @ S) > 0 else 1.0
            out.append((" x ".join(names[j] for j in E), q, eps))
    return out


rows = []
for m in sa.METRICS:
    groups = wide(m); G = [g.values for g in groups]
    an = sa.mixed_anova(G, ["Fusion", "SSM"], list(map(len, [FU, SS])))
    eps = {n: e for n, q, e in gg_table(groups)}
    for _, r in an.iterrows():
        base = r.effect.replace(" x Condition", "")
        e = eps.get(base, 1.0)
        if r.effect == "Condition":
            e = 1.0
        rows.append(dict(metric=m, N=a.N, effect=r.effect, df1=r.df1, df2=r.df2, F=r.F, p=r.p,
                         partial_eta2=r.partial_eta2, GG_epsilon=e,
                         p_GG=stats.f.sf(r.F, r.df1 * e, r.df2 * e)))
pd.DataFrame(rows).to_csv("anova_original.csv", index=False)
print(pd.DataFrame(rows)[["metric", "effect", "F", "p", "GG_epsilon", "p_GG"]].to_string(index=False))

# ------------------------------------------------------------------ 3. sensitivity: one random condition per user
rng = np.random.default_rng(11)
acc = {}
for rep in range(a.reps):
    pick = {u: SC[rng.integers(len(SC))] for u in allusers}
    # a user is only assigned to a condition in which the user was a test user
    pick = {u: rng.choice([c for c in SC if u in sets[c]]) for u in allusers}
    keep = {c: {u for u, cc in pick.items() if cc == c} for c in SC}
    for m in sa.METRICS:
        groups = [g.values for g in wide(m, keep)]
        an = sa.mixed_anova(groups, ["Fusion", "SSM"], [len(FU), len(SS)])
        for _, r in an.iterrows():
            acc.setdefault((m, r.effect), []).append((r.F, r.partial_eta2, r.p))
rows = []
for (m, eff), v in acc.items():
    v = np.array(v)
    rows.append(dict(metric=m, N=a.N, effect=eff, median_F=np.median(v[:, 0]), median_eta2=np.median(v[:, 1]),
                     share_p_below_05=(v[:, 2] < .05).mean(), share_p_below_001=(v[:, 2] < .001).mean()))
pd.DataFrame(rows).to_csv("anova_single_condition.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
