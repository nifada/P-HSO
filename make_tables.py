"""Builds the values of Tables III-VIII, the fusion weights, the ablation and the average ranks.
Run inside results/ after stats_analysis.py:
    cd results && python ../code/make_tables.py
Inputs : results_per_user.csv, table_mean_ci.csv, table_anova.csv, table_pairwise.csv, chosen_k.json
Output : tables.json and run_results.xlsx
"""
import json, numpy as np, pandas as pd

R = pd.read_csv("results_per_user.csv")
SC = ["ML0", "ML5", "ML10"]
NS = [5, 10, 15, 20]
ORDER = ["CBSO", "SVD", "NMF", "PCA", "HSO-SVD", "HSO-NMF",
         "P-HSO-Res-M", "P-HSO-Lin-M", "P-HSO-JC-M", "P-HSO-Res-X", "P-HSO-Lin-X", "P-HSO-JC-X",
         "P-HSO-Res-W", "P-HSO-Lin-W", "P-HSO-JC-W", "P-HSO-Res-PW", "P-HSO-Lin-PW", "P-HSO-JC-PW"]
REF = {"CBSO": "CBSO [28]", "SVD": "SVD [33]", "NMF": "NMF [35], [36]", "HSO-SVD": "HSO-SVD [22]", "HSO-NMF": "HSO-NMF [22]"}

pu = (R.groupby(["scenario", "method", "N", "user"])[["precision", "ndcg"]].mean().reset_index())
agg = pu.groupby(["scenario", "method", "N"]).agg(p_mean=("precision", "mean"), p_sd=("precision", "std"),
                                                  n_mean=("ndcg", "mean"), n_sd=("ndcg", "std"),
                                                  users=("user", "nunique")).reset_index()


def perf_table(metric):
    rows = []
    for m in ORDER:
        row = [REF.get(m, m)]
        for sc in SC:
            for N in NS:
                r = agg[(agg.method == m) & (agg.scenario == sc) & (agg.N == N)].iloc[0]
                if metric == "p":
                    row.append(f"{r.p_mean:.4f} ± {r.p_sd:.4f}")
                else:
                    row.append(f"{r.n_mean:.4f} ± {r.n_sd:.4f}")
        rows.append(row)
    return rows


# best value per column (P-HSO and baselines) for bolding
def best_cols(metric):
    out = {}
    for sc in SC:
        for N in NS:
            a = agg[(agg.scenario == sc) & (agg.N == N) & agg.method.isin(ORDER)]
            col = "p_mean" if metric == "p" else "n_mean"
            out[(sc, N)] = a.loc[a[col].idxmax(), "method"]
    return out


# Table III: median k over 5 splits
ck = json.load(open("chosen_k.json"))
k = pd.DataFrame([dict(scenario=key.split("|")[0], fold=key.split("|")[1], method=key.split("|")[2], k=v)
                  for key, v in ck.items() if v is not None and not key.endswith("|alpha")])
latent = ["SVD", "NMF", "PCA", "HSO-SVD", "HSO-NMF"] + [m for m in ORDER if m.startswith("P-HSO")]
t3 = []
for m in latent:
    row = [REF.get(m, m)]
    for sc in SC:
        v = k[(k.method == m) & (k.scenario == sc)].k
        row.append(f"{int(np.median(v))} ({int(v.min())}-{int(v.max())})")
    t3.append(row)

al = []
for m in [x for x in ORDER if x.endswith("-W") or x.endswith("-PW")]:
    row = [m]
    for sc in SC:
        v = [ck[f"{sc}|fold{f}|{m}|alpha"] for f in range(1, 6)]
        row.append(f"{np.median(v):.1f} ({min(v):.1f}-{max(v):.1f})")
    al.append(row)

# ablation: zero vs mean imputation at N = 10
abl = []
for m in [x for x in ORDER if x.startswith("P-HSO") and (x.endswith("-M") or x.endswith("-X"))]:
    row = [m]
    for sc in SC:
        z = agg[(agg.method == m) & (agg.scenario == sc) & (agg.N == 10)].iloc[0]
        mi = agg[(agg.method == m + "-meanimp") & (agg.scenario == sc) & (agg.N == 10)].iloc[0]
        row += [f"{z.p_mean:.4f}", f"{mi.p_mean:.4f}", f"{z.n_mean:.4f}", f"{mi.n_mean:.4f}"]
    abl.append(row)

# ANOVA (Table VIII)
A = pd.read_csv("table_anova.csv")
eff = ["Fusion", "SSM", "Condition", "Fusion x SSM", "Fusion x Condition", "SSM x Condition", "Fusion x SSM x Condition"]
t6 = []
for e in eff:
    row = [e.replace(" x ", " × ")]
    for met in ["precision", "ndcg"]:
        for N in NS:
            r = A[(A.effect == e) & (A.metric == met) & (A.N == N)].iloc[0]
            p = "< .001" if r.p < 0.001 else f"{r.p:.3f}"
            row.append(f"F({int(r.df1)},{int(r.df2)}) = {r.F:.2f}; p {p if p.startswith('<') else '= ' + p}; ηp² = {r.partial_eta2:.3f}")
    t6.append(row)

# pairwise (best P-HSO vs baselines), N = 10
PW = pd.read_csv("table_pairwise.csv")
pw = []
for sc in SC:
    for met in ["precision", "ndcg"]:
        s = PW[(PW.scenario == sc) & (PW.N == 10) & (PW.metric == met)]
        for _, r in s.iterrows():
            pw.append([sc, met, r.best_phso, r.baseline, f"{r.mean_diff:+.4f}", f"{r.p_holm:.2g}", f"{r.r_rb:+.3f}"])

users = {sc: int(pu[pu.scenario == sc].user.nunique()) for sc in SC}
out = dict(alpha=al, t4=perf_table("p"), t5=perf_table("n"), best_p={f"{a}|{b}": v for (a, b), v in best_cols("p").items()},
           best_n={f"{a}|{b}": v for (a, b), v in best_cols("n").items()}, t3=t3, abl=abl, t6=t6, pw=pw, users=users)
json.dump(out, open("tables.json", "w"), ensure_ascii=False, indent=1)

with pd.ExcelWriter("run_results.xlsx") as xw:
    hdr = ["Method"] + [f"{sc} @{N}" for sc in SC for N in NS]
    pd.DataFrame(out["t4"], columns=hdr).to_excel(xw, "Table IV Precision", index=False)
    pd.DataFrame(out["t5"], columns=hdr).to_excel(xw, "Table V NDCG", index=False)
    pd.DataFrame(t3, columns=["Method"] + [f"{sc} median (range)" for sc in SC]).to_excel(xw, "Table III k", index=False)
    pd.DataFrame(al, columns=["Method"] + [f"{sc} alpha median (range)" for sc in SC]).to_excel(xw, "Fusion weight alpha", index=False)
    pd.DataFrame(abl, columns=["Configuration"] + [f"{sc} {x}" for sc in SC for x in ("P@10 zero", "P@10 mean-imp", "NDCG@10 zero", "NDCG@10 mean-imp")]).to_excel(xw, "Ablation imputation", index=False)
    pd.DataFrame(t6, columns=["Effect"] + [f"{m} @{N}" for m in ("Precision", "NDCG") for N in NS]).to_excel(xw, "Table VIII ANOVA", index=False)
    A.to_excel(xw, "ANOVA full", index=False)
    PW.to_excel(xw, "Pairwise Wilcoxon", index=False)
    # Table VII: Top-10 differences (P-HSO - baseline), Holm significance stars and r_rb
    st = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    t7 = PW[PW.N == 10].copy()
    t7["cell"] = [f"{d:+.4f}{st(p)} ({r:.2f})" for d, p, r in zip(t7.mean_diff, t7.p_holm, t7.r_rb)]
    t7 = t7.pivot_table(index=["metric", "baseline"], columns="scenario", values="cell", aggfunc="first").reindex(columns=SC)
    t7.to_excel(xw, "Table VII Wilcoxon Top-10")
    pd.read_csv("table_mean_ci.csv").to_excel(xw, "Mean SD CI all", index=False)
    # average rank over condition x list size x metric; ties receive the average of their ranks (Demsar 2006)
    mc = pd.read_csv("table_mean_ci.csv")
    x = mc[mc.method.isin(ORDER)].pivot_table(index=["scenario", "N", "metric"], columns="method", values="mean").round(12)
    rk = x.rank(axis=1, ascending=False, method="average")
    tab = rk.T
    tab.columns = [f"{a} @{b} {c}" for a, b, c in tab.columns]
    tab.insert(0, "Average rank", rk.mean().round(3))
    tab.sort_values("Average rank").to_excel(xw, "Average rank")
    # Table VI: average rank within each condition (4 list sizes x 2 metrics) and overall (mean of the three)
    t6r = pd.DataFrame({sc: rk.loc[sc].mean() for sc in SC})
    t6r["Overall"] = t6r[SC].mean(axis=1)
    t6r.round(3).sort_values("Overall").rename_axis("Method").to_excel(xw, "Table VI Average rank")
print("users per condition:", users)
print(pd.DataFrame(out["t4"], columns=["Method"] + [f"{sc}@{N}" for sc in SC for N in NS])[["Method", "ML0@10", "ML5@10", "ML10@10"]].to_string(index=False))
