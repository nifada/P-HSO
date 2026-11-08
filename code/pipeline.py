"""
pipeline.py -- runs every method of the paper and writes per-user results.

    python code/pipeline.py --scenarios 0  --out results/run/results_ML0.csv
    python code/pipeline.py --scenarios 5  --out results/run/results_ML5.csv
    python code/pipeline.py --scenarios 10 --out results/run/results_ML10.csv

Run each cold-start condition separately, as above: the validation users are drawn with a
fixed seed per run, so this reproduces the validation users of the paper.

Methods (Section II, Tables I-II)
  CBSO, SVD, NMF, PCA, HSO-SVD, HSO-NMF, and P-HSO with Resnik / Lin / Jiang-Conrath and
  mean (M), multiplicative (X), weighted (W) and personalized weighted (PW) fusion;
  ablation: PCA and P-HSO-M/X with user-mean imputation ("-meanimp").
Protocol (Sections II.B, II.G, II.H)
  * test users keep h = 0, 5 or 10 training ratings (ML0, ML5, ML10); the one boundary user per split whose
    ratings were split to reach 20,000 test ratings is excluded (--keep-boundary to include);
  * candidate set: all movies the user did not rate in training; ground truth: all test ratings;
  * k (and alpha for W/PW) is selected by Precision@10 on validation users: 20% of the
    non-test training users, given the same treatment as test users;
  * ties are broken at random.
Equations: Eq. (1)-(6) cf_pca, Eq. (13) cb_scores, Eq. (14)-(15) minmax_rows / minmax_vec,
Eq. (16)-(18) fuse, Eq. (19) cb_personal, Eq. (20)-(23) evaluate (column "ndcg"; "ndcg_std"
uses the standard IDCG over min(|G_u|, N)).
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

N_LIST = [5, 10, 15, 20]
K_GRID = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          30, 40, 50, 60, 70, 80, 90, 100]
SCENARIOS = [0, 5, 10]
FOLDS = [1, 2, 3, 4, 5]
VAL_FRAC = 0.2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "semantic_scores")
SSMS = {1: "Res", 2: "Lin", 3: "JC"}
RNG = np.random.default_rng(2026)


# ----------------------------------------------------------------------- data
def load_fold(base, f, n):
    def rd(p):
        a = [list(map(int, l.strip().split("::")))[:3] for l in open(p) if l.strip()]
        return np.array(a, dtype=float)
    return (rd(os.path.join(base, f"training-fold{f}-rating{n}.txt")),
            rd(os.path.join(base, f"tes-fold{f}-rating{n}.txt")))


def rating_matrix(train, n_user, n_item):
    V = np.zeros((n_user, n_item))
    V[train[:, 0].astype(int) - 1, train[:, 1].astype(int) - 1] = train[:, 2]
    return V


def inner_validation(train, test, n, rng):
    """held-out split of the TRAINING data: returns (val_train, val_test)."""
    test_users = set(np.unique(test[:, 0]).astype(int))
    users = np.array(sorted(set(np.unique(train[:, 0]).astype(int)) - test_users))
    vu = set(rng.choice(users, int(round(VAL_FRAC * len(users))), replace=False))
    keep, hold = [], []
    for u, idx in pd.Series(np.arange(len(train))).groupby(train[:, 0].astype(int)):
        idx = idx.values
        if u in vu:
            p = rng.permutation(idx)
            keep.extend(p[:n]); hold.extend(p[n:])
        else:
            keep.extend(idx)
    return train[np.sort(keep)], train[np.sort(hold)]


def eval_users(train, test, n, keep_boundary):
    users = np.unique(test[:, 0]).astype(int)
    if keep_boundary:
        return users
    cnt = pd.Series(train[:, 0].astype(int)).value_counts()
    return np.array([u for u in users if cnt.get(u, 0) == n])


# ------------------------------------------------------------------ CF models
def cf_pca(V, k, impute="zero"):
    X = V.copy()
    if impute == "mean":
        obs = V > 0
        cnt = obs.sum(1)
        glob = V[obs].mean()
        umean = np.where(cnt > 0, V.sum(1) / np.maximum(cnt, 1), glob)
        X = np.where(obs, V, umean[:, None])
    pca = PCA(n_components=k, svd_solver="full")
    Z = pca.fit_transform(X)            # sklearn centres columns (item means)
    return pca.inverse_transform(Z)     # ... and adds them back here


def cf_svd(V, k):                       # truncated SVD of the zero-filled, uncentred matrix
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


REQUIRE_NIMFA = False                   # set by --require-nimfa


def cf_nmf(V, k, iters=2000):           # nimfa settings of the original code: NNDSVD seed, Euclidean update, 2,000 iterations
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compat"))
        import old_numpy_names            # NumPy-2 aliases for nimfa 1.4.0 (see compat/)
        import nimfa
        fit = nimfa.Nmf(V, seed="nndsvd", max_iter=iters, rank=k,
                        update="euclidean", objective="fro")()
        return np.asarray(fit.basis() @ fit.coef())
    except ImportError:                 # same algorithm in scikit-learn
        if REQUIRE_NIMFA:
            raise
        import warnings
        from sklearn.decomposition import NMF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NMF(k, init="nndsvd", solver="mu", beta_loss="frobenius", max_iter=iters, tol=0)
            W = m.fit_transform(V)
        return W @ m.components_


# ------------------------------------------------------------------ CB scores
def cb_scores(n_item, ssver, dbfile, cache_dir=CACHE):
    """s_i = mean_{j != i} SSM(i, j) over the whole catalogue (Eq. 13), cached."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"scb_all_{SSMS[ssver]}.npy")
    if os.path.exists(path):
        return np.load(path)
    import ssmpy
    ssmpy.semantic_base(dbfile)
    ssmpy.ssm.mica = False          # DiShIn shared IC, extrinsic IC (Eqs. 7-8)
    ssmpy.ssm.intrinsic = False
    f = {1: ssmpy.ssm_resnik, 2: ssmpy.ssm_lin, 3: ssmpy.ssm_jiang_conrath}[ssver]
    ids = [ssmpy.get_id(str(i + 1)) for i in range(n_item)]
    S = np.zeros((n_item, n_item))
    for i in range(n_item):
        for j in range(i + 1, n_item):
            S[i, j] = S[j, i] = f(ids[i], ids[j])
        if i % 100 == 0:
            print(f"  CB {SSMS[ssver]}: {i}/{n_item}", flush=True)
    s = S.sum(1) / (n_item - 1)
    np.save(path, s)
    return s


def cb_pairs(n_item, ssver, dbfile, cache_dir=CACHE):
    """full movie x movie similarity matrix (diagonal 0), cached; needed for personalised CB"""
    path = os.path.join(cache_dir, f"scb_pairs_{SSMS[ssver]}.npy")
    if os.path.exists(path):
        return np.load(path)
    import ssmpy
    ssmpy.semantic_base(dbfile)
    ssmpy.ssm.mica = False
    ssmpy.ssm.intrinsic = False
    f = {1: ssmpy.ssm_resnik, 2: ssmpy.ssm_lin, 3: ssmpy.ssm_jiang_conrath}[ssver]
    ids = [ssmpy.get_id(str(i + 1)) for i in range(n_item)]
    S = np.zeros((n_item, n_item), dtype=np.float32)
    for i in range(n_item):
        for j in range(i + 1, n_item):
            S[i, j] = S[j, i] = f(ids[i], ids[j])
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, S)
    return S


def cb_personal(S, train, n_user, s_global):
    """CB_u[i] = mean similarity of movie i to the movies u rated in training, min-max per user"""
    R = np.zeros((n_user, S.shape[0]), dtype=np.float32)
    R[train[:, 0].astype(int) - 1, train[:, 1].astype(int) - 1] = 1
    cnt = R.sum(1, keepdims=True)
    P_ = (R @ S) / np.maximum(cnt, 1)
    P_ = np.where(cnt > 0, P_, minmax_vec(s_global)[None, :])
    return minmax_rows(P_).astype(np.float32)


ALPHAS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


# --------------------------------------------------------------------- fusion
TOL = 1e-9   # scores whose range is below TOL x scale are treated as constant (floating-point residue)


def minmax_rows(M):
    """Eq. (14): min-max per row over all movies. A row whose scores are all equal (up to rounding),
    e.g. SVD/NMF for a user without training ratings, carries no information and is set to 0."""
    lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
    scale = np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
    const = (hi - lo) <= TOL * scale
    return np.where(const, 0.0, (M - lo) / np.where(const, 1, hi - lo))


def minmax_vec(v):
    """Eq. (15): min-max over the catalogue (same rule for a constant vector)."""
    lo, hi = v.min(), v.max()
    if hi - lo <= TOL * max(1.0, abs(lo), abs(hi)):
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def fuse(cf, cb, how, normalise=True, alpha=None, cbp=None):
    if cb is None:
        return cf
    if normalise:
        cf, cb = minmax_rows(cf), minmax_vec(cb)
    if how == "X":
        return cf * cb[None, :]
    if how == "W":
        return (1 - alpha) * cf + alpha * cb[None, :]
    if how == "PW":
        return (1 - alpha) * cf + alpha * cbp
    return (cf + cb[None, :]) / 2


# ----------------------------------------------------------------- evaluation
def evaluate(scores, train, test, users, nmax=max(N_LIST)):
    users = np.asarray(users, dtype=int)
    if len(users) == 0:
        return pd.DataFrame(columns=["user", "N", "precision", "ndcg", "ndcg_std"])
    S = scores[users - 1].astype(np.float64) + RNG.random((len(users), scores.shape[1])) * 1e-9
    pos = {u: r for r, u in enumerate(users)}
    tr = train[np.isin(train[:, 0].astype(int), users)]
    S[[pos[u] for u in tr[:, 0].astype(int)], tr[:, 1].astype(int) - 1] = -np.inf
    G = np.zeros(S.shape, bool)
    te = test[np.isin(test[:, 0].astype(int), users)]
    G[[pos[u] for u in te[:, 0].astype(int)], te[:, 1].astype(int) - 1] = True
    top = np.argpartition(-S, nmax, axis=1)[:, :nmax]
    order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
    top = np.take_along_axis(top, order, 1)
    hit = np.take_along_axis(G, top, 1).astype(float)
    ng = G.sum(1)
    disc = 1 / np.log2(np.arange(2, nmax + 2))
    rows = []
    for N in N_LIST:
        dcg = (hit[:, :N] * disc[:N]).sum(1)
        idcg = np.array([disc[:min(g, N)].sum() for g in ng])
        rows.append(pd.DataFrame({"user": users, "N": N, "precision": hit[:, :N].sum(1) / N,
                                  "ndcg": dcg / disc[:N].sum(),                     # manuscript Eqs. (19)-(21)
                                  "ndcg_std": np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1), 0.0)}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------- driver
def methods():
    """(name, cf_model, impute, ssver or None, fusion or None)"""
    m = [("CBSO", None, None, 1, None),
         ("SVD", "svd", None, None, None),
         ("NMF", "nmf", None, None, None),
         ("PCA", "pca", "zero", None, None),
         ("PCA-meanimp", "pca", "mean", None, None),
         ("HSO-SVD", "svd", None, 1, "X"),
         ("HSO-NMF", "nmf", None, 1, "X")]
    for imp, tag in [("zero", ""), ("mean", "-meanimp")]:
        for sv, sn in SSMS.items():
            for fu in ["M", "X"]:
                m.append((f"P-HSO-{sn}-{fu}{tag}", "pca", imp, sv, fu))
    for sv, sn in SSMS.items():
        for fu in ["W", "PW"]:
            m.append((f"P-HSO-{sn}-{fu}", "pca", "zero", sv, fu))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "splits_full"))
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "ontology", "db_v6_all.db"))
    ap.add_argument("--nuser", type=int, default=943)
    ap.add_argument("--nitem", type=int, default=1682)
    ap.add_argument("--no-nmf", action="store_true", help="skip NMF (slow)")
    ap.add_argument("--no-normalise", action="store_true", help="ablation: fuse raw scores")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scenarios", default="0,5,10", help="e.g. 0 or 0,5,10")
    ap.add_argument("--keep-boundary", action="store_true",
                    help="also evaluate the one test user per fold whose ratings were split")
    ap.add_argument("--nmf-grid", default="2,3,5,8,10,15,20,30,50,100",
                    help="k values for NMF (slow); default is a reduced grid")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "run", "results_per_user.csv"))
    ap.add_argument("--only-models", default="", help="e.g. nmf: run only methods with these CF models")
    ap.add_argument("--require-nimfa", action="store_true", help="fail instead of using scikit-learn NMF")
    ap.add_argument("--part", default="", help="i/m: run only every m-th (condition, split), offset i; "
                    "validation splits are still drawn in the full order, so they match a complete run")
    a = ap.parse_args()
    grid = [2, 5, 10, 20] if a.quick else K_GRID
    nmf_grid = [2, 5, 10] if a.quick else [int(x) for x in a.nmf_grid.split(",")]
    meths = [m for m in methods() if not (a.no_nmf and m[1] == "nmf")]
    if a.only_models:
        meths = [m for m in meths if m[1] in a.only_models.split(",")]
    global REQUIRE_NIMFA
    REQUIRE_NIMFA = a.require_nimfa

    cb = {sv: cb_scores(a.nitem, sv, a.db) for sv in SSMS}
    pairs = {sv: cb_pairs(a.nitem, sv, a.db) for sv in SSMS}
    out, chosen = [], {}
    rng = np.random.default_rng(2026)
    job_no = -1
    for n in [int(x) for x in a.scenarios.split(",")]:
        for f in FOLDS:
            tr, te = load_fold(a.data, f, n)
            vtr, vte = inner_validation(tr, te, n, rng)
            job_no += 1
            if a.part and job_no % int(a.part.split("/")[1]) != int(a.part.split("/")[0]):
                continue
            vusers = np.unique(vte[:, 0]).astype(int)
            users = eval_users(tr, te, n, a.keep_boundary)
            V = rating_matrix(tr, a.nuser, a.nitem)
            Vv = rating_matrix(vtr, a.nuser, a.nitem)
            cbp = {sv: cb_personal(pairs[sv], tr, a.nuser, cb[sv]) for sv in SSMS}
            cbpv = {sv: cb_personal(pairs[sv], vtr, a.nuser, cb[sv]) for sv in SSMS}
            groups = {}
            for mt in meths:
                groups.setdefault((mt[1], mt[2]), []).append(mt)
            for (model, imp), ms in groups.items():
                def cf(M, k):
                    return {"pca": lambda: cf_pca(M, k, imp), "svd": lambda: cf_svd(M, k),
                            "nmf": lambda: cf_nmf(M, k)}[model]().astype(np.float32)
                grid_m = (nmf_grid if model == "nmf" else grid) if model in ("pca", "svd", "nmf") else [None]
                # ---- inner validation: one CF fit per k, shared by all methods of the group
                best = {m[0]: (grid_m[0], -1, None) for m in ms}
                if len(grid_m) > 1:
                    for k in grid_m:
                        cfv = cf(Vv, k)
                        for name, _, _, sv, fu in ms:
                            for al in (ALPHAS if fu in ("W", "PW") else [None]):
                                sc = fuse(cfv, cb[sv] if sv else None, fu, not a.no_normalise,
                                          alpha=al, cbp=cbpv[sv] if fu == "PW" else None)
                                p10 = evaluate(sc, vtr, vte, vusers).query("N==10").precision.mean()
                                if p10 > best[name][1]:
                                    best[name] = (k, p10, al)
                # ---- fit on the full training fold with the chosen k and evaluate on test
                fits = {}
                for name, _, _, sv, fu in ms:
                    k, al = best[name][0], best[name][2]
                    chosen[f"ML{n}|fold{f}|{name}"] = k
                    if al is not None:
                        chosen[f"ML{n}|fold{f}|{name}|alpha"] = al
                    if model is None:
                        sc = np.tile(cb[sv], (a.nuser, 1))
                    else:
                        if k not in fits:
                            fits[k] = cf(V, k)
                        sc = fuse(fits[k], cb[sv] if sv else None, fu, not a.no_normalise,
                                  alpha=al, cbp=cbp[sv] if fu == "PW" else None)
                    r = evaluate(sc, tr, te, users)
                    r.insert(0, "fold", f)
                    r.insert(0, "method", name)
                    r.insert(0, "scenario", f"ML{n}")
                    r["impute"], r["ssm"], r["fusion"], r["k"], r["alpha"] = imp, SSMS.get(sv), fu, k, al
                    out.append(r)
            print(f"ML{n} fold {f} done ({len(users)} test users)", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            pd.concat(out).to_csv(a.out, index=False)       # save after each fold
    json.dump(chosen, open(a.out.replace(".csv", "_chosen_k.json"), "w"), indent=1)
    print("saved", a.out, "and chosen_k.json")


if __name__ == "__main__":
    sys.exit(main())
