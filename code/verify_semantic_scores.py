"""Recompute every pairwise similarity and the Eq. (13) scores with ssmpy and compare them with
the cached scores used by pipeline.py (cache/semantic_scores). Takes about 10-20 minutes.

    python code/verify_semantic_scores.py
"""
import os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "semantic_scores")
import ssmpy

ssmpy.semantic_base(os.path.join(ROOT, "data", "ontology", "db_v6_all.db"))
ssmpy.ssm.mica = False          # DiShIn shared IC
ssmpy.ssm.intrinsic = False     # extrinsic IC (Eq. 7)
ids = {i: ssmpy.get_id(str(i)) for i in range(1, 1683)}
for name, fn in (("Res", ssmpy.ssm_resnik), ("Lin", ssmpy.ssm_lin), ("JC", ssmpy.ssm_jiang_conrath)):
    P = np.zeros((1682, 1682))
    for i in range(1, 1683):
        for j in range(i + 1, 1683):
            P[i-1, j-1] = P[j-1, i-1] = fn(ids[i], ids[j])
    s = P.sum(1) / 1681
    old_s = np.load(os.path.join(CACHE, f"scb_all_{name}.npy"))
    old_P = np.load(os.path.join(CACHE, f"scb_pairs_{name}.npy")).astype(float)
    np.fill_diagonal(old_P, 0)
    print(f"{name}: max |pairwise difference| {np.abs(P - old_P).max():.1e}, "
          f"max |Eq. (13) difference| {np.abs(s - old_s).max():.1e}", flush=True)
