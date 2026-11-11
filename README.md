# P-HSO: PCA-based Hybrid Semantic Ontology movie recommendation

Code, data splits, ontology and results for the paper

> N. Ifada, E. W. Cahyaningrum, F. H. Rachman, and D. A. Dewi, "Evaluating Fusion Strategies and
> Ontology-Based Semantic Similarity Measures in PCA-Based Hybrid Movie Recommendation Systems under
> Cold-Start Conditions," *JOIV: International Journal on Informatics Visualization* (under review).

The repository reproduces every number, table and figure of the results section: Tables III–VIII,
Figs. 3–4, the average ranks, the paired tests and the imputation ablation.

## Contents

```
code/
  prepare_data.py            rebuild the 30 split files from the MovieLens 100K
  pipeline.py                all methods and the evaluation protocol -> per-user results
  stats_analysis.py          mean, SD, bootstrap CIs, mixed-design ANOVA, paired Wilcoxon tests
  make_tables.py             values of Tables III-VIII, fusion weights, ablation, average ranks
  make_fig3.py, make_fig4.py Fig. 3 (all methods) and Fig. 4 (fusion x similarity measure)
  verify_semantic_scores.py  recompute all semantic similarities with ssmpy and compare
  compat/old_numpy_names.py  lets nimfa 1.4.0 run under NumPy 2 (see Notes)
data/
  splits/                    test-set membership of every split (user::movie, no ratings) + checksums
  ontology/db_v6_all.db      movie-genre ontology as an ssmpy semantic base
  ontology/OntologiMovieAll.csv  genre;movie pairs used to build the ontology
results/
  results_per_user.csv       every metric for every method, condition, split, list size and test user
  chosen_k.json              selected k (and fusion weight alpha) per method, condition and split
  table_mean_ci.csv, table_anova.csv, table_pairwise.csv, tables.json
  run_results.xlsx  all tables in one workbook
  figures/fig3.png, figures/fig4.png
```

## Data

MovieLens 100K may not be redistributed without permission from GroupLens, so this repository
contains no ratings. Download `ml-100k.zip` from <https://grouplens.org/datasets/movielens/100k/>,
unzip it, and rebuild the exact split files used in the paper:

```bash
python code/prepare_data.py --udata path/to/ml-100k/u.data
```

This writes `data/splits_full/` (30 files: `training-fold{f}-rating{h}.txt` and
`tes-fold{f}-rating{h}.txt` for h = 0, 5, 10 and f = 1, ..., 5) and checks each file against
`data/splits/checksums.sha256`.

**Splits.** In each of five random splits, test users are drawn at random until their ratings make up
20% (20,000) of all ratings; each test user keeps h random training ratings (h = 0, 5, 10 for ML0, ML5,
ML10) and the rest form the test set; all other users keep all ratings in training. The splits were
generated without a fixed random seed, so they are distributed as membership files rather than
regenerated. Because the test set is filled until it
holds exactly 20,000 ratings, only part of the ratings of the last test user drawn in each split are
moved to the test set; this user keeps more than h training ratings and is therefore excluded from
evaluation (`--keep-boundary` includes it).

**Ontology.** `db_v6_all.db` contains the 1,682 movies as instances of 20 genre classes: the 18
MovieLens genres described in the paper, plus `unknown` and `Short` (one movie each). Information content is extrinsic: each movie
is counted once and the frequency of a genre is the number of its movies plus one (Eq. 7); no ratings
are used.

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. data (see above)
python code/prepare_data.py --udata path/to/ml-100k/u.data

# 2. all methods, one run per cold-start condition (the seed of the validation users is per run)
python code/pipeline.py --scenarios 0  --out results/run/results_ML0.csv
python code/pipeline.py --scenarios 5  --out results/run/results_ML5.csv
python code/pipeline.py --scenarios 10 --out results/run/results_ML10.csv

# 3. combine, then statistics, tables and figures
python -c "import pandas as pd, json, glob; pd.concat(pd.read_csv(f) for f in sorted(glob.glob('results/run/results_ML*[0-9].csv'))).to_csv('results/run/results_per_user.csv', index=False); json.dump({k: v for f in sorted(glob.glob('results/run/results_ML*_chosen_k.json')) for k, v in json.load(open(f)).items()}, open('results/run/chosen_k.json', 'w'), indent=1)"
cd results/run
python ../../code/stats_analysis.py results_per_user.csv
python ../../code/make_tables.py
python ../../code/make_fig3.py table_mean_ci.csv fig3.png
python ../../code/make_fig4.py table_mean_ci.csv fig4.png
```

The first run computes the semantic similarities of all movie pairs with ssmpy and caches them in
`cache/semantic_scores/` (about 20 minutes). NMF with nimfa (2,000 iterations for every candidate k)
dominates the run time; expect several hours in total. `--no-nmf` skips NMF and HSO-NMF and
`--quick` runs a small k grid as a smoke test (both change the tie-breaking of CBSO; see below).

To recompute the statistics, tables and figures from the published per-user results only:

```bash
cd results
python ../code/stats_analysis.py results_per_user.csv
python ../code/make_tables.py
```

This reproduces `table_mean_ci.csv`, `table_anova.csv`, `table_pairwise.csv` and `tables.json` exactly.

## Where each result comes from

| Paper | File |
|---|---|
| Table III (selected k) | `chosen_k.json`; sheet "Table III k" |
| Tables IV and V (mean ± SD) | `table_mean_ci.csv`; sheets "Table IV Precision", "Table V NDCG" |
| Table VI (average rank per condition and overall) | sheet "Table VI Average rank"; per-cell ranks in sheet "Average rank" |
| Table VII (paired Wilcoxon tests at Top-10) | `table_pairwise.csv`; sheet "Table VII Wilcoxon Top-10" |
| Table VIII (mixed-design ANOVA) | `table_anova.csv`; sheets "Table VIII ANOVA", "ANOVA full" |
| Fig. 3 (all methods at Top-10) and Fig. 4 (P-HSO by fusion strategy and similarity measure), with 95% bootstrap CIs | `table_mean_ci.csv`; `figures/fig3.png`, `figures/fig4.png` |
| Average ranks (Section III.A) | sheet "Average rank": rank of each method in each of the 24 combinations of condition, list size and metric (ties receive the average of their ranks) and their mean |
| Paired Wilcoxon tests (Section III.A) | `table_pairwise.csv`: best P-HSO configuration per condition, list size and metric against each baseline; Holm correction within each group of baseline comparisons (six; five in ML0, where the best P-HSO is identical to PCA); effect size `r_rb` |
| Fusion weights (Section III.C) | `chosen_k.json` (`...|alpha`); sheet "Fusion weight alpha" |
| Imputation ablation (Section III.A) | sheet "Ablation imputation" |

Precision and NDCG are on the 0–1 scale. In `results_per_user.csv`, `ndcg` follows Eqs. (21)–(23)
of the paper; `ndcg_std` uses the standard IDCG over min(|G_u|, N) and is provided for reference.

## Equations and code

| Equation | Code (`code/pipeline.py`) |
|---|---|
| (1)–(6) centring, covariance, eigenvectors, projection, reconstruction | `cf_pca` (scikit-learn `PCA`, which obtains the eigenvectors from the SVD of X) |
| (7)–(12) IC, shared IC, Resnik, Lin, Jiang-Conrath | ssmpy with `mica=False` (DiShIn) and `intrinsic=False` (extrinsic IC), in `cb_scores` / `cb_pairs` |
| (13) item-level semantic score | `cb_scores` |
| (14)–(15) min-max normalization | `minmax_rows`, `minmax_vec` (a user whose CF scores are all equal gets 0) |
| (16)–(18) multiplicative, mean, weighted fusion | `fuse` |
| (19) personalized semantic score | `cb_personal` |
| (20)–(23) Precision, DCG, IDCG, NDCG | `evaluate` |

Baselines: SVD is the truncated SVD of the zero-filled, uncentred matrix (`cf_svd`); NMF uses nimfa
with NNDSVD seeding, Euclidean updates and 2,000 iterations (`cf_nmf`); HSO-SVD and HSO-NMF combine
them with the Resnik score by multiplicative fusion.

## How the published results were produced

The published `results_per_user.csv` combines three runs of `pipeline.py`, each started once per
condition as above:

1. all methods;
2. NMF and HSO-NMF with nimfa (`--only-models nmf --require-nimfa`), replacing the NMF results of run 1;
3. HSO-SVD in ML0 after the rule for users whose CF scores are all equal was added to Eq. (14),
   replacing the ML0 HSO-SVD results of run 1 (the rule does not affect any other method or condition).

Ties are broken at random, with one random-number generator per run (seed 2026). A fresh run of
all methods per condition reproduces the published values except those of SVD, NMF, HSO-SVD and
HSO-NMF in ML0: these models have no information about the test users, so their ranking there is
random (and their selected k in ML0 is arbitrary). As a check, a fresh ML10 run of all methods
except NMF reproduced every per-user value, selected k and fusion weight of 22 of the 23 ML10
methods exactly; the exception, CBSO, is explained below.

CBSO gives every user the same scores, and movies with the same genres share a score (194 distinct
scores among 1,682 movies), so its results depend on the tie-breaking. Options that skip models
(`--no-nmf`, `--only-models`) change the random numbers used in later splits and therefore change
CBSO slightly (in the check above, Precision@10 in ML10 was 0.2101 instead of 0.2093).

## Notes

* **nimfa and NumPy 2.** nimfa 1.4.0 (2019) uses NumPy names that NumPy 2 removed (`np.mat` and
  others). `code/compat/old_numpy_names.py` restores them as aliases of their documented replacements
  before nimfa is imported; nimfa itself is unchanged. `--require-nimfa` makes the pipeline stop
  instead of falling back to scikit-learn's NMF when nimfa is missing.
* **Semantic scores.** `verify_semantic_scores.py` recomputes all similarities with ssmpy; the cached
  scores agree with ssmpy to within float32 rounding (Eq. 13 scores identically).

## Citation

If you use this code, please cite the paper above and the original HSO study:

> N. Ifada, E. W. Cahyaningrum, and F. H. Rachman, "The Implementation of Hybrid Semantic
> Ontology-based Model on Movie Recommendation System," in *Proc. CENIM 2022*, pp. 1–7,
> doi: 10.1109/CENIM56801.2022.10037277.

MovieLens 100K: F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context,"
*ACM TiiS*, vol. 5, no. 4, 2015, doi: 10.1145/2827872.
