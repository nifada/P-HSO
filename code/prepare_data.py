"""Rebuild the 30 split files used in the paper from the original MovieLens 100K ratings.

MovieLens 100K may not be redistributed without permission from GroupLens, so this repository
contains only the test-set membership of every split (user::movie pairs, no ratings). The
ratings are taken from your own copy of ml-100k/u.data:

    python code/prepare_data.py --udata path/to/ml-100k/u.data

Output: data/splits_full/training-fold{f}-rating{h}.txt and tes-fold{f}-rating{h}.txt
(format user::movie::rating, sorted by user and movie, Windows line endings), exactly the files used in the
experiments. Every file is checked against data/splits/checksums.sha256 (checksums of the file content,
so line endings do not matter).
"""
import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--udata", required=True, help="path to ml-100k/u.data")
    ap.add_argument("--keys", default=os.path.join(ROOT, "data", "splits"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "splits_full"))
    a = ap.parse_args()

    ratings = {}
    for line in open(a.udata):
        if line.strip():
            u, i, r, _ = line.split("\t")
            ratings[(int(u), int(i))] = int(r)
    if len(ratings) != 100000:
        sys.exit(f"expected 100,000 ratings in u.data, found {len(ratings)}")
    order = sorted(ratings)

    expected = {}
    for line in open(os.path.join(a.keys, "checksums.sha256")):
        if line.strip() and not line.startswith("#"):
            h, name = line.split()
            expected[name] = h

    os.makedirs(a.out, exist_ok=True)
    bad = 0
    for n in (0, 5, 10):
        for f in range(1, 6):
            test = {tuple(map(int, l.split("::"))) for l in
                    open(os.path.join(a.keys, f"test-keys-fold{f}-rating{n}.txt")) if l.strip()}
            files = {f"tes-fold{f}-rating{n}.txt": [k for k in order if k in test],
                     f"training-fold{f}-rating{n}.txt": [k for k in order if k not in test]}
            for name, keys in files.items():
                text = "".join(f"{u}::{i}::{ratings[(u, i)]}\r\n" for u, i in keys)   # original files use Windows line endings
                open(os.path.join(a.out, name), "w", newline="").write(text)
                content = "".join(f"{u}::{i}::{ratings[(u, i)]}\n" for u, i in keys)   # line endings ignored
                ok = hashlib.sha256(content.encode()).hexdigest() == expected[name]
                bad += not ok
                print(f"{name:32s} {len(keys):6d} ratings  {'OK' if ok else 'CHECKSUM MISMATCH'}")
    if bad:
        sys.exit(f"{bad} file(s) do not match the published checksums")
    print(f"all 30 files written to {a.out} and verified")


if __name__ == "__main__":
    main()
