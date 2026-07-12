#!/usr/bin/env python
"""Rewrite a PAI `features.csv` from the export-era schema to the schema expected
by `physical_ai_av` >= 0.2.2 (`physical_ai_av.dataset.Features`), which the
Alpamayo 1.5 recipe loader (`alpamayo.data.pai_utils`) uses.

Old schema (written by scripts/export_py123d_to_pai.py):
    columns: filename_template, chunk_filename_template, clip_files_in_zip
      * chunk_filename_template uses `{chunk:04d}`
      * clip_files_in_zip maps clip_id -> {inner_key: <actual filename>}

New schema (physical_ai_av 0.2.2):
    columns: directory, chunk_path, clip_files_in_zip
      * groupby("directory") builds feature namespaces
      * chunk_path uses `{chunk_id}` and is passed to str.format(chunk_id=...)
      * clip_files_in_zip is ONE template dict per feature: {inner_key: "{clip_id}...."}

Only metadata is rewritten; the on-disk chunk zips are untouched. The original is
backed up to features.csv.oldschema (unless it already exists).

    python scripts/aurora/convert_features_csv_pai022.py --pai-dir data/carla_alpamayo_pai
"""
import argparse
import json
import os
import shutil

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pai-dir", required=True)
    ap.add_argument("--features", default="features.csv")
    args = ap.parse_args()

    path = os.path.join(args.pai_dir, args.features)
    df = pd.read_csv(path, index_col="feature")

    directories, chunk_paths, templates = [], [], []
    for feat in df.index:
        row = df.loc[feat]
        # directory = leading path component of the feature's chunk file (labels/ or camera/)
        chunk_tmpl = str(row["chunk_filename_template"])
        directories.append(chunk_tmpl.split("/")[0])
        # {chunk:04d} / {chunk} -> {chunk_id:04d} / {chunk_id}
        chunk_paths.append(chunk_tmpl.replace("{chunk:", "{chunk_id:").replace("{chunk}", "{chunk_id}"))
        # per-clip actual filenames -> one template dict by substituting the clip_id
        cfiz = row["clip_files_in_zip"]
        cfiz = json.loads(cfiz) if isinstance(cfiz, str) else cfiz
        first_clip = next(iter(cfiz))
        inner = cfiz[first_clip]
        tmpl = {k: v.replace(first_clip, "{clip_id}") for k, v in inner.items()}
        templates.append(json.dumps(tmpl))

    out = pd.DataFrame(
        {"directory": directories, "chunk_path": chunk_paths, "clip_files_in_zip": templates},
        index=df.index,
    )
    out.index.name = "feature"

    backup = path + ".oldschema"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"backed up original -> {backup}")
    out.to_csv(path)
    print(f"wrote {len(out)} features in physical_ai_av>=0.2.2 schema -> {path}")
    for feat in out.index:
        print(f"  {feat}: dir={out.at[feat,'directory']!r} chunk_path={out.at[feat,'chunk_path']!r}")
        print(f"      clip_files_in_zip={out.at[feat,'clip_files_in_zip']}")


if __name__ == "__main__":
    main()
