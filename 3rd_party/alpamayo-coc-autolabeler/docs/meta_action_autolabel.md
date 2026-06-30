# Meta-Action Autolabeling Pipeline

This page is the detailed reference for Step 1 of the main README workflow. It
documents the Physical-AI dataset input layout, command arguments, output format,
worker guidance, and optional visualization commands for meta-action autolabeling.

## Prepare the Data (Physical-AI AV Dataset)

Use the Physical-AI AV dataset prepared in the [main README](../README.md).

Expected inputs for meta-action autolabeling include:

- `data_dir` for parquet clip data. Example, `/path/to/physical_ai_data`

Example structure for `data_dir` (trajdata parquet inputs):

```text
/path/to/physical_ai_data/
├── labels/
│   ├── egomotion/
│       ├── egomotion.<chunk_id>.zip
│       ├── egomotion.<chunk_id>.zip
│       ├── egomotion.<chunk_id>.zip
│       └── ...
```

## Run the Meta-Action Autolabeling Pipeline

Run the meta-action command shown in the main README. No preprocessing is
required for the egomotion zip files. The dataloader reads egomotion directly
from:

```text
/path/to/physical_ai_data/labels/egomotion/*.zip
```

The command uses these arguments:

- `--dataset_name`: trajdata dataset key. Use `pai` for the Physical-AI AV
  dataset.
- `--meta_action_names`: meta-action types to generate. Use `all_ego` for the
  default ego-motion set.
- `--data_dir`: root directory of the prepared Physical-AI AV dataset.
- `--cache_dir`: trajdata cache.
- `--save_dir`: output root for raw and post-processed meta-action results.
- `--num_workers`: number of CPU workers used for dataset loading and clip
  processing.

Expected outputs for meta-action autolabeling include:

- `cache_dir`: formatted trajectory data cache.
- `save_dir`: meta-action outputs.

Note: This path supports ego-only workflows (e.g., `all_ego`) and does not
need maps.

This command produces:

```text
/path/to/meta_action/resultdir/
├── tmp/
│   └── raw_results/
│       ├── <clip_id_1>.json
│       ├── <clip_id_2>.json
│       └── ...
└── final_outputs/
    ├── <clip_id_1>.txt
    ├── <clip_id_2>.txt
    └── ...
```

- `tmp/raw_results/*.json`: intermediate raw meta-action strings.
- `final_outputs/*.txt`: formatted final outputs generated from raw strings.

If you maintain a clip list file (for example, `clipid.txt`) in `save_dir`,
you can reuse it for visualization via `--clip_list_path`.

Each line in `final_outputs/<clip_id>.txt` follows:

```text
<MetaActionName> - Agent:<agent_id>, Start:<start_ts>, End:<end_ts>
```

Example:

```text
  GentleAcceleration - Agent:<ego>, Start:0, End:35
  GoStraight - Agent:<ego>, Start:0, End:67
  GoStraight - Agent:<ego>, Start:103, End:125
  MaintainSpeed - Agent:<ego>, Start:35, End:53
  MaintainSpeed - Agent:<ego>, Start:69, End:105
  StrongDeceleration - Agent:<ego>, Start:53, End:69
  StrongDeceleration - Agent:<ego>, Start:116, End:200
  SteerLeft - Agent:<ego>, Start:67, End:84
  SteerLeft - Agent:<ego>, Start:125, End:151
  SteerRight - Agent:<ego>, Start:84, End:103
  GentleDeceleration - Agent:<ego>, Start:105, End:116
  SharpSteerLeft - Agent:<ego>, Start:151, End:200
```

Higher `num_workers` improves throughput, subject to CPU limits. For larger
machines and datasets, typical values are 32, 64, or 128.

## Visualize Meta Actions on Videos

After generating meta-action labels, render visualizations with:

Example command for a small test chunk of the PAI dataset:

```bash
python scripts/vis_video_pai.py \
  --physical_ai_root /path/to/physical_ai_data \
  --meta_action_dir /path/to/meta_action/resultdir/final_outputs \
  --vis_dir /path/to/meta_action/vis_dir \
  --cache_dir /path/to/trajdata_cache/pai \
  --work_root /path/to/meta_action/work_root \
  --num_workers 8
```

Arguments:

- `--physical_ai_root`: root directory of the prepared Physical-AI AV dataset.
- `--meta_action_dir`: `final_outputs` directory produced by meta-action
  autolabeling.
- `--vis_dir`: output directory for rendered visualization videos.
- `--cache_dir`: trajdata cache for the PAI dataset.
- `--work_root`: working directory for prepared intermediate video inputs.
- `--num_workers`: number of CPU workers used for visualization preparation.

An example visualization output is shown in
[`../assets/vis_results.png`](../assets/vis_results.png).

## License

This project is licensed under the [Apache-2.0](../LICENSE) License.
