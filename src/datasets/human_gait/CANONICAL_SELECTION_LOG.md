# human_gait canonical selection log

Date: 2026-03-18

## Scope

This note records the current canonical gait selection setup for `human_gait`.

The canonical profiles are generated with:

- script: `src/datasets/build_canonical_gait_profiles.py`
- dataset selector: `human_gait`
- input root: `data/human_gait/eurobench`
- outputs:
  - `data/human_gait/processed_canonical`
  - `data/human_gait/plots_canonical`

## Current selection settings

Current `human_gait` settings inside `build_canonical_gait_profiles.py`:

- `human_gait_keep_percentile = 40.0`
- `min_stride_s = 0.5`
- `max_stride_s = 2.0`
- `require_toe_off = True`

The selector works per `subject + task`, not across the whole dataset at once.

Tasks currently included:

- `Gait`
- `FastGait`
- `SlowGait`
- `2minWalk`

## Selection criterion

Each candidate gait cycle is scored against canonical sagittal templates for:

- hip flexion
- knee flexion
- ankle dorsiflexion

The selector is robust to:

- sign flips
- angle wrapping / unwrapping
- constant angular offsets
- dataset-specific marker definitions for right lower limb landmarks

For each accepted cycle:

1. trajectories are aligned to a local gait frame
2. multiple hip / knee / ankle definitions are tested
3. the best-scoring definition per joint is selected
4. the top `40%` lowest-score cycles are retained for the canonical profile

## Current result summary

Source summary:

- `data/human_gait/processed_canonical/human_gait_canonical_groups_summary.csv`

Current totals after the `40%` setting:

- groups processed: `22`
- groups with status `ok`: `22`
- candidate cycles: `191`
- selected cycles: `82`

Distribution of selected cycles per group:

- `1` cycle: `4` groups
- `2` cycles: `5` groups
- `3` cycles: `7` groups
- `5` cycles: `1` group
- `7` cycles: `2` groups
- `9` cycles: `2` groups
- `10` cycles: `1` group

Important: some groups still keep only `1` cycle because they have very few valid candidates to begin with. This is expected for some `FastGait` groups.

## Plot interpretation

Canonical plots are written to `data/human_gait/plots_canonical`.

Meaning of the lines:

- gray dashed lines: selected cycles
- teal dashed line: reference template used for scoring
- black solid line: canonical median of the selected cycles

The plot style was updated so selected cycles remain visible even when only a few are present.

## Main output files per group

For each `subject/task` group the pipeline writes:

- `*_canonical_marker_angles_norm101.csv`
- `*_canonical_Trajectories.csv`
- `*_canonical_cycle_metrics.csv`
- `*_canonical_summary.yaml`
- `*_canonical_profiles.png`

## Examples

- `data/human_gait/processed_canonical/Gait/P01_S01/P01_S01_Gait_canonical_marker_angles_norm101.csv`
- `data/human_gait/processed_canonical/Gait/P01_S01/P01_S01_Gait_canonical_Trajectories.csv`
- `data/human_gait/plots_canonical/Gait/P01_S01/P01_S01_Gait_canonical_profiles.png`

## Follow-up options

If needed later, the next reasonable adjustments are:

- add `min_keep = 2` or `min_keep = 3` for `human_gait`
- remove the template line from plots
- export a population-level canonical profile across all `human_gait` groups
