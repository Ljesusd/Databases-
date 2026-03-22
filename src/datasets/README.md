# Scripts by Dataset

This file is the working inventory to prepare repository uploads.

Status tags used here:
- `active`: part of the current documented pipeline.
- `aux`: useful script, but not required in the default pipeline.

## Shared utilities (`src/`)

- `src/detect_gait_events_markers.py` (`active`)
- `src/segment_gait_cycle.py` (`active`)
- `src/segment_gait_cycle_angles.py` (`active`)
- `src/segment_gait_cycle_marker_angles.py` (`active`)
- `src/plot_fusion_knee.py` (`active`, plotting utility)

## `src/datasets/healthypig`

- `process_subjects_trial0.py` (`active`)
- `test_all_trials_canonical.py` (`active`, QC/canonical selection)
- `plot_flattening_subjects_flexion.py` (`active`, QC visualization)
- `flag_knee_outliers.py` (`active`, QC report)
- `extract_joint_angles.py` (`aux`, helper)
- `extract_landmarks.py` (`aux`, helper)

## `src/datasets/human_gait`

- `process_human_gait.py` (`active`)
- `process_human_gait_angles.py` (`active`)
- `plot_population_profiles.py` (`active`)
- `plot_sensor_insoles_acceleration.py` (`aux`, sensor-only visualization)
- `plot_inertial_bin_acceleration.py` (`aux`, sensor-only visualization)

## `src/datasets/multimodal_walking_speeds`

- `process_multimodal_walking_speeds.py` (`active`)
- `plot_c3_joint_gait_cycles.py` (`active`)
- `plot_gait_cycle_example.py` (`aux`, quick marker signal check)

## `src/datasets/biomechanics_human_walking`

- `prepare_biomechanics_human_walking.py` (`active`)
- `convert_biomechanics_human_walking.py` (`active`)
- `estimate_gait_events_biomechanics_human_walking.py` (`active`)
- `plot_gait_cycle_biomechanics_human_walking.py` (`active`)
- `plot_gait_cycle_from_matlab_angles.py` (`active`)

## `src/datasets/multisensor`

- `convert_multisensor_c3d.py` (`active`)
- `process_multisensor_markers.py` (`active`)
- `plot_population_profiles_flexion_all_trials.py` (`active`)
- `select_subjects_closest_literature.py` (`active`)

## `src/datasets/gait_analysis_assessment`

- `convert_gait_analysis_assessment.py` (`active`)
- `analyze_gait_analysis_assessment.py` (`active`)

## `src/datasets/lower_limb_kinematic`

- `convert_lower_limb_kinematic.py` (`active`)

## `src/datasets/benchmark_bilateral_lower_limb`

- Dataset source `5362627` copied under:
  `data/benchmark_datasets_for_bilateral_lower_limb/raw/5362627`
- `convert_benchmark_bilateral_lower_limb.py` (`active`)
  Converts ABxxx processed trials (folder or zip) to Eurobench
  `*_jointAngles.csv`, `*_gaitEvents.yaml`, and `*_info.yaml`.
- `plot_benchmark_bilateral_lower_limb.py` (`active`)
  Batch plots time-series and event rasters per trial.
- `plot_benchmark_bilateral_lower_limb_gait_cycle.py` (`active`)
  Batch plots normalized gait-cycle profiles (0-100%) per trial.

## `src/datasets/running_injury_clinic_kinematic`

- `filter_healthy_subset.py` (`active`)
- `convert_running_injury_clinic_kinematic.py` (`active`)
- `analyze_running_injury_clinic_kinematic.py` (`active`)

## `src/datasets/mypredict`

- `convert_mypredict_to_eurobench.py` (`active`)
  Converts `mp*.hdf5` files into Eurobench-style `*_Trajectories.csv`,
  `*_point_gaitEvents.yaml`, `*_jointAngles.csv`, `*_gaitEvents.yaml`,
  and `*_info.yaml` files under `data/mypredict/eurobench`.
- `build_canonical_gait_trajectories.py` (`active`)
  Builds one normalized canonical gait cycle per subject under
  `data/mypredict/processed_canonical`, with a viewer-friendly
  `*_canonical_Trajectories.csv`, `*_canonical_marker_angles_norm101.csv`,
  per-cycle QC metrics, and quick-look plots. Cycle selection is template-based
  and keeps the subset that best matches canonical hip/knee/ankle gait curves.

## `src/datasets/camargo`

- `convert_camargo_to_eurobench.py` (`active`)
  Converts the Camargo MATLAB-table dataset into Eurobench-style
  `*_Trajectories.csv`, `*_jointAngles.csv`, `*_gaitEvents.yaml`,
  `*_point_gaitEvents.yaml`, and `*_info.yaml` files under
  `data/camargo/eurobench`. Requires the Python package `mat-io` to decode
  MATLAB `table` files.
