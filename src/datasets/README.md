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
