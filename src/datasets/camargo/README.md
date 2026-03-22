# Camargo Dataset Notes

- Raw data root: `data/camargo/raw`
- Extracted subject folders: `data/camargo/raw/subjects`
- Eurobench export root: `data/camargo/eurobench`

## Current Local Download

- `18` subject archives extracted: `AB06`, `AB07`, `AB08`, `AB09`, `AB10`,
  `AB12`, `AB13`, `AB14`, `AB16`, `AB17`, `AB19`, `AB20`, `AB21`, `AB23`,
  `AB24`, `AB27`, `AB28`, `AB30`
- Raw inventory:
  - `data/camargo/raw_inventory.csv`
  - `data/camargo/raw_inventory_summary.yaml`

## Conversion

Camargo stores most trial signals as MATLAB `table` objects. The converter uses
the Python package `mat-io` to load them.

Install dependency:

```bash
python3 -m pip install mat-io
```

Run conversion:

```bash
python3 src/datasets/camargo/convert_camargo_to_eurobench.py --overwrite
```

Useful optional exports:

```bash
python3 src/datasets/camargo/convert_camargo_to_eurobench.py \
  --save-imu --save-emg --save-gon --save-id --save-jp --save-fp --overwrite
```

## Eurobench Outputs

Each trial exports:

- `*_Trajectories.csv` from `markers`
- `*_jointAngles.csv` from `ik_offset` when available, otherwise `ik`
- `*_gaitEvents.yaml` and `*_point_gaitEvents.yaml` from `gcLeft` / `gcRight`
- `*_info.yaml` with condition metadata and source file references
