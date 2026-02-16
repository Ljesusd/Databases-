from pathlib import Path
import sys

import numpy as np


JOINT_CHANNELS = {
    "hip": "RHipAngles",
    "knee": "RKneeAngles",
    "ankle": "RAnkleAngles",
}


def extract_joint_angles(c3d_path: str, joints=("hip", "knee", "ankle"), scale=None):
    import ezc3d

    c3d = ezc3d.c3d(c3d_path)
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    point_scale = c3d["parameters"]["POINT"]["SCALE"]["value"][0]

    if scale is None:
        scale = 1.0 if point_scale < 0 else point_scale

    rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    n_frames = c3d["data"]["points"].shape[2]
    time = np.arange(n_frames) / rate

    angles = {}
    for joint in joints:
        label = JOINT_CHANNELS[joint]
        idx = labels.index(label)
        data = c3d["data"]["points"][:3, idx, :].T * scale
        angles[joint] = data

    return time, angles


def main():
    c3d_path = "data/_test_c3d/SUBJ1 (0).c3d"
    time, angles = extract_joint_angles(c3d_path)
    print(time.shape)
    for k, v in angles.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
