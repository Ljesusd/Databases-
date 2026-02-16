import pandas as pd

LANDMARK_COLUMNS = {
    "hip": ["RTHI_x", "RTHI_y", "RTHI_z"],
    "knee": ["RKNE_x", "RKNE_y", "RKNE_z"],
    "ankle": ["RANK_x", "RANK_y", "RANK_z"],
}


def extract_landmarks(csv_path, landmarks=("hip", "knee", "ankle")):
    """
    Extract 3D trajectories for selected landmarks from a Eurobench CSV.

    Parameters
    ----------
    csv_path : str
        Path to Eurobench Trajectories CSV.
    landmarks : tuple
        Landmarks to extract: any of ("hip", "knee", "ankle").

    Returns
    -------
    time : ndarray, shape (N,)
        Time vector.
    data : dict
        Keys = landmark names
        Values = ndarray of shape (N, 3)
    """
    df = pd.read_csv(csv_path)

    time = df["time"].values
    data = {}

    for lm in landmarks:
        cols = LANDMARK_COLUMNS[lm]
        data[lm] = df[cols].values

    return time, data
