"""Calendar marks compatible with FAN, without its package dependencies.

Feature order and arithmetic follow FAN/torch_timeseries/utils/timefeatures.py
(see LICENSES/README.txt). Continuous marks are required by
FEDformer and the corrected Autoformer timeF contract. Only the current
profiles' hourly, minutely and daily frequencies are exposed.
"""

import numpy as np
import pandas as pd


def time_features(dates, timeenc=1, freq="h"):
    """Return [rows, features] marks; integer minutes are 15-minute buckets.

    Accept a DataFrame with a ``date`` column or a datetime-like sequence.
    Integer marks are int64, continuous marks float64, as in the old loader.
    """
    values = dates["date"].values if isinstance(dates, pd.DataFrame) else dates
    index = pd.DatetimeIndex(pd.to_datetime(values))
    freq = freq.lower()
    if freq not in ("h", "t", "d"):
        raise ValueError(f"Unsupported mark frequency: {freq!r}; expected h, t or d")
    if timeenc == 0:
        calendar = (index.month, index.day, index.dayofweek, index.hour,
                    index.minute // 15)
        width = {"d": 3, "h": 4, "t": 5}[freq]
        return np.vstack(calendar[:width]).T.astype(np.int64)
    if timeenc != 1:
        raise ValueError("timeenc must be 0 (calendar) or 1 (continuous)")

    day = (index.day - 1) / 30.0 - 0.5
    year = (index.dayofyear - 1) / 365.0 - 0.5
    weekday = index.dayofweek / 6.0 - 0.5
    hour = index.hour / 23.0 - 0.5
    minute = index.minute / 59.0 - 0.5
    if freq == "d":
        features = [weekday, day, year]
    elif freq == "h":
        features = [hour, weekday, day, year]
    else:
        features = [minute, hour, weekday, day, year]
    return np.vstack(features).T
