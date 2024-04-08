import pandas as pd
from sklearn import preprocessing
import math


def train_val_test_split_simple(df_in, ratio=(0.7, 0.2, 0.1)):
    assert math.isclose(sum(ratio), 1)
    tr, val, tst = ratio
    val_start = tr + val
    n = len(df_in)
    return df_in[0:int(n * tr)].copy(), df_in[int(n * tr):int(n * val_start)].copy(), df_in[int(n * val_start):].copy(), n


def train_val_split_simple(df_in, ratio=(0.8, 0.2)):
    assert math.isclose(sum(ratio), 1)
    tr, _ = ratio
    n = len(df_in)
    return df_in[0:int(n * tr)].copy(), df_in[int(n * tr):].copy(), n


AGGREGATION_MAPPING = {
    'Glucose': pd.Series.mean,
    'Rapid Insulin': pd.Series.sum,
    'Rapid Insulin basal': pd.Series.sum,
    'Rapid Insulin sub_tissue': pd.Series.sum,
    'Rapid Insulin blood_plasma': pd.Series.sum,
    'Rapid Insulin RIOB': pd.Series.sum,
    'Rapid Insulin IOB': pd.Series.sum,
    'Long Insulin': pd.Series.sum,
    'Carbohydrates': pd.Series.sum,
    'Carbohydrates gut': pd.Series.sum,
    'Carbohydrates plasma': pd.Series.sum,
    'Glycemic Load': pd.Series.sum,
    "GI": pd.Series.max,
    'bpm': pd.Series.mean,
    'distance': pd.Series.sum,
    'calories': pd.Series.sum,
    'Hour': pd.Series.mean,
    'Minute': pd.Series.mean,
    "Day of Month": pd.Series.mean,
    "Day of Year": pd.Series.mean,
    'protein': pd.Series.sum,
    'carbohydrate': pd.Series.sum,
    'fat': pd.Series.sum,
    'sugar': pd.Series.sum,
    'energy': pd.Series.sum,
    'finger_stick': pd.Series.mean,
    'steps': pd.Series.sum,
    'steps_ma': pd.Series.sum,
    "Glucose missing": pd.Series.mean,
}


def resample_data(in_df: pd.DataFrame, use_features: list, min_freq=15, copy=True) -> pd.DataFrame:
    if copy:
        in_df = in_df.copy()
    resampled_df = in_df.set_index('Time').resample(f'{min_freq}T').agg(
        {feature: AGGREGATION_MAPPING[feature] for feature in use_features}).reset_index()
    return resampled_df


def interpolate_gaps(in_df: pd.DataFrame, method="linear", limit=None) -> None:
    if method in ["spline", "polynomial"]:
        in_df["Glucose"] = in_df["Glucose"].interpolate(method=method, order=2, limit=limit)
        if "bpm" in in_df.columns:
            in_df["bpm"] = in_df["bpm"].interpolate(method=method, order=2, limit=limit)
    else:
        in_df["Glucose"] = in_df["Glucose"].interpolate(method=method, limit=limit)
        if "bpm" in in_df.columns:
            in_df["bpm"] = in_df["bpm"].interpolate(method=method, limit=limit)


def interpolate_gaps_up_to(in_df: pd.DataFrame, limit: int):
    """Will only interpolate the gap if it has less NaNs than the limit"""
    mask = in_df.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    mask["Glucose"] = (grp.groupby("Glucose")['ones'].transform('count') <= limit) | in_df["Glucose"].notnull()
    in_df["Glucose"] = in_df["Glucose"].interpolate(method="linear", limit=limit)[mask["Glucose"]]


def min_max_normalize(train_df, val_df, test_df, features):
    min_max_scaler = preprocessing.MinMaxScaler()
    train_df[features] = min_max_scaler.fit_transform(train_df[features])

    val_df[features] = min_max_scaler.transform(val_df[features])
    test_df[features] = min_max_scaler.transform(test_df[features])
    return min_max_scaler


class InsulinExpDecayFunction:
    # Source: https://github.com/LoopKit/Loop/issues/388

    def __init__(self, sampling_freq, hours_duration: int = 5, peak_activity_min: int = 55):
        self.sampling_freq = sampling_freq
        self.hours_duration = hours_duration
        self.peak_activity_min = peak_activity_min

        # Time duration [minutes] , 3-6 hours for Fiasp
        self.td = hours_duration * 60
        # Peak activity time [minutes], 45-85 minutes
        self.tp = peak_activity_min

        # Time constant of exponential decay
        self.tau = self.tp * (1 - self.tp / self.td) / (1 - 2 * self.tp / self.td)
        # Rise time factor
        self.a = 2 * self.tau / self.td
        # Auxiliary scale factor
        self.S = 1 / (1 - self.a + (1 + self.a) * math.exp(-self.td / self.tau))

    # Insulin activity function of time
    # Ia(t) = (S/tau^2)*t*(1-t/td)*exp(-t/tau)
    def insulin_activity(self, t):
        return (self.S / self.tau ** 2) * t * (1 - t / self.td) * math.exp(-t / self.tau)

    # Insulin on board function
    # IOB(t) = 1-S*(1-a)*((t^2/(tau*td*(1-a)) - t/tau - 1)*exp(-t/tau)+1)
    def insulin_on_board(self, t):
        return (
                1 - self.S * (1 - self.a) *
                ((t ** 2 / (self.tau * self.td * (1 - self.a)) - t / self.tau - 1) * math.exp(-t / self.tau) + 1)
        )

    def insulin_on_board_vector(self, iob_window, dose):
        w_len = len(iob_window)
        result = [current_iob for current_iob in iob_window]
        for i in range(w_len):
            result[i] += dose * self.insulin_on_board(i * self.sampling_freq)
        return result
