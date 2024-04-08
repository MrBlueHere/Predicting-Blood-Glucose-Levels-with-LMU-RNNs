import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import cached_property
from typing import Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
import copy


class SubWindow:
    def __init__(self, start: int, end: Optional[int], window_size: int):
        self.slice = slice(start, end)
        self.indices = np.arange(window_size)[self.slice]


@dataclass
class Noise:
    feature_name: str
    rate: float
    magnitude: float


@dataclass
class NoiseConfig:
    noise: list[Noise]


class CustomDataset(Dataset):
    def __init__(self, np_windows: list, window_generator):
        self.data = np_windows
        self.window_generator = window_generator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data[index]
        return self.window_generator.split_window_torch(torch.tensor(features))


class WindowGenerator:
    """ Inspiration: https://www.tensorflow.org/tutorials/structured_data/time_series """

    NOISY_COLUMNS = ["Rapid Insulin", "Carbohydrates"]

    def __init__(
            self,
            input_width: int,
            label_width: int,
            shift: int,
            train_df_in: pd.DataFrame, val_df_in: pd.DataFrame, test_df_in: pd.DataFrame,
            batch_size: int = 32,
            drop_noisy=False,
            noisy_period: int = None,
            use_columns: [] = None,
            label_columns: [] = None,
            look_ahead_columns: [] = None,
            noisify_configs: NoiseConfig = None,
            shuffle: bool = False,
            provider="tf",
            gaussian_sigma=None,
            interpolate=True,
    ):
        """
        Provides data windowing and plotting capabilities
        :param input_width: Size of the input window
        :param label_width: Size of the output window (single step or multiple)
        :param shift: The prediction horizon
        :param train_df_in:
        :param val_df_in:
        :param test_df_in:
        :param batch_size:
        :param drop_noisy: If true, windows are trimmed off when there is noise between current time and the predicted time step
        :param noisy_period: Configurable noisy period, if None, it is same as the shift
        :param use_columns: Columns to pick to be used from the passed dataframes
        :param look_ahead_columns: Columns whose future values can be used as inputs, the input format is (feature, offset)
        where offset is how many steps into the future can this feature be used
        """
        if isinstance(input_width, list):
            assert len(input_width) == len(
                [c for c in use_columns if c != "Time"]), "Input widths and number of columns to use must match"

        if look_ahead_columns is None:
            look_ahead_columns = []
        else:
            # Make sure we don't look more into the future than the shift
            assert all(offset <= shift for _, offset in look_ahead_columns)
        if use_columns is None:
            use_columns = ["Time", "Glucose"]
        self.label_columns = label_columns or ["Glucose"]

        self.noisify_configs = noisify_configs
        self.shuffle = shuffle
        self.provider = provider
        self.gaussian_sigma = gaussian_sigma
        self.interpolate = interpolate

        self.train_df, self.val_df, self.test_df = train_df_in, val_df_in, test_df_in

        self.batch_size = batch_size
        if isinstance(drop_noisy, tuple):
            self.train_drop_noisy, self.val_drop_noisy, self.test_drop_noisy = drop_noisy
        else:
            self.train_drop_noisy = self.val_drop_noisy = self.test_drop_noisy = drop_noisy
        self.noisy_period = noisy_period
        self.using_columns = [col for col in use_columns if col != "Time"]

        # Work out the label column indices
        self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(self.train_df[self.using_columns].columns)}
        self.look_ahead_columns = {self.column_indices[col]: val for col, val in look_ahead_columns}

        # Work out the window parameters, if we have a list for input width, one width per each
        # feature than take the max and preserve information about slices widths for all the features
        if isinstance(input_width, list):
            self.feature_input_widths = input_width
            self.input_width = max(self.feature_input_widths)
        else:
            self.feature_input_widths = None
            self.input_width = input_width

        self.label_width = label_width
        self.shift = shift

        self.input_window = SubWindow(0, self.input_width, self.total_window_size)
        self.label_window = SubWindow(self.total_window_size - self.label_width, None, self.total_window_size)

    @property
    def total_window_size(self) -> int:
        return self.input_width + self.shift + self.label_width - 1

    def split_window_tf_variable(self, features):
        """
        Slices the window variably for each feature based on the individual feature widths
        :param features: Flattened array of feature inputs
        :return:
        """
        n_of_features = len(self.using_columns)
        for idx, feature_width in enumerate(self.feature_input_widths):
            feature_start = self.input_width * n_of_features - feature_width * n_of_features + idx
            feature_end = feature_start + feature_width * n_of_features
            if idx in self.look_ahead_columns:
                feature_end += self.look_ahead_columns[idx] * n_of_features

            feature_slice = features[:, slice(feature_start, feature_end, n_of_features)]
            if idx > 0:
                inputs = tf.concat([inputs, feature_slice], 1)
            else:
                inputs = feature_slice

        # !!! We assume only Glucose can be the label
        glucose_idx = self.column_indices["Glucose"]
        labels_start = self.total_window_size * n_of_features - self.label_width * n_of_features + glucose_idx
        labels_end = labels_start + self.label_width * n_of_features
        labels = features[:, slice(labels_start, labels_end, n_of_features)]

        inputs_lengths = sum(self.feature_input_widths)
        inputs_lengths += sum(look_ahead_len for look_ahead_len in self.look_ahead_columns.values())

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, inputs_lengths])
        labels.set_shape([None, self.label_width])
        return inputs, labels

    def split_window_tf(self, features):
        inputs = features[:, self.input_window.slice, :]
        labels = features[:, self.label_window.slice, :]

        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1
        )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def split_window_torch_univariate(self, features):
        inputs = features[self.input_window.slice]
        labels = features[self.label_window.slice]
        return inputs, labels

    def split_window_torch(self, features):
        """
        Expects unbatched data
        :param features:
        :return:
        """
        if len(self.using_columns) == 1:
            return self.split_window_torch_univariate(features)

        inputs = features[self.input_window.slice, :]
        labels = features[self.label_window.slice, :]

        label_indices = [self.column_indices[name] for name in self.label_columns]
        labels = torch.stack(
            [labels[:, index] for index in label_indices],
            dim=-1
        )
        return inputs, labels

    def tf_dataset_from_numpy_windows(self, np_windows: list, is_batched: bool):
        ds = tf.data.Dataset.from_tensor_slices(np_windows)
        ds = ds.batch(self.batch_size if is_batched else 1)
        if self.feature_input_widths:
            ds = ds.map(self.split_window_tf_variable)
        else:
            ds = ds.map(self.split_window_tf)
        return ds

    def torch_dataset_from_numpy_windows(self, np_windows: list, is_batched: bool):
        # Create the custom dataset
        dataset = CustomDataset(np_windows, self)
        data_loader = DataLoader(dataset, batch_size=self.batch_size if is_batched else 1, drop_last=True, shuffle=False)
        return data_loader

    def _get_rolling(self, data):
        data = data.copy()
        windows = []
        rolling_windows = data.rolling(window=self.total_window_size)
        for _w in rolling_windows:
            # Not a full window
            if len(_w) < self.total_window_size:
                continue
            windows.append(_w)
        return windows

    def make_dataset_w_pandas(self, data, drop_noisy, is_batched=True, raw_windows=False):
        data = data.copy()

        if "finger_stick" in self.using_columns:
            # Replace nan with 0
            data["finger_stick"].fillna(0, inplace=True)

        # Fill missing BG based on finger_stick if it exists and is not 0
        data['Glucose'] = data.apply(lambda x: x["Glucose"] if not np.isnan(x["Glucose"]) or not x["finger_stick"] else x["finger_stick"], axis=1)

        # Create windows (stride=1)
        rolling_windows = data.rolling(window=self.total_window_size)
        print(f"all rolling: {len([w for w in rolling_windows])}")

        window_count = 0
        too_many_nans_cnt = 0
        empty_label_cnt = 0
        noise_end = self.input_width + self.noisy_period if self.noisy_period else None

        # Filter out gaps and take full windows
        all_rolling = []
        for _w in rolling_windows:
            # Not a full window
            if len(_w) < self.total_window_size:
                continue

            # If the data to be predicted is none, skip it, otherwise interpolate
            if np.isnan(_w["Glucose"][self.label_window.slice]).any():
                empty_label_cnt += 1
                continue

            # No input data - can't do anything
            elif _w["Glucose"][self.input_window.slice].isna().all():
                too_many_nans_cnt += 1
                continue

            # Interpolate the few missing
            elif _w["Glucose"][self.input_window.slice].isna().any():
                if not self.interpolate:
                    too_many_nans_cnt += 1
                    continue
                _w = _w.reset_index()
                _w.loc[0: self.input_width - 1, "Glucose"] = _w["Glucose"][self.input_window.slice].interpolate().bfill()
                window_count += 1
                all_rolling.append(_w)

            # No need to interpolate
            else:
                all_rolling.append(_w)
                window_count += 1

        np_windows = []
        for _w in all_rolling:
            # Drop windows which contain any noise (non-glucose features) in the time now and time of prediction
            # as this can skew the input-output mapping
            if drop_noisy and not (_w[self.NOISY_COLUMNS][slice(self.input_width, noise_end)] == 0).all().all():
                continue
            # Also drop any windows with considerable physical activity
            if (
                 drop_noisy and "calories" in self.train_df.columns and
                (_w["calories"][slice(self.input_width, noise_end)] > 0.6).any().any()
            ):
                continue

            # Apply gaussian 1D filter on the input
            if self.gaussian_sigma is not None:
                _w.loc[0: self.input_width - 1, "Glucose"] = gaussian_filter1d(_w.loc[0: self.input_width - 1, "Glucose"], self.gaussian_sigma)

            # Flatten if we have differently sized windows for individual input features, or in case of torch univariate
            if self.feature_input_widths or (len(self.using_columns) == 1 and self.provider == "torch"):
                np_windows.append(np.array(_w[self.using_columns], dtype=np.float32).flatten())
            else:
                np_windows.append(np.array(_w[self.using_columns], dtype=np.float32))

        print(window_count, "size before reduction")
        print(len(np_windows), "size after reduction")
        print(too_many_nans_cnt, "windows with too many nans")
        print(empty_label_cnt, "windows with empty labels")
        if raw_windows:
            return np_windows
        if self.provider == "tf":
            return self.tf_dataset_from_numpy_windows(np_windows, is_batched)
        elif self.provider == "torch":
            return self.torch_dataset_from_numpy_windows(np_windows, is_batched)

    @staticmethod
    def noisify(feature: pd.Series, rate: float, magnitude: float) -> pd.Series:
        # Randomly choose which elements of the feature array will get noise assigned with the given rate
        if rate == 1:
            mask = np.ones(len(feature))
        else:
            mask = np.random.choice([0, 1], size=len(feature), p=(1 - rate, rate))
        # Generate random noise by which the randomly picked elements will get multiplied
        noise = np.random.uniform(-magnitude, magnitude, len(feature)) + 1
        mask = np.multiply(mask, noise)
        # Fill zeroes with 1s, so that the elements that weren't picked will remain the same
        mask[mask == 0] = 1
        return np.multiply(feature, mask)

    def create_noisy_windows(self):
        wins = copy.deepcopy(self.cached_train_windows)
        # Add noise per feature
        for conf in self.noisify_configs.noise:
            for win in wins:
                win[:, self.column_indices[conf.feature_name]] = self.noisify(win[:, self.column_indices[conf.feature_name]], conf.rate, conf.magnitude)
        return wins

    @cached_property
    def cached_train_windows(self) -> list:
        return self.make_dataset_w_pandas(self.train_df, self.train_drop_noisy, raw_windows=True)

    @cached_property
    def cached_val_windows(self) -> list:
        return self.make_dataset_w_pandas(self.val_df, self.val_drop_noisy, raw_windows=True)

    @cached_property
    def cached_test_windows(self) -> list:
        if len(self.test_df) == 0:
            return []
        return self.make_dataset_w_pandas(self.test_df, self.test_drop_noisy, raw_windows=True)

    @cached_property
    def cached_train(self):
        return self.make_dataset_w_pandas(self.train_df, self.train_drop_noisy)

    @property
    def train(self):
        if self.shuffle:
            windows = self.cached_train_windows
            random.shuffle(windows)
            if self.provider == "tf":
                return self.tf_dataset_from_numpy_windows(windows, True)
            elif self.provider == "torch":
                print("return torch")
                return self.torch_dataset_from_numpy_windows(windows, True)
        if self.noisify_configs:
            wins = self.create_noisy_windows()
            if self.provider == "tf":
                return self.tf_dataset_from_numpy_windows(wins, True)
            elif self.provider == "torch":
                return self.torch_dataset_from_numpy_windows(wins, True)

        return self.cached_train

    @cached_property
    def val(self):
        return self.make_dataset_w_pandas(self.val_df, self.val_drop_noisy)

    @cached_property
    def test(self):
        return self.make_dataset_w_pandas(self.test_df, self.test_drop_noisy, False)

    @cached_property
    def train_and_val(self):
        if self.noisify_configs:
            return self.create_noisy_ds(pd.concat([self.train_df, self.val_df]), self.train_drop_noisy)
        return self.make_dataset_w_pandas(pd.concat([self.train_df, self.val_df]), self.train_drop_noisy)

    @cached_property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        return next(iter(self.train))

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input width: {self.input_width}',
            f'Shift: {self.shift}',
            f'Input indices: {self.input_window.indices}',
            f'Input column name(s): {self.using_columns}',
            f'Label indices: {self.label_window.indices}',
            f'Label column name(s): {self.label_columns}'])


class MultiWindowGenerator:
    def __init__(self, window_generators: list[WindowGenerator] = None, shuffle=True, provider="tf"):
        if window_generators is None:
            window_generators = []
        assert len(set([repr(wg) for wg in window_generators])) <= 1, "All window generators must have the same params"
        self.window_generators = window_generators
        self.shuffle = shuffle
        self.provider = provider

    def add_window(self, window_generator: WindowGenerator):
        assert all(repr(window_generator) == repr(wg) for wg in self.window_generators)
        self.window_generators.append(window_generator)

    def _build_windows(self, df_type: str):
        all_windows = []
        if df_type == "train":
            for window_gen in self.window_generators:
                if window_gen.noisify_configs:
                    all_windows.extend(window_gen.create_noisy_windows())
                else:
                    all_windows.extend(window_gen.cached_train_windows)
        elif df_type == "val":
            for window_gen in self.window_generators:
                all_windows.extend(window_gen.cached_val_windows)
        elif df_type == "test":
            for window_gen in self.window_generators:
                all_windows.extend(window_gen.cached_test_windows)
        else:
            raise ValueError("Invalid df_type, choose train, val or test")

        if self.shuffle:
            random.shuffle(all_windows)
        # Can choose any window to build them up if they have the same parameters
        if self.provider == "tf":
            return self.window_generators[0].tf_dataset_from_numpy_windows(all_windows, False if df_type == "test" else True)
        elif self.provider == "torch":
            print("return torch")
            return self.window_generators[0].torch_dataset_from_numpy_windows(all_windows, True)
        else:
            raise ValueError("Invalid provider, choose tf or torch")

    @cached_property
    def train_cached(self):
        return self._build_windows("train")

    @property
    def train(self, cached=True):
        if any(window_gen.noisify_configs is not None for window_gen in self.window_generators):
            cached = False
        if cached:
            return self.train_cached
        return self._build_windows("train")

    @cached_property
    def val(self):
        return self._build_windows("val")

    @cached_property
    def test(self):
        return self._build_windows("test")

    @cached_property
    def train_and_val(self):
        return self._build_windows("train").concatenate(self._build_windows("val"))
