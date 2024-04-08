import pandas as pd
import numpy as np
from core.window_generator import WindowGenerator
import tensorflow as tf
import pytest


def get_window_mocked(columns_to_use, input_width, label_width, shift, batch_size=32, drop_noisy=False, noisy_period=None, look_ahead_columns=None):
    return WindowGenerator(
        input_width=input_width,
        label_width=label_width if label_width == 1 else label_width,
        shift=shift,
        train_df_in=pd.DataFrame(columns=columns_to_use),
        val_df_in=pd.DataFrame(columns=columns_to_use),
        test_df_in=pd.DataFrame(columns=columns_to_use),
        batch_size=batch_size,
        use_columns=columns_to_use,
        drop_noisy=drop_noisy,
        noisy_period=noisy_period,
        look_ahead_columns=look_ahead_columns or []
    )


@pytest.mark.parametrize(
    "columns, input_width, label_width, shift, input_series, expected",
    [
        (
            ["Glucose", "Rapid Insulin"],
            [2, 3],
            1,
            2,
            np.array([[10.3, 2, 9, 0, 8.8, 0, 8.2, 0, 7.9, 1]]),
            (np.array([9, 8.8, 2, 0, 0]), np.array([7.9]))
        ),
        (
            ["Glucose", "Rapid Insulin"],
            [2, 2],
            1,
            3,
            np.array([[10.3, 2, 9, 0, 8.8, 0, 8.2, 0, 7.9, 1]]),
            (np.array([10.3, 9, 2, 0]), np.array([7.9]))
        ),
        (
            ["Glucose", "Rapid Insulin"],
            [2, 2],
            1,
            2,
            np.array([[10.3, 2, 9, 0, 8.8, 0, 8.2, 0]]),
            (np.array([10.3, 9, 2, 0]), np.array([8.2]))
        ),
        (
            ["Glucose", "Rapid Insulin"],
            [2, 1],
            2,
            1,
            np.array([[10.3, 2, 9, 0, 8.8, 0, 8.2, 0]]),
            (np.array([10.3, 9, 0]), np.array([8.8, 8.2]))
        ),
    ]
)
def test_split_window_variable(columns, input_width, label_width, shift, input_series, expected):
    expected_inputs, expected_labels = (tf.convert_to_tensor(x) for x in expected)
    input_series = tf.convert_to_tensor(input_series)

    win = get_window_mocked(columns, input_width, label_width, shift)
    inputs, labels = win.split_window_tf_variable(input_series)
    assert tf.math.reduce_all(tf.math.equal(inputs, expected_inputs))
    assert tf.math.reduce_all(tf.math.equal(labels, expected_labels))


@pytest.mark.parametrize(
    "columns, lookahead_cols, input_width, label_width, shift, input_series, expected",
    [
        (
            ["Glucose", "Rapid Insulin IOB", "Carbohydrates"],
            [("Rapid Insulin IOB", 2)],
            [2, 0, 1],
            1,
            3,
            np.array([[10.3, 2, 25, 9, 0, 30, 8.8, 0, 0, 8.2, 0, 0, 7.9, 1, 0]]),
            (np.array([10.3, 9, 0, 0, 30]), np.array([7.9]))
        ),
        (
            ["Glucose", "Rapid Insulin IOB", "Carbohydrates"],
            [("Rapid Insulin IOB", 2)],
            [2, 1, 1],
            1,
            2,
            np.array([[10.3, 2, 25, 9, 0, 30, 8.8, 0, 0, 8.2, 1, 0]]),
            (np.array([10.3, 9, 0, 0, 1, 30]), np.array([8.2]))
        ),
    ]
)
def test_split_window_variable_lookahead(columns, lookahead_cols, input_width, label_width, shift, input_series, expected):
    expected_inputs, expected_labels = (tf.convert_to_tensor(x) for x in expected)
    input_series = tf.convert_to_tensor(input_series)

    win = get_window_mocked(columns, input_width, label_width, shift, look_ahead_columns=lookahead_cols)
    inputs, labels = win.split_window_tf_variable(input_series)
    assert tf.math.reduce_all(tf.math.equal(inputs, expected_inputs))
    assert tf.math.reduce_all(tf.math.equal(labels, expected_labels))
