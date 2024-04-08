import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import keras_lmu
import keras_tuner as kt
from keras_tuner import HyperModel
from core.pre_processing import build_ohio_dataset, build_windows
from core.window_generator import WindowGenerator


SAMPL_FREQ = 5
INPUT_LENGTH = 30
INPUT_DIMENSION = 3


class LMUHyperModel(HyperModel):
    def __init__(self, min_order: int, max_order: int, min_units: int, max_units: int,):
        self.min_order = min_order
        self.max_order = max_order
        self.min_units = min_units
        self.max_units = max_units
        super(LMUHyperModel, self).__init__()

    def build(self, hp):
        lmu_layer = keras_lmu.LMU(
            memory_d=INPUT_DIMENSION,
            order=hp.Int("order", min_value=self.min_order, max_value=self.max_order, sampling="log"),
            theta=INPUT_LENGTH // SAMPL_FREQ,
            hidden_to_memory=hp.Choice("hidden_to_memory", values=[True, False]),
            memory_to_memory=hp.Choice("memory_to_memory", values=[True, False]),
            input_to_hidden=hp.Choice("input_to_hidden", values=[True, False]),
            hidden_cell=tf.keras.layers.LSTMCell(units=hp.Int("units", min_value=self.min_units, max_value=self.max_units, sampling="log")),
            return_sequences=False,
            dropout=hp.Choice("dropout", values=[0.0, 0.1, 0.2, 0.5]),
            recurrent_dropout=hp.Choice("recurrent_dropout", values=[0.0, 0.1, 0.2, 0.5]),
        )
        inputs = tf.keras.Input((INPUT_LENGTH // SAMPL_FREQ, INPUT_DIMENSION))
        lmus = lmu_layer(inputs)

        outputs = tf.keras.layers.Dense(1)(lmus)
        output_plus_input = tf.keras.layers.Add()([outputs, inputs[:, -1, 0]])
        model = tf.keras.Model(inputs=inputs, outputs=output_plus_input)

        hp_learning_rate = 1e-3

        model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate),
            metrics=[tf.metrics.RootMeanSquaredError()]
        )
        return model


def run_tuner(dataset: str, project_name: str, w: WindowGenerator, min_order: int, max_order: int, min_units: int, max_units: int):
    tuner = kt.Hyperband(
        LMUHyperModel(min_order, max_order, min_units, max_units),
        objective=kt.Objective("val_root_mean_squared_error", direction="min"),
        max_epochs=250,
        factor=3,
        hyperband_iterations=1,
        project_name=project_name,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        overwrite=False
    )
    print(tuner.search_space_summary())

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=35,
        mode='min',
        restore_best_weights=True
    )
    b_lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9)
    checkpoint_filepath = f'/tmp/{project_name}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    callbacks = [early_stopping, b_lr_reducer, model_checkpoint_callback]

    tuner.search(w.train, epochs=250, validation_data=w.val, callbacks=callbacks, verbose=2)
    print("COMPLETED")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="LMU hyperparameter tuner")
    parser.add_argument("--dataset", type=str, default="2018", choices=["2018", "2020", "all"], help="Dataset to use")
    parser.add_argument("--ph", type=int, default=6, help="Prediction horizon in number of steps")
    parser.add_argument("--input_length", type=int, default=30, help="Input length in minutes")
    parser.add_argument("--project_name", type=str, help="Project name for the tuner")
    parser.add_argument("--min_order", type=int, default=4, help="Minimum LMU order")
    parser.add_argument("--max_order", type=int, default=64, help="Maximum LMU order")
    parser.add_argument("--min_units", type=int, default=32, help="Minimum LMU units")
    parser.add_argument("--max_units", type=int, default=512, help="Maximum LMU units")
    parser.add_argument("--pretrain_on_2018_train_only", action="store_true", help="Pretrain only on train portion of 2018 dataset")
    parser.add_argument("--features", nargs="+", default=["Time", "Glucose", "Rapid Insulin sub_tissue", "Carbohydrates gut"], help="Features to use")
    return parser.parse_args()


def main(args):
    global INPUT_LENGTH, INPUT_DIMENSION

    dataset = build_ohio_dataset(args.dataset, default_time_to_peak=100, pretrain_on_2018_train_only=args.pretrain_on_2018_train_only)
    w, _ = build_windows(
        dataset,
        args.features,
        args.input_length // SAMPL_FREQ,
        1,
        args.ph,
        drop_noisy=False,
        batch_size=256,
        min_max_scale=False
    )
    INPUT_LENGTH = args.input_length
    INPUT_DIMENSION = len(args.features) - 1
    run_tuner(args.dataset, args.project_name, w, args.min_order, args.max_order, args.min_units, args.max_units)


if __name__ == "__main__":
    main(parse_args())
