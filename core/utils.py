from core.window_generator import WindowGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


MMOL_TO_MGDL = 18.016


def model_fit(model, window, patience=2, max_epochs=10, monitor='val_loss', verbose='auto', store_model="best_model"):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode='min',
        restore_best_weights=True
    )
    b_lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9)

    checkpoint_filepath = f'/tmp/checkpoint/{store_model}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    callbacks = [early_stopping, b_lr_reducer, model_checkpoint_callback]

    history = model.fit(window.train, epochs=max_epochs, validation_data=window.val, callbacks=callbacks, verbose=verbose)
    return history


def compile_and_fit(model, window, patience=2, max_epochs=10, monitor='val_loss', learning_rate=10**-2, verbose='auto', store_model="best_model"):
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.metrics.RootMeanSquaredError()]
    )
    return model_fit(model, window, patience, max_epochs, monitor, verbose, store_model)


def save_model(model, name):
    model.save(f'models/{name}')


def min_max_normalized_to_orig(min_max_scaler, norm_val):
    """
    Takes a RMSE error on min-max normalized values and converts it to the original scale
    :param min_max_scaler:
    :param norm_val:
    :return:
    """
    d_max, d_min = min_max_scaler.data_max_[0], min_max_scaler.data_min_[0]
    err_inverse = (d_max - d_min)
    return err_inverse * norm_val


def perf_min_max_normalized_to_orig(min_max_scaler, perf):
    return {k: [min_max_normalized_to_orig(min_max_scaler, x) for x in v] for k, v in perf.items()}


def perf_mmol_to_dgl(perf):
    return {k: [mmol_to_dgl(x) for x in v] for k, v in perf.items()}


def mmol_to_dgl(val):
    return MMOL_TO_MGDL * val


def bg_denormalize(min_max_scaler, norm_val, unit_to="mgdl"):
    """
    Converts BG values from notmalized to "unit_to"
    :param min_max_scaler:
    :param norm_val:
    :param unit_to:
    :return:
    """
    d_max, d_min = min_max_scaler.data_max_[0], min_max_scaler.data_min_[0]
    orig = norm_val * (d_max - d_min) + d_min
    if unit_to == "mgdl":
        orig *= MMOL_TO_MGDL
    return orig


def show_performance(min_max_scaler, perf, units="norm"):
    if units == "norm":
        pass
    elif units == "mmol":
        perf = perf_min_max_normalized_to_orig(min_max_scaler, perf)
    elif units == "mgdl":
        perf = perf_mmol_to_dgl(perf_min_max_normalized_to_orig(min_max_scaler, perf))
    for mod_perf, perf_data in perf.items():
        if isinstance(perf_data, list):
            print(mod_perf, perf_data[1])
        else:
            print(mod_perf, perf_data)


def get_flat_arrays(data_in):
    x_arr, y_arr = [], []
    for x, y in data_in:
        x_arr.append(x.numpy().flatten())
        y_arr.append(y.numpy().flatten())
    return x_arr, np.ravel(y_arr)


def plot_loss(history, label, log_scale=True):
    # Use a log scale on y-axis to show the wide range of values.
    if log_scale:
        plt.semilogy(history.epoch, history.history['loss'], label='Train ' + label)
        plt.semilogy(history.epoch, history.history['val_loss'], label='Val ' + label, linestyle="--")
    else:
        plt.plot(history.epoch, history.history['loss'], label='Train ' + label)
        plt.plot(history.epoch, history.history['val_loss'], label='Val ' + label, linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


def get_predictions(model: tf.keras.Model, window: WindowGenerator, min_max_scaler):
    all_predictions = []
    all_targets = []
    for batch in window.val:
        inputs, targets = batch
        targets = targets.numpy().flatten()
        predictions = model.predict(inputs).flatten()

        # De-normalize
        d_max, d_min = min_max_scaler.data_max_[0], min_max_scaler.data_min_[0]
        targets = targets * (d_max - d_min) + d_min
        predictions = predictions * (d_max - d_min) + d_min

        all_predictions.extend(predictions)
        all_targets.extend(targets)
    return all_predictions, all_targets
