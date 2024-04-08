from core.data_utils import InsulinExpDecayFunction
from core.HvorkaInsulinModel import InsulinModel
from core.data_utils import train_val_split_simple
from core.window_generator import MultiWindowGenerator, WindowGenerator
from core.HvorkaInsulinModel import HovorkaGlucoseModel
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from core.data_utils import resample_data, interpolate_gaps
from sklearn import preprocessing
from core.ohio_dataset import load_ohio
import numpy as np


ORIG_FREQ = 5
SAMPL_FREQ = 5
FREQ_CORRECTION = ORIG_FREQ // SAMPL_FREQ

PATIENTS_2018 = ["559", "563", "570", "575", "588", "591"]
PATIENTS_2020 = ["540", "544", "552", "567", "584", "596"]


def get_window(tr_df, v_df, tst_df, columns_to_use, input_width, label_width, shift, batch_size=32, drop_noisy=False,
               noisy_period=None, look_ahead_columns=[], noise_configs=None, gaussian_sigma=None, interpolate=True):
    return WindowGenerator(
        input_width=input_width * FREQ_CORRECTION,
        label_width=label_width if label_width == 1 else label_width * FREQ_CORRECTION,
        shift=shift * FREQ_CORRECTION,
        train_df_in=tr_df,
        val_df_in=v_df,
        test_df_in=tst_df,
        batch_size=batch_size,
        use_columns=columns_to_use,
        drop_noisy=drop_noisy,
        noisy_period=noisy_period,
        look_ahead_columns=look_ahead_columns,
        noisify_configs=noise_configs,
        gaussian_sigma=gaussian_sigma,
        interpolate=interpolate,
    )


def run_pre_processing(train_df_in, test_df_in, time_to_peak=60, gaussian_sigma=None):
    for_resampling = ['Glucose', 'Rapid Insulin', 'Carbohydrates', "finger_stick", "Rapid Insulin basal"]
    train_df_in = resample_data(train_df_in, for_resampling, SAMPL_FREQ)
    test_df_in = resample_data(test_df_in, for_resampling, SAMPL_FREQ)
    using_features = for_resampling

    if gaussian_sigma is not None:
        train_df_in["Glucose"] = gaussian_filter1d(train_df_in["Glucose"], gaussian_sigma)
        test_df_in["Glucose"] = gaussian_filter1d(test_df_in["Glucose"], gaussian_sigma)

    for _df in [train_df_in, test_df_in]:
        # Basal rate is reported as number of units release per hour, we need to convert it to the sampling rate we are using
        _df["Rapid Insulin basal"] = _df["Rapid Insulin basal"] / 60 * SAMPL_FREQ

        exp_decay_func = InsulinExpDecayFunction(SAMPL_FREQ, hours_duration=5, peak_activity_min=time_to_peak)
        _df["Rapid Insulin IOB"] = 0
        max_n = len(_df)
        duration_samples = exp_decay_func.td // SAMPL_FREQ
        for idx, data in _df.iterrows():
            # Insulin injected
            if data["Rapid Insulin"] != 0:
                idx = int(idx)
                dur_end = int(min(idx + duration_samples, max_n))
                ins_dose = data["Rapid Insulin"]
                _df.loc[slice(int(idx), dur_end), "Rapid Insulin IOB"] = \
                    exp_decay_func.insulin_on_board_vector(_df.loc[slice(int(idx), dur_end), "Rapid Insulin IOB"], ins_dose)

        insulin_system = InsulinModel(time_to_peak // SAMPL_FREQ)
        _df["Rapid Insulin sub_tissue"] = 0
        _df["Rapid Insulin blood_plasma"] = 0
        for idx, data in _df.iterrows():
            insulin_system.update_compartments(data["Rapid Insulin"])
            sub_tissue_ins, blood_plasma_ins = insulin_system.get_variables()
            _df.loc[int(idx), "Rapid Insulin sub_tissue"] = sub_tissue_ins
            _df.loc[int(idx), "Rapid Insulin blood_plasma"] = blood_plasma_ins

        gluc_system = HovorkaGlucoseModel(60 // SAMPL_FREQ)
        _df["Carbohydrates gut"] = 0
        _df["Carbohydrates plasma"] = 0
        for idx, data in _df.iterrows():
            gluc_system.update_compartments(data["Carbohydrates"])
            gut_glucose, plasma_glucose = gluc_system.get_variables()
            _df.loc[int(idx), "Carbohydrates gut"] = gut_glucose
            _df.loc[int(idx), "Carbohydrates plasma"] = plasma_glucose

        # Add missingness indicator for glucose
        _df["Glucose missing"] = _df["Glucose"].isna().astype(int)

        day = 24*60*60
        timestamp_s = _df["Time"].map(pd.Timestamp.timestamp)
        _df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        _df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    using_features.append("Rapid Insulin IOB")
    using_features.append("Rapid Insulin sub_tissue")
    using_features.append("Rapid Insulin blood_plasma")
    return train_df_in, test_df_in, using_features


def identify_gaps(df_in: pd.DataFrame) -> int:
    gaps = df_in.dropna(subset=["Glucose"])["Time"].diff() > pd.Timedelta(minutes=SAMPL_FREQ)
    return len(gaps[gaps == True])


def build_ohio_dataset(dataset: str, time_to_peak: dict = None, gaussian_sigma=None, default_time_to_peak=100, pretrain_on_2018_train_only=False):
    patient_dfs = []
    datasets = []
    if dataset == "2018":
        datasets = ["2018"]
    elif dataset == "2020":
        datasets = ["2020"]
    elif dataset == "all":
        datasets = ["2018", "2020"]

    for dataset in datasets:
        for patient in PATIENTS_2018 if dataset == "2018" else PATIENTS_2020:
            ohio_train_df, ohio_test_df = load_ohio(dataset, patient)
            ttp = time_to_peak[patient] if time_to_peak else default_time_to_peak
            print(f"Using time to peak: {ttp} for patient {patient}")
            train_df, test_df, using_features = run_pre_processing(ohio_train_df, ohio_test_df, ttp, gaussian_sigma)

            # If we are using all data we want to pre-train on whole 2018 and eval on 2020 test
            if len(datasets) > 1 and dataset == "2018" and not pretrain_on_2018_train_only:
               train_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
               test_df = pd.DataFrame(columns=train_df.columns)

            train_df, val_df, n = train_val_split_simple(train_df, ratio=(0.8, 0.2))

            for df_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
                print(f"Identified {identify_gaps(df)} gaps for {df_name} set of {patient}")

            # Fill in missing BG based on finger_stick first, then interpolate
            for df in [train_df, val_df]:
                # Fill missing Glucose if we have finger_stick values
                df["Glucose"] = df["Glucose"].fillna(df["finger_stick"])
                interpolate_gaps(df, limit=30 // SAMPL_FREQ)

            if "finger_stick" in using_features:
                # Replace NaN with 0
                train_df["finger_stick"].fillna(0, inplace=True)
                val_df["finger_stick"].fillna(0, inplace=True)
                test_df["finger_stick"].fillna(0, inplace=True)
            patient_dfs.append((train_df, val_df, test_df, patient))
    return patient_dfs


def build_windows(patient_dfs, features_to_use, input_width, label_width, shift, batch_size=32, drop_noisy=False,
                  noisy_period=None, look_ahead_columns=[], noise_configs=None, min_max_scale=True,
                  gaussian_sigma=None, pre_train=False, interpolate=True):
    # Min-max normalize all together
    min_max_scaler = None
    if min_max_scale:
        all_tr = pd.concat([tr for tr, _, _, _ in patient_dfs])
        scale_features = [col for col in features_to_use if col != "Time"]
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(all_tr[scale_features])

    windows = {}
    for p_tr_df, p_v_df, p_tst_df, patient_id in patient_dfs:
        p_tr_df = p_tr_df.copy()
        p_v_df = p_v_df.copy()
        p_tst_df = p_tst_df.copy()
        if min_max_scale:
            p_tr_df[scale_features] = min_max_scaler.transform(p_tr_df[scale_features])
            p_v_df[scale_features] = min_max_scaler.transform(p_v_df[scale_features])
            if len(p_tst_df):
                p_tst_df[scale_features] = min_max_scaler.transform(p_tst_df[scale_features])
        w = get_window(
            p_tr_df,
            p_v_df,
            p_tst_df,
            features_to_use,
            input_width,
            label_width,
            shift,
            batch_size=batch_size,
            drop_noisy=drop_noisy,
            noisy_period=noisy_period,
            look_ahead_columns=look_ahead_columns,
            noise_configs=noise_configs,
            gaussian_sigma=gaussian_sigma,
            interpolate=interpolate,
        )
        windows[patient_id] = w
    if pre_train:
        # In pre-training, return 2 multi-windows
        multi_window_pre_train = MultiWindowGenerator([windows[p_id] for p_id in PATIENTS_2018])
        multi_window_train = MultiWindowGenerator([windows[p_id] for p_id in PATIENTS_2020])
        return multi_window_pre_train, multi_window_train, min_max_scaler
    return MultiWindowGenerator([windows[p_id] for p_id in windows]), min_max_scaler


def get_patient_window(patient_df, features_to_use, input_width, label_width, shift, batch_size=32, drop_noisy=False,
                       noisy_period=None, look_ahead_columns=[], noise_configs=None, min_max_scale=True,
                       gaussian_sigma=None, interpolate=True):
    # Min-max normalize all together
    min_max_scaler = None
    if min_max_scale:
        scale_features = [col for col in features_to_use if col != "Time"]
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(patient_df[scale_features])

    p_tr_df, p_v_df, p_tst_df = patient_df
    p_tr_df = p_tr_df.copy()
    p_v_df = p_v_df.copy()
    p_tst_df = p_tst_df.copy()
    if min_max_scale:
        p_tr_df[scale_features] = min_max_scaler.transform(p_tr_df[scale_features])
        p_v_df[scale_features] = min_max_scaler.transform(p_v_df[scale_features])
        p_tst_df[scale_features] = min_max_scaler.transform(p_tst_df[scale_features])

    return get_window(
        p_tr_df,
        p_v_df,
        p_tst_df,
        features_to_use,
        input_width,
        label_width,
        shift,
        batch_size=batch_size,
        drop_noisy=drop_noisy,
        noisy_period=noisy_period,
        look_ahead_columns=look_ahead_columns,
        noise_configs=noise_configs,
        gaussian_sigma=gaussian_sigma,
        interpolate=interpolate,
    )


def evaluate_on_patients(in_model, model_name, dataset, patients, ph=6, input_len=30, features=None, gaussian_sigma=None, interpolate=True):
    if features is None:
        features = ["Time", "Glucose", "Rapid Insulin sub_tissue", "Carbohydrates gut"]

    patient_windows = {}
    for p_tr_df, p_v_df, p_tst_df, patient_id in dataset:
        if patient_id not in patients:
            continue
        patient_windows[patient_id] = get_patient_window(
            (p_tr_df, p_v_df, p_tst_df),
            features,
            input_len // SAMPL_FREQ,
            1,
            ph,
            drop_noisy=False,
            batch_size=256,
            min_max_scale=False,
            gaussian_sigma=gaussian_sigma,
            interpolate=interpolate,
        )

    results_val = []
    results_test = []

    for patient_id, patient_w in patient_windows.items():
        print("Evaluating on patient_id:", patient_id)
        val_p = in_model.evaluate(patient_w.val, verbose=0)[1]
        test_p = in_model.evaluate(patient_w.test, verbose=0)[1]
        results_val.append((model_name, patient_id, val_p))
        results_test.append((model_name, patient_id, test_p))
    return results_val, results_test
