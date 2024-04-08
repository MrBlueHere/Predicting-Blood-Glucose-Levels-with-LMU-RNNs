import os
import xml.etree.ElementTree as ET
import pandas as pd


# Here add the path to the OhioT1DM dataset
DATA_DIR = os.path.join("..", "data", "OhioT1DM")
SAMPL_FREQ = 5


def load_ohio(dataset, subject):
    """
    Load OhioT1DM training and testing files into a dataframe
    :param dataset: name of dataset
    :param subject: name of subject
    :return: dataframe
    """
    train_path, test_path = _compute_file_names(dataset, subject)

    [train_xml, test_xml] = [ET.parse(set).getroot() for set in [train_path, test_path]]
    [train, test] = [_extract_data_from_xml(xml) for xml in [train_xml, test_xml]]
    return train, test


def _compute_file_names(dataset, subject):
    """
    Compute the name of the files, given dataset and subject
    :param dataset: name of dataset
    :param subject: name of subject
    :return: path to training file, path to testing file
    """
    train_dir = os.path.join(DATA_DIR, dataset, "train")
    train_path = os.path.join(train_dir, f"{subject}-ws-training.xml")
    test_dir = os.path.join(DATA_DIR, dataset, "test")
    test_path = os.path.join(test_dir, f"{subject}-ws-testing.xml")
    return train_path, test_path


def _extract_data_from_xml(xml) -> pd.DataFrame:
    """
    extract glucose, carbs, and insulin from xml and merge the data
    :param xml:
    :return: dataframe
    """
    glucose_df = _get_glucose_from_xml(xml)
    carbs_df = _get_carbs_from_xml(xml)
    insulin_df = _get_insulin_from_xml(xml)
    insulin_basal_df = _get_basal_insulin_from_xml(xml)
    fingerstick_df = _get_fingerstick_from_xml(xml)

    df = pd.merge(glucose_df, carbs_df, how="outer", on="Time")
    df = pd.merge(df, insulin_df, how="outer", on="Time")
    df = pd.merge(df, insulin_basal_df, how="outer", on="Time")
    df = pd.merge(df, fingerstick_df, how="outer", on="Time")
    #df = pd.merge(df, steps_df, how="outer", on="Time")
    df = df.sort_values("Time")

    return df


def _get_field_idx(etree, field_name: str) -> int:
    field_idx = -1
    for idx, field in enumerate(etree):
        if field.tag == field_name:
            field_idx = idx
            break
    return field_idx


def _get_field_labels(etree, field_name) -> list:
    field_idx = _get_field_idx(etree, field_name)
    return list(etree[field_idx][0].attrib.keys()) if len(etree[field_idx]) > 0 else []


def _iter_fields(etree, field_name):
    field_idx = _get_field_idx(etree, field_name)
    if len(etree[field_idx]) == 0:
        return None
    for event in etree[field_idx].iter("event"):
        yield list(event.attrib.values())


def _get_carbs_from_xml(xml) -> pd.DataFrame:
    labels = _get_field_labels(xml, field_name="meal") or []
    carbs = list(_iter_fields(xml, field_name="meal")) or []
    carbs_df = pd.DataFrame(data=carbs, columns=labels)
    if len(labels) == 0:
        print("Empty carbs")
        carbs_df["ts"] = []
        carbs_df["carbs"] = []
        carbs_df.rename(columns={'ts': 'Time', 'carbs': 'Carbohydrates'}, inplace=True)
        return carbs_df
    carbs_df.drop("type", axis=1, inplace=True)
    carbs_df["ts"] = pd.to_datetime(carbs_df["ts"], format="%d-%m-%Y %H:%M:%S")
    carbs_df["carbs"] = carbs_df["carbs"].astype("float")
    carbs_df.rename(columns={'ts': 'Time', 'carbs': 'Carbohydrates'}, inplace=True)
    return carbs_df


def _get_basal_insulin_from_xml(xml) -> pd.DataFrame:
    basal_df = pd.DataFrame(data=list(_iter_fields(xml, field_name="basal")), columns=_get_field_labels(xml, field_name="basal"))
    basal_df["ts"] = pd.to_datetime(basal_df["ts"], format="%d-%m-%Y %H:%M:%S")
    basal_df["value"] = basal_df["value"].astype("float")
    basal_df = basal_df.set_index("ts").resample(f"{SAMPL_FREQ}T").ffill().reset_index()

    # Override with temp_basal
    temp_basal_df = pd.DataFrame(data=list(_iter_fields(xml, field_name="temp_basal")), columns=_get_field_labels(xml, field_name="temp_basal"))
    if len(temp_basal_df) != 0:
        temp_basal_df["ts_begin"] = pd.to_datetime(temp_basal_df["ts_begin"], format="%d-%m-%Y %H:%M:%S")
        temp_basal_df["ts_end"] = pd.to_datetime(temp_basal_df["ts_end"], format="%d-%m-%Y %H:%M:%S")
        temp_basal_df["value"] = temp_basal_df["value"].astype("float")
        # For all values in temp_basal_df, filter out rows from basal_df based on ts_begin and ts_end and override the value
        for index, row in temp_basal_df.iterrows():
            mask = (basal_df['ts'] >= row['ts_begin']) & (basal_df['ts'] <= row['ts_end'])
            basal_df.loc[mask, 'value'] = row['value']

    basal_df.rename(columns={'ts': 'Time', 'value': 'Rapid Insulin basal'}, inplace=True)
    return basal_df


def _get_insulin_from_xml(xml) -> pd.DataFrame:
    labels = _get_field_labels(xml, field_name="bolus")
    insulin = list(_iter_fields(xml, field_name="bolus"))
    insulin_df = pd.DataFrame(data=insulin, columns=labels)
    for col in ["ts_end", "type", "bwz_carb_input"]:
        insulin_df.drop(col, axis=1, inplace=True, errors="ignore")
    insulin_df["ts_begin"] = pd.to_datetime(insulin_df["ts_begin"], format="%d-%m-%Y %H:%M:%S")
    insulin_df["dose"] = insulin_df["dose"].astype("float")
    insulin_df.rename(columns={'ts_begin': 'Time', 'dose': 'Rapid Insulin'}, inplace=True)
    return insulin_df


def _get_insulin_from_xml_with_squared(xml) -> pd.DataFrame:
    labels = _get_field_labels(xml, field_name="bolus")
    insulin = list(_iter_fields(xml, field_name="bolus"))
    insulin_df = pd.DataFrame(data=insulin, columns=labels)
    for col in ["type", "bwz_carb_input"]:
        insulin_df.drop(col, axis=1, inplace=True, errors="ignore")

    insulin_df["ts_begin"] = pd.to_datetime(insulin_df["ts_begin"], format="%d-%m-%Y %H:%M:%S")
    insulin_df["ts_end"] = pd.to_datetime(insulin_df["ts_end"], format="%d-%m-%Y %H:%M:%S")
    insulin_df["dose"] = insulin_df["dose"].astype("float")

    # Fill the values between ts_begin and ts_end with does / timedelta
    samples_df = pd.DataFrame(columns=['ts', 'dose'])
    for index, row in insulin_df.iterrows():
        if row['ts_begin'] == row['ts_end']:
            samples_df = samples_df.append({'ts': row['ts_begin'], 'dose': row['dose']}, ignore_index=True)
        else:
            time_range = pd.date_range(start=row['ts_begin'], end=row['ts_end'], freq=f'{SAMPL_FREQ}T')
            time_delta_minutes = (row['ts_end'] - row['ts_begin']).total_seconds() / 60
            dose = row['dose'] / time_delta_minutes * SAMPL_FREQ
            for time in time_range:
                samples_df = samples_df.append({'ts': time, 'dose': dose}, ignore_index=True)

    samples_df.rename(columns={'ts': 'Time', 'dose': 'Rapid Insulin'}, inplace=True)
    return samples_df


def _get_glucose_from_xml(xml) -> pd.DataFrame:
    labels = _get_field_labels(xml, field_name="glucose_level")
    glucose = list(_iter_fields(xml, field_name="glucose_level"))
    glucose_df = pd.DataFrame(data=glucose, columns=labels)
    glucose_df["ts"] = pd.to_datetime(glucose_df["ts"], format="%d-%m-%Y %H:%M:%S")
    glucose_df["value"] = glucose_df["value"].astype("float")
    glucose_df.rename(columns={'ts': 'Time', 'value': 'Glucose'}, inplace=True)
    return glucose_df


def _get_fingerstick_from_xml(xml) -> pd.DataFrame:
    labels = _get_field_labels(xml, field_name="finger_stick")
    finger_stick = list(_iter_fields(xml, field_name="finger_stick"))
    finger_stick_df = pd.DataFrame(data=finger_stick, columns=labels)
    finger_stick_df["ts"] = pd.to_datetime(finger_stick_df["ts"], format="%d-%m-%Y %H:%M:%S")
    finger_stick_df["value"] = finger_stick_df["value"].astype("float")
    finger_stick_df.rename(columns={'ts': 'Time', 'value': 'finger_stick'}, inplace=True)
    return finger_stick_df
