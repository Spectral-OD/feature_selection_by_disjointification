from datetime import datetime
from scipy.stats import wilcoxon
from pathlib import Path
import pandas as pd
import csv

def get_dt_in_fmt(function=datetime.now, fmt="%m_%d_%Y__%H_%M_%S"):
    time = datetime.strftime(function(), fmt)
    return time


def wilcoxon_p_value(x, y):
    w_test = wilcoxon(x, y)
    return w_test.pvalue


def make_transposed_csv(input_csv_file, to_file=False, return_path=False):
    features_df_path = input_csv_file
    features_t_df_path = str(features_df_path).replace(features_df_path.stem, features_df_path.stem + "_t")
    col_names = []
    cols = []

    with open(features_df_path) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            col_name = row[0]
            col = row[1:]
            col_names.append(col_name)
            cols.append(col)
    col_names[0] = "samplename"
    series_list = []
    for col, col_name in zip(cols, col_names):
        seri = pd.Series(data=col, name=col_name)
        series_list.append(seri)

    df_transposed = pd.concat(series_list, axis=1)
    if to_file:
        df_transposed.to_csv(features_t_df_path)
        if return_path:
            return features_t_df_path
    return df_transposed


def get_int_or_fraction(select, out_of, interpret_one_as_all=True):
    selector = select > 1 if interpret_one_as_all else select >= 1
    out = int(select) if selector else int(select * out_of)
    return out