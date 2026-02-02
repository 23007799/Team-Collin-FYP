# data_prep.py
import pandas as pd
import numpy as np
import re


def load_data(path):
    data = pd.read_excel(path)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
    elif "date" in data.columns:
        data["Date"] = pd.to_datetime(data["date"])
    else:
        raise ValueError("No Date column found")

    return data.sort_values("Date").reset_index(drop=True)


def parse_human_numbers(df, columns):
    def _parse(x):
        if pd.isna(x) or isinstance(x, (int, float)):
            return x
        s = str(x).replace(",", "").replace("$", "")
        if s.endswith("%"):
            return float(s[:-1]) / 100
        m = re.match(r"([0-9.]+)([KMBT]?)", s)
        if m:
            mult = {"":1, "K":1e3, "M":1e6, "B":1e9, "T":1e12}
            return float(m.group(1)) * mult[m.group(2)]
        return pd.to_numeric(s, errors="coerce")

    for c in columns:
        if c in df.columns:
            df[c] = df[c].apply(_parse)

    return df


def split_by_year(data, target_year):
    dates = np.sort(data["Date"].unique())
    years = data["Date"].dt.year.unique()

    if target_year in years:
        test = data[data["Date"].dt.year == target_year]
        valid = data[data["Date"].dt.year == target_year - 1]
        train = data[~data.index.isin(test.index | valid.index)]
    else:
        n = len(dates)
        train = data[data["Date"] <= dates[int(0.7*n)]]
        valid = data[(data["Date"] > dates[int(0.7*n)]) &
                     (data["Date"] <= dates[int(0.85*n)])]
        test  = data[data["Date"] > dates[int(0.85*n)]]

    return train, valid, test
