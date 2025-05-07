import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cmd = [
    "python", "-u", "./Time-Series-Library/run.py",
    "--task_name", "anomaly_detection",
    "--is_training", "1",
    "--root_path", "./data",
    "--model_id", "MSL",
    "--model", "Crossformer",
    "--data", "MSL",
    "--features", "M",
    "--seq_len", "100",
    "--pred_len", "0",
    "--d_model", "128",
    "--d_ff", "128",
    "--e_layers", "3",
    "--enc_in", "55",
    "--c_out", "55",
    "--anomaly_ratio", "1",
    "--batch_size", "128",
    "--train_epochs", "10"
]

subprocess.run(cmd)



# import sys, os
# import numpy as np
# import pandas as pd
# import torch
# from darts import TimeSeries
# from darts.models import RNNModel
# from darts.ad.anomaly_model import ForecastingAnomalyModel
# from darts.ad.scorers import KMeansScorer
# from darts.ad.detectors import QuantileDetector
# from sklearn.metrics import roc_auc_score, average_precision_score

# def load_series(path: str) -> tuple[TimeSeries, np.ndarray]:
#     df = pd.read_csv(path)
#     series = TimeSeries.from_dataframe(df, time_col='timestamp', value_cols=[col for col in df.columns if col.startswith('value_')])
#     anomaly = df['anomaly'].to_numpy(dtype=int)
#     return series, anomaly

# series_train, y_train = load_series('./data/yahoo_train.csv')
# series_test, y_test = load_series('./data/yahoo_test.csv')

# series_train = series_train.astype(np.float32)
# series_test = series_test.astype(np.float32)
# torch.set_default_dtype(torch.float32)

# model = RNNModel(lags=1)
# model.fit(series_train)

# fa_model = ForecastingAnomalyModel(model=model, scorer=KMeansScorer())
# fa_model.fit(series_train, allow_model_training=False)
# scores = fa_model.score(series_test)

# detector = QuantileDetector(high_quantile=0.995)
# detector.fit(scores)
# y_pred = (detector.detect(scores).values() > 0).any(axis=1).astype(int)

# offset = len(y_test) - len(y_pred)
# y_test_aligned = y_test[offset:]

# auroc = roc_auc_score(y_test_aligned, y_pred)
# auprc = average_precision_score(y_test_aligned, y_pred)
# print(f"AUROC: {auroc:.4f}")
# print(f"AUPRC: {auprc:.4f}")

# for i, (true, pred) in enumerate(zip(y_test_aligned, y_pred)):
#     if true != pred:
#         print(f"Failed prediction at point {series_test.time_index[offset + i]} with true label {true}")