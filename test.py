import os
import subprocess

cmd = [
    "python", "-u", "./Time-Series-Library/run.py",
    "--task_name", "anomaly_detection",
    "--is_training", "1",
    "--root_path", "./data/",
    # "--data_path", "MSL.csv",
    "--model_id", "MSL",
    "--model", "Autoformer",
    "--data", "MSL",
    "--features", "M",
    "--seq_len", "100",
    "--label_len", "50",
    "--pred_len", "100",
    "--e_layers", "2",
    "--d_layers", "1",
    "--factor", "3",
    "--enc_in", "55",
    "--dec_in", "55",
    "--c_out", "55",
    "--gpu", "0",
    "--d_model", "128",
    "--d_ff", "128",
    "--anomaly_ratio", "1",
    "--batch_size", "128",
    "--train_epochs", "10",
]

subprocess.run(cmd)