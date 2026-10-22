#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import logging
import argparse
import numpy as np

import torch

from tqdm import tqdm

from neural_utils import run_neural_model

from preprocess import read_single_dataset

from models import arima
import json


# --------------------------------------------------------------------------- #
#  Logging setup
# --------------------------------------------------------------------------- #
class TqdmLoggingHandler(logging.Handler):
    """Console log handler that writes via tqdm.write so it doesn't clobber the
    progress bars."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(country, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"run_{country}.log")

    logger = logging.getLogger("pandemic_tgnn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Persistent, timestamped log you can `tail -f` on the VM.
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Console handler that co-operates with tqdm.
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info("Logging to %s", log_path)
    return logger


def experiment(args, config):
    """
    Run the experiment for a single country/dataset.

    Writes the results in the results folder and the best checkpoint per
    (model, shift) in the models folder.
    """
    logger = setup_logging(args.country, "../results")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cpu":
        logger.warning(
            "CUDA not available -- running on CPU. The neural models "
            "(MPNN / MPNN_LSTM / LSTM) will be VERY slow. If this VM is "
            "supposed to have a GPU, check your torch/CUDA install."
        )

    country_idx = config['country_idx']
    country_keys = list(country_idx.keys())

    if args.country not in country_idx:
        raise ValueError(
            f"Country '{args.country}' not found in config['country_idx']. "
            f"Available: {country_keys}"
        )

    countries_to_run = [args.country]

    # Where the best checkpoints are persisted (uploaded to GCS by the wrapper).
    # Defaults to a subdir of ../results because that path is known to be
    # writable in the container (the run log lives there); the container's
    # root filesystem is not, so '../models' would fail with PermissionError.
    models_dir = os.environ.get("MODELS_DIR", os.path.join("..", "results", "models"))
    os.makedirs(models_dir, exist_ok=True)

    for country in countries_to_run:
        idx = country_idx[country]
        logger.info("========== Country: %s (config index %d) ==========", country, idx)

        # ---- Read ONLY this country's dataset (not all four) -------------- #
        logger.info("Reading dataset for %s only...", country)
        t0 = time.time()
        meta_labs, meta_graphs, meta_features, meta_y = read_single_dataset(
            args.window, config, country_keys, idx
        )
        logger.info("Dataset read in %.1fs", time.time() - t0)

        # Single-country read -> everything lives at position 0.
        labels = meta_labs[0]
        gs_adj = meta_graphs[0]
        features = meta_features[0]
        y = meta_y[0]
        n_samples = len(gs_adj)
        nfeat = meta_features[0][0].shape[1]
        n_nodes = gs_adj[0].shape[0]

        logger.info(
            "n_samples=%d | n_nodes=%d | nfeat=%d | shifts=1..%d",
            n_samples, n_nodes, nfeat, args.ahead - 1,
        )

        results_dir = "../results"
        os.makedirs(results_dir, exist_ok=True)
        fw = open(os.path.join(results_dir, f"results_{country}.csv"), "a")

        for model_name in args.models:
            model_start = time.time()
            logger.info("---------- Model: %s ----------", model_name)

            # ---- ARIMA predicts by default all days until ahead ----------- #
            if model_name == "ARIMA":
                error, var = arima(args.ahead, args.start_exp, n_samples, labels)
                count = len(range(args.start_exp, n_samples - args.ahead))
                for i, e in enumerate(error):
                    fw.write(
                        "ARIMA," + str(i)
                        + ",{:.5f}".format(e / (count * n_nodes))
                        + ",{:.5f}".format(np.std(var[i])) + "\n"
                    )
                fw.flush()
                logger.info("ARIMA done in %.1fs", time.time() - model_start)
                continue

            is_neural = model_name in ("MPNN", "MPNN_LSTM", "MPNN_LSTM_Gated", "LSTM")

            # ---- Predict each day until you reach `ahead` ----------------- #
            shift_range = list(range(1, args.ahead))
            for shift in tqdm(
                shift_range, desc=f"{country} | {model_name} (shifts)", unit="shift"
            ):
                shift_start = time.time()
                result = []
                exp = 0

                sample_range = range(args.start_exp, n_samples - shift)
                # Inner bar only for the expensive neural models (each iteration
                # here is a full model training).
                sample_iter = (
                    tqdm(sample_range, desc=f"  shift {shift}", unit="fit", leave=False)
                    if is_neural else sample_range
                )

                for test_sample in sample_iter:
                    exp += 1

                    # --------------------- Baselines ----------------------- #
                    if model_name == "AVG":
                        avg = labels.iloc[:, :test_sample - 1].mean(axis=1)
                        targets_lab = labels.iloc[:, test_sample + shift]
                        error = np.sum(abs(avg - targets_lab)) / n_nodes
                        result.append(error)

                    elif model_name == "LAST_DAY":
                        win_lab = labels.iloc[:, test_sample - 1]
                        targets_lab = labels.iloc[:, test_sample + shift]
                        error = np.sum(abs(win_lab - targets_lab)) / n_nodes
                        if not np.isnan(error):
                            result.append(error)
                        else:
                            exp -= 1

                    elif model_name == "AVG_WINDOW":
                        win_lab = labels.iloc[:, (test_sample - args.window):test_sample]
                        targets_lab = labels.iloc[:, test_sample + shift]
                        error = np.sum(abs(win_lab.mean(1) - targets_lab)) / n_nodes
                        if not np.isnan(error):
                            result.append(error)
                        else:
                            exp -= 1

                    else:
                        # ----------------- Neural models ------------------- #
                        idx_train = list(range(args.window - 1, test_sample - args.sep))
                        idx_val = list(range(test_sample - args.sep, test_sample, 2))
                        idx_train = idx_train + list(
                            range(test_sample - args.sep + 1, test_sample, 2)
                        )

                        error = run_neural_model(
                            model_name, n_nodes, args.early_stop,
                            idx_train, args.window, shift,
                            args.batch_size, y, device, test_sample,
                            args.graph_window, args.recur, gs_adj,
                            features, idx_val, args.hidden, args.dropout,
                            args.lr, nfeat, args.epochs,
                            model_dir=models_dir,
                            tag=f"{country}_{model_name}_shift{shift}",
                        )
                        result.append(error)

                mean_err = float(np.mean(result)) if result else float("nan")
                std_err = float(np.std(result)) if result else float("nan")

                # Same CSV format as before: model,shift,mean,std
                fw.write("{},{},{:.5f},{:.5f}\n".format(
                    model_name, shift, mean_err, std_err))
                fw.flush()  # so partial results survive if the job dies

                logger.info(
                    "%s | shift %d/%d done in %.1fs | MAE=%.5f (n=%d)",
                    model_name, shift, args.ahead - 1,
                    time.time() - shift_start, mean_err, len(result),
                )

            logger.info(
                "Model %s finished in %.1f min",
                model_name, (time.time() - model_start) / 60.0,
            )

        fw.close()

    logger.info("All done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur', default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')

    # ---- Pick the dataset and which models to run ------------------------- #
    parser.add_argument('--country', type=str, default='England',
                        help='Which country/dataset to run (must exist in config country_idx).')
    parser.add_argument('--models', nargs='+',
                        default=["MPNN", "MPNN_LSTM", "LSTM", "ARIMA", "AVG_WINDOW", "AVG"],
                        help='Which models to run. e.g. --models MPNN_LSTM to run only the TGNN.')

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    args = parser.parse_args()
    experiment(args, config)