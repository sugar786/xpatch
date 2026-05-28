import os
import re
import argparse
import csv


def parse_log_file(path):
    result = {
        "log_file": path,
        "mse": None,
        "mae": None,
        "last_epoch": None,
        "last_S_P_alignment_corr": None,
        "last_P_entropy": None,
        "last_P_entropy_norm": None,
        "last_P_max_mean": None,
        "last_P_balance_gap": None,
        "last_P_dist_mean": None,
        "last_P_dist_std": None,
        "last_aux_pred_ratio": None,
    }

    mse_mae_pattern = re.compile(r"mse:([0-9eE\.\+\-]+),\s*mae:([0-9eE\.\+\-]+)")
    epoch_pattern = re.compile(r"\[CCM Diagnose\]\s*epoch:\s*([0-9]+)")
    corr_pattern = re.compile(r"S-P alignment corr:\s*([0-9eE\.\+\-]+)")
    entropy_pattern = re.compile(r"P entropy:\s*([0-9eE\.\+\-]+),\s*normalized:\s*([0-9eE\.\+\-]+)")
    pmax_pattern = re.compile(r"P max mean:\s*([0-9eE\.\+\-]+)")
    balance_pattern = re.compile(r"P balance gap:\s*([0-9eE\.\+\-]+)")
    pdist_pattern = re.compile(r"P dist mean/std:\s*([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)")
    aux_ratio_pattern = re.compile(r"aux/pred ratio:\s*([0-9eE\.\+\-]+)")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            m = mse_mae_pattern.search(line)
            if m:
                result["mse"] = float(m.group(1))
                result["mae"] = float(m.group(2))

            m = epoch_pattern.search(line)
            if m:
                result["last_epoch"] = int(m.group(1))

            m = corr_pattern.search(line)
            if m:
                result["last_S_P_alignment_corr"] = float(m.group(1))

            m = entropy_pattern.search(line)
            if m:
                result["last_P_entropy"] = float(m.group(1))
                result["last_P_entropy_norm"] = float(m.group(2))

            m = pmax_pattern.search(line)
            if m:
                result["last_P_max_mean"] = float(m.group(1))

            m = balance_pattern.search(line)
            if m:
                result["last_P_balance_gap"] = float(m.group(1))

            m = pdist_pattern.search(line)
            if m:
                result["last_P_dist_mean"] = float(m.group(1))
                result["last_P_dist_std"] = float(m.group(2))

            m = aux_ratio_pattern.search(line)
            if m:
                result["last_aux_pred_ratio"] = float(m.group(1))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="ccm_summary.csv")
    args = parser.parse_args()

    rows = []

    for root, dirs, files in os.walk(args.log_dir):
        for file in files:
            if not file.endswith(".log"):
                continue

            path = os.path.join(root, file)
            row = parse_log_file(path)
            rows.append(row)

    rows = sorted(rows, key=lambda x: x["log_file"])

    fieldnames = [
        "log_file",
        "mse",
        "mae",
        "last_epoch",
        "last_S_P_alignment_corr",
        "last_P_entropy",
        "last_P_entropy_norm",
        "last_P_max_mean",
        "last_P_balance_gap",
        "last_P_dist_mean",
        "last_P_dist_std",
        "last_aux_pred_ratio",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("Saved summary to:", args.out_csv)
    print("Parsed logs:", len(rows))


if __name__ == "__main__":
    main()
