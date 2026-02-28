from __future__ import annotations

import argparse
from pathlib import Path

from Backend.features.anomaly import build_anomaly_report, save_report_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling z-score anomaly detection from train_data + data_cache"
    )
    parser.add_argument("--train-data", default="train_data", help="Path to train_data directory")
    parser.add_argument("--data-cache", default="data_cache", help="Path to data_cache directory")
    parser.add_argument("--window", type=int, default=96, help="Rolling window size in bars")
    parser.add_argument("--z-threshold", type=float, default=3.0, help="Absolute z-score threshold")
    parser.add_argument(
        "--out-json",
        default="data_cache/anomaly_summary.json",
        help="Output path for anomaly summary JSON",
    )
    parser.add_argument(
        "--skip-parquet-flags",
        action="store_true",
        help="Do not write per-indicator anomaly flag parquet files",
    )
    parser.add_argument(
        "--max-indicators",
        type=int,
        default=None,
        help="Optional cap on number of indicator files to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = build_anomaly_report(
        data_cache_dir=Path(args.data_cache),
        train_data_dir=Path(args.train_data),
        z_threshold=args.z_threshold,
        window=args.window,
        write_parquet=not args.skip_parquet_flags,
        max_indicators=args.max_indicators,
    )

    save_report_json(report, args.out_json)

    print("Anomaly detection complete")
    print(f"Symbols in train_data: {report['symbols_in_train']}")
    print(f"Indicators scanned: {report['indicators_scanned']}")
    print(f"Indicators with current anomalies: {report['indicators_with_flags']}")
    print(f"Total current anomaly flags: {report['total_current_flags']}")
    print(f"Summary saved to: {args.out_json}")


if __name__ == "__main__":
    main()
