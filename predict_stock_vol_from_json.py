import argparse
from pathlib import Path

from json_inference_common import (
    add_common_args,
    booster_objective,
    build_prediction_matrix,
    factor_dfs_from_load_panel_data,
    load_boosters,
    predict_ensemble,
    print_inference_header,
    write_regression_output,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inference for models trained by stock volatility.py"
    )
    add_common_args(parser)
    args = parser.parse_args()

    boosters, model_paths = load_boosters(args.model_glob)
    expected_features = boosters[0].feature_names or []
    if not expected_features:
        raise RuntimeError("Model does not contain feature names")

    factor_dfs = factor_dfs_from_load_panel_data(
        module_path=str(Path("stock volatility.py").resolve()),
        class_name="StockVolatilityModel",
        module_name="stock_vol_module",
        factor_position=1,
    )

    x = build_prediction_matrix(
        factor_dfs=factor_dfs,
        expected_features=expected_features,
        scope=args.scope,
        sample_submission_path=args.sample_submission,
    )

    print_inference_header(model_paths, x)
    objective = booster_objective(boosters[0])
    pred = predict_ensemble(boosters, x, objective=objective)

    if pred.ndim != 1:
        raise RuntimeError("Expected regression output for stock volatility model")

    write_regression_output(pred, x.index, args.out)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
