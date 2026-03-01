import argparse
from pathlib import Path

from json_inference_common import (
    add_common_args,
    booster_objective,
    build_prediction_matrix,
    capture_factor_dfs_via_run,
    load_boosters,
    predict_ensemble,
    print_inference_header,
    write_multiclass_time_output,
)


REGIME_NAME_MAP = {
    0: "bearish_calm",
    1: "bearish_volatile",
    2: "neutral",
    3: "bullish_calm",
    4: "bullish_volatile",
}


def main():
    parser = argparse.ArgumentParser(
        description="Inference for models trained by market regime.py"
    )
    add_common_args(parser)
    args = parser.parse_args()

    boosters, model_paths = load_boosters(args.model_glob)
    expected_features = boosters[0].feature_names or []
    if not expected_features:
        raise RuntimeError("Model does not contain feature names")

    factor_dfs = capture_factor_dfs_via_run(
        module_path=str(Path("market regime.py").resolve()),
        class_name="OptimizedModel",
        module_name="market_regime_module",
        train_method_name="train",
        arg_index_factor_dfs=1,
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

    if pred.ndim != 2:
        raise RuntimeError("Expected multiclass output for market regime model")

    class_names = [REGIME_NAME_MAP[i] for i in range(pred.shape[1])]
    write_multiclass_time_output(pred, x.index, class_names=class_names, out_csv=args.out)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
