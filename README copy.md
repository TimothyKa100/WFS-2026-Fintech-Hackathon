# JSON Model Inference Scripts

These scripts take raw/scraped data, run the **same feature engineering pipeline** from your training files, align to the JSON booster feature names, and write prediction CSVs.

## Scripts

- `inference/predict_1h_from_json.py` → `1 hour ahead.py`
- `inference/predict_volatility15m_from_json.py` → `Volatility.py`
- `inference/predict_market_regime_from_json.py` → `market regime.py`
- `inference/predict_stock7d_cls_from_json.py` → `7 days ahead stock copy.py`
- `inference/predict_stock_vol_from_json.py` → `stock volatility.py`

## Scope Options

- `--scope sample` (default): predict only `id` rows in `sample_submission.csv`
- `--scope latest`: predict only latest timestamp for all symbols
- `--scope all`: predict all available timestamps/symbols

## Commands (from repo root)

Use your venv Python:

`D:/Hackathon/avenir-hku-web/venv/Scripts/python.exe`

### 1-hour return (1 hour ahead)

```powershell
D:/Hackathon/avenir-hku-web/venv/Scripts/python.exe inference/predict_1h_from_json.py `
  --model-glob "models/20260228_215655_model_fold*.json" `
  --scope sample `
  --sample-submission sample_submission.csv `
  --out submit_1h.csv
```

### Volatility 15m/6h style (`Volatility.py` groups)

```powershell
D:/Hackathon/avenir-hku-web/venv/Scripts/python.exe inference/predict_volatility15m_from_json.py `
  --model-glob "models/20260301_001611_model_fold*.json" `
  --scope sample `
  --sample-submission sample_submission.csv `
  --out submit_vol15m.csv
```

### Market regime

```powershell
D:/Hackathon/avenir-hku-web/venv/Scripts/python.exe inference/predict_market_regime_from_json.py `
  --model-glob "models/20260228_232438_market_regime_fold*.json" `
  --scope all `
  --out market_regime_infer.csv
```

### Stock 7-day classification

```powershell
D:/Hackathon/avenir-hku-web/venv/Scripts/python.exe inference/predict_stock7d_cls_from_json.py `
  --model-glob "models/<your_stock7d_prefix>_fold*.json" `
  --scope sample `
  --sample-submission sample_submission.csv `
  --out stock_7d_cls_infer.csv
```

### Stock volatility

```powershell
D:/Hackathon/avenir-hku-web/venv/Scripts/python.exe inference/predict_stock_vol_from_json.py `
  --model-glob "models/<your_stockvol_prefix>_fold*.json" `
  --scope sample `
  --sample-submission sample_submission.csv `
  --out stock_vol_infer.csv
```

## Notes

- You pass **raw scraped data**, not pre-computed features.
- Scripts recompute features and then select the exact columns required by booster feature names.
- For stock scripts, data is read from `SCRAPED_DATA`/`SCRAPPED_DATA` (same as training file logic).
- For 1h/Volatility/market-regime scripts, data is read from `kline_data/train_data` and cache paths used by those files.
