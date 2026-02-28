from pathlib import Path

import pandas as pd


def preview_scraped_data(folder: str = "SCRAPED_DATA", sample_size: int = 5, head_rows: int = 5) -> None:
	data_dir = Path(folder)
	parquet_files = sorted(data_dir.glob("*.parquet"))

	if not parquet_files:
		print(f"No parquet files found in {data_dir.resolve()}")
		return

	selected_files = parquet_files[:sample_size]
	print(f"Found {len(parquet_files)} parquet files in {data_dir.resolve()}")
	print(f"Showing first {len(selected_files)} files\n")

	for file_path in selected_files:
		df = pd.read_parquet(file_path)
		print(f"=== {file_path.name} ===")
		print(f"Shape: {df.shape}")
		print(df.head(head_rows))
		print()


if __name__ == "__main__":
	preview_scraped_data(folder="SCRAPED_DATA", sample_size=5, head_rows=5)
