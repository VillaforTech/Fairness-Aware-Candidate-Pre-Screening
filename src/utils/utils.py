from pathlib import Path

import pandas as pd


def save_predictions(model_name, y_true, y_pred, index, out_dir="data/predictions"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({"index": index, "y_true": y_true.values, "y_pred": y_pred})

    file_path = out / f"{model_name}_preds.csv"
    df_out.to_csv(file_path, index=False)

    print(f" Saved predictions to {file_path}")
