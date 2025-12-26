import pandas as pd

TARGET_COL = "Class"
AMOUNT_COL = "Amount"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic validation
    assert TARGET_COL in df.columns, "Target column missing"
    assert AMOUNT_COL in df.columns, "Amount column missing"

    # Ensure binary target
    assert set(df[TARGET_COL].unique()).issubset({0, 1})

    return df