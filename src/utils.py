import pandas as pd
import numpy as np

def simulate_imbalance(
    df: pd.DataFrame,
    target_col: str,
    fraud_rate: float,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Downsample fraud class to simulate real-world imbalance.
    fraud_rate = desired proportion of fraud (e.g. 0.01, 0.05)
    """

    assert 0 < fraud_rate < 0.5, "Fraud rate must be between 0 and 0.5"

    fraud_df = df[df[target_col] == 1]
    non_fraud_df = df[df[target_col] == 0]

    n_total = len(df)
    n_fraud_desired = int(fraud_rate * n_total)

    fraud_sampled = fraud_df.sample(
        n=n_fraud_desired,
        random_state=random_state
    )

    simulated_df = pd.concat([fraud_sampled, non_fraud_df])
    simulated_df = simulated_df.sort_index()  # preserve order

    return simulated_df