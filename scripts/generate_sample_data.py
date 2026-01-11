"""Script to generate synthetic customer churn data."""

import random
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer churn data.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic customer data
    """
    random.seed(seed)
    np.random.seed(seed)

    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(n_samples)],
        "age": np.random.randint(18, 81, size=n_samples),
        "income": np.clip(
            np.random.normal(50000, 20000, size=n_samples),
            10000,
            200000,
        ).round(2),
        "gender": np.random.choice(
            ["M", "F", "Other"],
            size=n_samples,
            p=[0.48, 0.48, 0.04],
        ),
        "region": np.random.choice(
            ["North", "South", "East", "West"],
            size=n_samples,
            p=[0.25, 0.25, 0.25, 0.25],
        ),
        "tenure_months": np.random.randint(0, 73, size=n_samples),
        "monthly_charges": np.random.uniform(20, 100, size=n_samples).round(2),
        "total_charges": np.random.uniform(100, 5000, size=n_samples).round(2),
        "churn": np.random.choice(
            [0, 1],
            size=n_samples,
            p=[0.73, 0.27],
        ),
    }

    return pd.DataFrame(data)


def main():
    """Generate and save sample data."""
    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    df = generate_sample_data(n_samples=1000)

    # Save to CSV
    output_path = output_dir / "customers.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} customer records")
    print(f"Saved to {output_path}")
    print(f"\nChurn distribution:")
    print(df["churn"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
