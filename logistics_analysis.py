"""
Logistics Data Exploration and Strategic Planning
------------------------------------------------
Demonstration project using synthetic logistics data.

Run:
    pip install -r requirements.txt
    python logistics_analysis.py

Outputs are written to the outputs/ folder.
IMPORTANT: The generated dataset is synthetic and must not be described as real company data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_sample_data(n: int = 120) -> pd.DataFrame:
    """Create reproducible synthetic delivery records for demonstration."""
    rng = np.random.default_rng(RANDOM_SEED)
    distance = rng.integers(5, 81, size=n)
    weight = rng.integers(20, 801, size=n)
    traffic = rng.integers(1, 4, size=n)  # 1=low, 2=medium, 3=high
    stops = rng.integers(1, 11, size=n)
    vehicle_capacity = rng.choice([500, 750, 1000], size=n)
    promised_minutes = 20 + distance * 1.2 + stops * 3
    noise = rng.normal(0, 7, size=n)
    delivery_minutes = np.maximum(
        10, 15 + distance * 1.15 + traffic * 6 + stops * 2.2 + weight / 120 + noise
    )
    on_time = delivery_minutes <= promised_minutes
    transport_cost = 55 + distance * 4.2 + weight * 0.035 + stops * 8 + traffic * 5
    stockout = rng.choice([0, 1], size=n, p=[0.88, 0.12])

    return pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(1, n + 1)],
        "distance_km": distance,
        "order_weight_kg": weight,
        "traffic_level": traffic,
        "number_of_stops": stops,
        "vehicle_capacity_kg": vehicle_capacity,
        "promised_minutes": np.round(promised_minutes, 1),
        "delivery_minutes": np.round(delivery_minutes, 1),
        "transport_cost_inr": np.round(transport_cost, 2),
        "on_time": on_time.astype(int),
        "stockout": stockout,
    })


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic quality checks and cleaning."""
    df = df.drop_duplicates(subset="order_id").copy()
    numeric_columns = [
        "distance_km", "order_weight_kg", "traffic_level", "number_of_stops",
        "vehicle_capacity_kg", "promised_minutes", "delivery_minutes",
        "transport_cost_inr", "on_time", "stockout",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_columns)
    df = df[
        (df["distance_km"] >= 0)
        & (df["order_weight_kg"] >= 0)
        & (df["delivery_minutes"] >= 0)
        & (df["transport_cost_inr"] >= 0)
    ]
    df["delay_minutes"] = (
        df["delivery_minutes"] - df["promised_minutes"]
    ).clip(lower=0)
    df["capacity_utilization_pct"] = (
        df["order_weight_kg"] / df["vehicle_capacity_kg"] * 100
    ).clip(upper=100)
    return df


def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate core operational KPIs."""
    kpis = {
        "Total orders": len(df),
        "On-time delivery rate (%)": df["on_time"].mean() * 100,
        "Average delivery time (minutes)": df["delivery_minutes"].mean(),
        "Average delay among all orders (minutes)": df["delay_minutes"].mean(),
        "Average transport cost per order (INR)": df["transport_cost_inr"].mean(),
        "Inventory stockout rate (%)": df["stockout"].mean() * 100,
        "Average vehicle capacity utilization (%)": df["capacity_utilization_pct"].mean(),
    }
    return pd.DataFrame(
        [{"KPI": key, "Value": round(float(value), 2)} for key, value in kpis.items()]
    )


def train_delivery_time_model(df: pd.DataFrame):
    """Train a baseline regression model to estimate delivery duration."""
    features = [
        "distance_km", "order_weight_kg", "traffic_level",
        "number_of_stops", "vehicle_capacity_kg",
    ]
    X = df[features]
    y = df["delivery_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = pd.DataFrame([{
        "Model": "Linear Regression",
        "Test MAE (minutes)": round(float(mean_absolute_error(y_test, predictions)), 2),
        "Test R-squared": round(float(r2_score(y_test, predictions)), 3),
        "Test records": len(y_test),
        "Note": "Synthetic demonstration data; not a validated production model."
    }])
    prediction_output = X_test.copy()
    prediction_output["actual_delivery_minutes"] = y_test.values
    prediction_output["predicted_delivery_minutes"] = np.round(predictions, 2)
    return model, features, metrics, prediction_output


def cluster_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Group orders by distance and weight to explore delivery profiles."""
    features = df[["distance_km", "order_weight_kg", "number_of_stops"]]
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
    result = df[["order_id", "distance_km", "order_weight_kg", "number_of_stops"]].copy()
    result["cluster"] = model.fit_predict(scaled)
    return result


def main():
    print("Creating synthetic logistics dataset...")
    raw = create_sample_data()
    raw.to_csv(OUTPUT_DIR / "sample_logistics_data.csv", index=False)

    df = clean_data(raw)
    df.to_csv(OUTPUT_DIR / "cleaned_logistics_data.csv", index=False)

    kpis = calculate_kpis(df)
    kpis.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)

    _, _, model_metrics, predictions = train_delivery_time_model(df)
    model_metrics.to_csv(OUTPUT_DIR / "regression_model_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "delivery_time_predictions.csv", index=False)

    clusters = cluster_orders(df)
    clusters.to_csv(OUTPUT_DIR / "order_clusters.csv", index=False)

    zone_cost = df.groupby("traffic_level", as_index=False).agg(
        order_count=("order_id", "count"),
        average_transport_cost_inr=("transport_cost_inr", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
        on_time_rate_pct=("on_time", lambda s: s.mean() * 100),
    )
    zone_cost.to_csv(OUTPUT_DIR / "traffic_level_summary.csv", index=False)

    print("\n=== KPI SUMMARY ===")
    print(kpis.to_string(index=False))
    print("\n=== REGRESSION EVALUATION ===")
    print(model_metrics.to_string(index=False))
    print("\n=== Traffic-level summary ===")
    print(zone_cost.round(2).to_string(index=False))
    print(f"\nDone. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
