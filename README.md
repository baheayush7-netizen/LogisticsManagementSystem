# Logistics Data Exploration and Strategic Planning

## Project overview
This project demonstrates how Python can support logistics planning through data cleaning, KPI reporting, baseline delivery-time regression, and exploratory order clustering.

## Important note
The program generates **synthetic demonstration data**. No real company data was provided. Do not report its outputs as real operational findings.

## KPIs calculated
- On-time delivery rate
- Average delivery time and delay
- Average transport cost per order
- Inventory stockout rate (illustrative field)
- Average vehicle capacity utilization

## Methods
- Data cleaning with pandas
- Linear regression with scikit-learn to estimate delivery duration
- K-means clustering to explore order profiles
- KPI grouping by traffic level

## Setup
Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
python logistics_analysis.py
```

## Outputs
The script creates an `outputs/` folder containing:
- `sample_logistics_data.csv`
- `cleaned_logistics_data.csv`
- `kpi_summary.csv`
- `regression_model_metrics.csv`
- `delivery_time_predictions.csv`
- `order_clusters.csv`
- `traffic_level_summary.csv`

## Limitations and next steps
The sample data is synthetic and the regression model is for demonstration only. For a real project, replace the data-generation function with approved operational extracts, validate KPI definitions with the logistics team, use time-based validation, and assess route optimization with real locations, vehicle capacities, driver hours, and delivery time windows. Route optimization is discussed in the report but is not executed by this starter script.
