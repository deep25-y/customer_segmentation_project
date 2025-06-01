# Customer Segmentation with RFM and K-Means

This project segments customers using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering, predicts Customer Lifetime Value (CLV), and identifies product affinities per segment.

## 📁 Folder Structure
- `data/`: Place your `ecommerce_data.csv` file here
- `outputs/`: Contains CSVs for customer segments and top products
- `visuals/`: Plots for clusters and elbow method
- `src/`: Source Python scripts

## 🚀 How to Run

1. Install requirements:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Add your dataset to `data/`
3. Run the script:
```bash
cd src
python rfm_kmeans_clv.py
```

## 🧠 Outputs
- `customer_segments.csv`: Clustered customer data with CLV
- `top_products_per_cluster.csv`: Top 3 products by cluster
- Visual plots for analysis

## 📊 Visualization
You can use Tableau or Power BI to create interactive dashboards using the output files.

Enjoy!
