
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('../data/ecommerce_data.csv', encoding='ISO-8859-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.dropna(inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# RFM Analysis
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Normalize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# PCA
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# KMeans clustering
sse = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(rfm_scaled)
    sse.append(km.inertia_)

plt.figure()
plt.plot(range(1, 10), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.savefig('../visuals/elbow_plot.png')

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
rfm['PCA1'] = rfm_pca[:, 0]
rfm['PCA2'] = rfm_pca[:, 1]

# Plot clusters
plt.figure()
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Customer Segments')
plt.savefig('../visuals/cluster_plot.png')

# CLV prediction
rfm['CLV'] = rfm['Monetary'] * rfm['Frequency']

# Merge cluster info
df = df.merge(rfm[['CustomerID', 'Cluster']], on='CustomerID')

# Top products per cluster
top_products = df.groupby(['Cluster', 'Description'])['TotalPrice'].sum().reset_index()
top_products = top_products.sort_values(['Cluster', 'TotalPrice'], ascending=[True, False])
top3_products = top_products.groupby('Cluster').head(3)

# Export results
rfm.to_csv('../outputs/customer_segments.csv', index=False)
top3_products.to_csv('../outputs/top_products_per_cluster.csv', index=False)

print("Segmentation complete. Outputs saved.")
