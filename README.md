# Customer Segmentation Using K-Means & PCA

## Project Overview
This project demonstrates an end-to-end unsupervised machine learning pipeline to segment retail customers based on their purchasing behavior. By clustering customers into distinct groups, this model translates raw transaction data into actionable business and marketing insights. 

## Objectives
* Apply unsupervised learning to identify hidden patterns in consumer data.
* Utilize the **Elbow Method** to mathematically determine the optimal number of customer segments.
* Implement **Principal Component Analysis (PCA)** for dimensionality reduction and 2D data visualization.
* Extract demographic insights to understand the composition of high-value versus budget-conscious consumers.

## Tech Stack
* **Language:** Python
* **Environment:** Jupyter Notebook / VS Code
* **Libraries:** Pandas, Scikit-Learn (K-Means, PCA, StandardScaler), Matplotlib

## Methodology
1. **Data Preprocessing:** Handled missing values and applied mandatory feature scaling (`StandardScaler`) to ensure balanced distance calculations.
2. **Clustering:** Deployed the K-Means algorithm, utilizing the Elbow Method to identify `K=5` as the optimal number of clusters.
3. **Dimensionality Reduction:** Compressed feature dimensions using PCA to generate a clear, color-coded 2D scatter plot of the customer segments.
4. **Demographic Analysis:** Extracted the percentage breakdown of male and female shoppers within each generated cluster.

## Key Business Insights
The model successfully grouped the customer base into five actionable target demographics:

* **Target VIPs (High Income / High Spending):** The most valuable loyal customers (54% Female, 46% Male).
* **Standard Shoppers (Average Income / Average Spending):** The everyday middle-class buyer.
* **Careful Spenders (High Income / Low Spending):** A prime demographic for targeted marketing campaigns (Majority Male at 54%).
* **Impulsive Buyers (Low Income / High Spending):** Customers making infrequent luxury purchases.
* **Budget-Conscious (Low Income / Low Spending):** Customers strictly buying essential items (Highest Female majority at 61%).