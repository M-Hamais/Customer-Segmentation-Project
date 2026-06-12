# 👥 Customer Segmentation Using K-Means & PCA

🚀 **[Launch Live Interactive Web App](Yhttps://customer-segmentation-project-1.streamlit.app/)**

---

## 🔍 Project Overview
This project demonstrates an end-to-end unsupervised machine learning pipeline to segment retail customers based on their purchasing behavior. By clustering customers into distinct groups, this model translates raw transaction data into actionable business and marketing insights.

### 💡 What this portal does
This interactive platform helps retail businesses instantly understand their customer base. Instead of looking at thousands of confusing individual rows, our AI analyzes two key shopping behaviors: how much a customer earns annually and how much they choose to spend at the store.

### 🎯 The Real-World Goal
The system automatically groups buyers with identical shopping habits into 5 distinct 'customer personas.' This allows business managers to design personalized discount offers, create targeted loyalty rewards for VIP shoppers, and craft smart marketing strategies that fit every budget level perfectly.

---

## 💻 Portal Interface & Layout Enhancements
To make these insights accessible, we built a live interactive web app with the following features:
* **🗺️ Interactive 2D PCA Cluster Map:** An interactive visualization projecting high-dimensional customer segments down to a 2D space. Includes real-time marker overlays to dynamically profile and plot simulated custom entries.
* **💡 Live Customer Profiler Sidebar:** Enter a simulated customer's demographics (Age, Gender, Annual Income, Spending Score) to instantly predict their cluster segment and coordinates on the PCA map.
* **📊 Mathematical Means & Formatted Tables:** Clear demographic tables featuring rounded metrics (Annual Income formatted to exactly one decimal place, Spending Score converted to integers) for high-scannability and clean layout.
* **📱 Responsive CSS Layout:** Custom HTML/CSS styled container cards built with flexible layouts and strict dimensions (`min-height` rules) to remain perfectly uniform across screen sizes (mobile, 14", 15.6", and 17" laptops).

---

## 🎯 Objectives
* ⚙️ Apply unsupervised learning to identify hidden patterns in consumer data.
* 📈 Utilize the **Elbow Method** to mathematically determine the optimal number of customer segments.
* 🧪 Implement **Principal Component Analysis (PCA)** for dimensionality reduction and 2D data visualization.
* 🔎 Extract demographic insights to understand the composition of high-value versus budget-conscious consumers.

---

## 🛠️ Tech Stack
* **Language:** Python 🐍
* **App Framework:** Streamlit 🎈 / HTML5 / CSS3
* **Environment:** Jupyter Notebook / VS Code 💻
* **Libraries:** Pandas, NumPy, Plotly, Scikit-Learn (K-Means, PCA, StandardScaler)

---

## 🧪 Methodology
1. **🧹 Data Preprocessing:** Handled missing values and applied mandatory feature scaling (`StandardScaler`) to ensure balanced distance calculations.
2. **🤖 Clustering:** Deployed the K-Means algorithm, utilizing the Elbow Method to identify `K=5` as the optimal number of clusters.
3. **📉 Dimensionality Reduction:** Compressed feature dimensions using PCA to generate a clear, color-coded 2D scatter plot of the customer segments.
4. **👥 Demographic Analysis:** Extracted the percentage breakdown of male and female shoppers within each generated cluster.

---

## 📈 Key Business Insights
The model successfully grouped the customer base into five actionable target demographics:

* **💎 Target VIPs (High Income / High Spending):** The most valuable loyal customers (54% Female, 46% Male).
* **🛒 Standard Shoppers (Average Income / Average Spending):** The everyday middle-class buyer.
* **🛡️ Careful Spenders (High Income / Low Spending):** A prime demographic for targeted marketing campaigns (Majority Male at 54%).
* **🛍️ Impulsive Buyers (Low Income / High Spending):** Customers making infrequent luxury purchases.
* **🏷️ Budget-Conscious (Low Income / Low Spending):** Customers strictly buying essential items (Highest Female majority at 61%).