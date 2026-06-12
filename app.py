import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean paths & constants at the top
DATA_PATH = "Mall_Customers.csv"
N_CLUSTERS = 5
RANDOM_STATE = 42

# Define cluster information with personas, descriptions and colors
CLUSTER_INFO = {
    0: {
        "name": "Standard Shoppers",
        "desc": "Customers with average annual income and average spending score. They represent the middle-class baseline shopper.",
        "color": "#9b5de5", # Vibrant Purple
        "icon": "🛒"
    },
    1: {
        "name": "Target VIPs",
        "desc": "High annual income and high spending score. These are the most valuable customers who should be prioritized for loyalty programs.",
        "color": "#00bbf9", # Sleek Blue
        "icon": "💎"
    },
    2: {
        "name": "Impulsive Buyers",
        "desc": "Low annual income but high spending score. They are highly active and likely purchase on impulse or trend-based promotions.",
        "color": "#f15bb5", # Hot Pink
        "icon": "🛍️"
    },
    3: {
        "name": "Careful Spenders",
        "desc": "High annual income but low spending score. They have the capacity to spend more but are conservative; prime targets for premium marketing.",
        "color": "#fee440", # Sunny Yellow
        "icon": "🛡️"
    },
    4: {
        "name": "Budget-Conscious",
        "desc": "Low annual income and low spending score. Highly price-sensitive shoppers who buy essentials and focus on budget constraints.",
        "color": "#00f5d4", # Mint Green
        "icon": "🏷️"
    }
}

# Inject premium custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff007f, #7f00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1.1rem;
    color: #8a8d90;
    margin-bottom: 2rem;
}
.sidebar-header {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 1rem;
}
.card {
    background-color: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.custom-metric {
    background-color: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 105px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.custom-metric:hover {
    transform: translateY(-5px);
    border-color: rgba(255, 0, 127, 0.4);
    box-shadow: 0 10px 20px rgba(255, 0, 127, 0.15), 0 0 8px rgba(255, 0, 127, 0.10);
}
</style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at '{file_path}'. Please check the path.")
        st.stop()

# Load and compute models
df_raw = load_data(DATA_PATH)
df = df_raw.copy()

# Features for clustering (as in original notebook: Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=RANDOM_STATE)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster Name'] = df['Cluster'].map(lambda x: CLUSTER_INFO[x]['name'])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA_1'] = X_pca[:, 0]
df['PCA_2'] = X_pca[:, 1]

# Header section
st.markdown('<div class="main-title">Customer Segmentation Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive AI dashboard powered by K-Means & PCA dimension reduction</div>', unsafe_allow_html=True)

# Sidebar - Customer Profiler
with st.sidebar:
    st.markdown('<div class="sidebar-header">👥 Live Customer Profiler</div>', unsafe_allow_html=True)
    st.write("Simulate a new customer's characteristics to predict their segment dynamically.")
    
    with st.form("profiler_form"):
        gender = st.selectbox("Gender", ["Female", "Male", "Both/Other"])
        age = st.slider("Age", min_value=18, max_value=70, value=30)
        income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=50)
        score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        
        submit_btn = st.form_submit_button("Analyze Profile", use_container_width=True)
        
    if submit_btn:
        # Assign a safe, neutral numeric value for Gender (e.g. 0.5 for Both/Other)
        gender_val = 0.5 if gender == "Both/Other" else (1.0 if gender == "Male" else 0.0)
        
        # Scale and predict cluster (Annual Income and Spending Score are inputs to the scaler)
        scaled_input = scaler.transform([[income, score]])
        pred_cluster = kmeans.predict(scaled_input)[0]
        pca_coords = pca.transform(scaled_input)[0]
        
        st.session_state['new_customer'] = {
            'Gender': gender,
            'GenderVal': gender_val,
            'Age': age,
            'Income': income,
            'Score': score,
            'PCA_1': pca_coords[0],
            'PCA_2': pca_coords[1],
            'Cluster': pred_cluster,
            'ClusterName': CLUSTER_INFO[pred_cluster]['name'],
            'Color': CLUSTER_INFO[pred_cluster]['color'],
            'Desc': CLUSTER_INFO[pred_cluster]['desc'],
            'Icon': CLUSTER_INFO[pred_cluster]['icon']
        }
        
    if 'new_customer' in st.session_state:
        nc = st.session_state['new_customer']
        
        # Display custom profiling card in sidebar
        st.markdown("---")
        st.markdown(f"""
        <div style="background-color: {nc['Color']}15; border: 1.5px solid {nc['Color']}; border-radius: 12px; padding: 15px; margin-top: 10px;">
            <h4 style="color: {nc['Color']}; margin: 0; font-size: 1.15rem;">{nc['Icon']} {nc['ClusterName']}</h4>
            <p style="font-size: 0.85rem; margin-top: 8px; color: inherit; opacity: 0.85;"><b>Cluster {nc['Cluster']}</b>: {nc['Desc']}</p>
            <hr style="margin: 8px 0; border-color: rgba(128,128,128,0.2);">
            <div style="font-size: 0.8rem; display: flex; justify-content: space-between;">
                <span>Age: <b>{nc['Age']}</b></span>
                <span>Income: <b>${nc['Income']}k</b></span>
                <span>Score: <b>{nc['Score']}/100</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Simulation", use_container_width=True):
            del st.session_state['new_customer']
            st.rerun()

# Tabs selection
tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Overview", 
    "💡 Cluster Characteristic Insights", 
    "🗺️ Interactive 2D PCA Cluster Map"
])

# Tab 1: Dataset Overview
with tab1:
    st.write("")
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="custom-metric">
            <div style="font-size: 0.875rem; color: #8a8d90; margin-bottom: 4px; font-weight: 400;">Total Customer Records</div>
            <div style="font-size: 2rem; font-weight: 700; line-height: 1.2;">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="custom-metric">
            <div style="font-size: 0.875rem; color: #8a8d90; margin-bottom: 4px; font-weight: 400;">Optimized Clusters Identified</div>
            <div style="font-size: 2rem; font-weight: 700; line-height: 1.2;">{N_CLUSTERS}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="custom-metric">
            <div style="font-size: 0.875rem; color: #8a8d90; margin-bottom: 4px; font-weight: 400;">Features Scaled & Modeled</div>
            <div style="font-size: 1.15rem; font-weight: 700; line-height: 1.3; word-wrap: break-word; margin-top: 6px;">Annual Income, Spending Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.subheader("Clean Raw Data Preview")
    st.write("Browse, sort, and filter the dataset elements directly below.")
    st.dataframe(df_raw, use_container_width=True, height=350)

# Tab 2: Cluster Insights
with tab2:
    st.write("")
    st.subheader("Mathematical Means per Segment")
    st.write("The matrix below displays the average demographic and spending attributes for each customer segment.")
    
    # Calculate group-by means
    summary_df = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Customer Count'})
    
    # Add persona names
    summary_df['Persona Label'] = [CLUSTER_INFO[i]['name'] for i in summary_df.index]
    summary_df = summary_df[['Persona Label', 'Customer Count', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Format display values: round Age to whole number and others to 2 decimals
    display_df = summary_df.copy()
    display_df['Annual Income (k$)'] = display_df['Annual Income (k$)'].round(2)
    display_df['Spending Score (1-100)'] = display_df['Spending Score (1-100)'].round(2)
    display_df['Age'] = display_df['Age'].round(0).astype(int)
    
    st.table(display_df)
    
    st.write("")
    st.subheader("Cluster Persona Descriptions")
    
    # Calculate dominant gender (mode) and its percentage for each cluster
    def get_dominant_gender_str(genders):
        mode_gender = genders.mode()[0]
        mode_count = (genders == mode_gender).sum()
        pct = (mode_count / len(genders)) * 100
        return f"{mode_gender} ({int(round(pct))}%)"
        
    dominant_genders = df.groupby('Cluster')['Gender'].agg(get_dominant_gender_str)
    
    # Render interactive cards for personas
    col_a, col_b = st.columns(2)
    for idx, (cid, info) in enumerate(CLUSTER_INFO.items()):
        target_col = col_a if idx % 2 == 0 else col_b
        with target_col:
            # Average variables for this specific cluster
            c_stats = summary_df.loc[cid]
            dom_gender = dominant_genders.loc[cid]
            with st.container(border=True):
                st.markdown(f"#### {info['icon']} {info['name']} (Cluster {cid})")
                st.write(info['desc'])
                st.markdown(f"""
                * **Volume**: {int(c_stats['Customer Count'])} shoppers ({int(c_stats['Customer Count'] / len(df) * 100)}%)
                * **Dominant Gender**: {dom_gender}
                * **Average Age**: {int(round(c_stats['Age']))} years old
                * **Average Annual Income**: ${c_stats['Annual Income (k$)']:.1f}k
                * **Average Spending Score**: {c_stats['Spending Score (1-100)']:.1f} / 100
                """)

# Tab 3: PCA Cluster Map
with tab3:
    st.write("")
    st.subheader("Customer Segments Projection")
    st.write("This map plots the 2D PCA representation of customers. PCA reduces our scaled financial indicators (Annual Income and Spending Score) to reveal how customers cluster together.")
    
    # Map colours
    color_map = {info['name']: info['color'] for info in CLUSTER_INFO.values()}
    
    # Base Scatter Plot
    fig = px.scatter(
        df,
        x='PCA_1',
        y='PCA_2',
        color='Cluster Name',
        color_discrete_map=color_map,
        category_orders={"Cluster Name": [info['name'] for info in CLUSTER_INFO.values()]},
        hover_name='Cluster Name',
        hover_data={
            'PCA_1': False,
            'PCA_2': False,
            'Age': True,
            'Annual Income (k$)': True,
            'Spending Score (1-100)': True,
            'Cluster Name': False
        },
        labels={'PCA_1': 'PCA Component 1', 'PCA_2': 'PCA Component 2', 'Cluster Name': 'Customer Segment'},
        template="plotly_dark"
    )
    
    # Customise plot style
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color='white')))
    
    # Overlay new customer if simulated
    if 'new_customer' in st.session_state:
        nc = st.session_state['new_customer']
        fig.add_trace(
            go.Scatter(
                x=[nc['PCA_1']],
                y=[nc['PCA_2']],
                mode='markers',
                marker=dict(
                    color='#ffffff',
                    size=18,
                    symbol='star-dot',
                    line=dict(color=nc['Color'], width=2.5)
                ),
                name='Simulated Customer',
                hoverinfo='text',
                hovertext=(
                    f"<b>⭐ SIMULATED CUSTOMER</b><br>"
                    f"Assigned Cluster: {nc['ClusterName']} (Cluster {nc['Cluster']})<br>"
                    f"Gender: {nc['Gender']}<br>"
                    f"Age: {nc['Age']}<br>"
                    f"Annual Income: ${nc['Income']}k<br>"
                    f"Spending Score: {nc['Score']}/100"
                )
            )
        )
        st.info("💡 A simulated customer is currently plotted as a highlighted star on the map below.")
        
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
