import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Zomato Restaurant Analysis", layout="wide", page_icon="🍽️")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #E23744;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E3B4E;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #E23744;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data(uploaded_file=None):
    """Load the Zomato dataset from uploaded file or default location"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            return df
        else:
            # Try to load from common paths
            try:
                df = pd.read_csv('zomato.csv')
                return df
            except:
                try:
                    df = pd.read_csv('data/zomato.csv')
                    return df
                except:
                    return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Data preprocessing function
@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset"""
    if df is None:
        return None
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Handle missing values
    if 'rate' in df.columns:
        df['rate'] = df['rate'].replace('NEW', np.nan).replace('-', np.nan)
        df['rate'] = df['rate'].astype(str).str.replace('/5', '').str.strip()
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    
    # Convert votes to numeric
    if 'votes' in df.columns:
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    
    # Convert approx_cost to numeric
    if 'approx_cost(for two people)' in df.columns:
        df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '')
        df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
    
    # Convert online_order and book_table to binary (create new columns, keep originals)
    if 'online_order' in df.columns:
        df['online_order_binary'] = df['online_order'].map({'Yes': 1, 'No': 0})
        df['online_order'] = df['online_order'].fillna('Unknown')
    if 'book_table' in df.columns:
        df['book_table_binary'] = df['book_table'].map({'Yes': 1, 'No': 0})
        df['book_table'] = df['book_table'].fillna('Unknown')
    
    return df

# Main app
def main():
    st.markdown('<h1 class="main-header">🍽️ Zomato Restaurant Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # File uploader in sidebar
    st.sidebar.title("📂 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Zomato CSV file", 
        type=['csv'],
        help="Upload the zomato.csv file from Kaggle dataset"
    )
    
    # Add download link for dataset
    st.sidebar.markdown("""
    ### 📥 Get the Dataset
    Download from Kaggle:
    [Zomato Dataset](https://www.kaggle.com/datasets/rajeshrampure/zomato-dataset)
    
    After downloading, upload the CSV file above.
    """)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Select Analysis", [
        "📊 Overview",
        "📍 Location Analysis",
        "⭐ Rating Analysis",
        "💰 Cost Analysis",
        "🍜 Cuisine Analysis",
        "🤖 Machine Learning Models",
        "🎯 Clustering & Segmentation",
        "💡 Recommendations"
    ])
    
    # Load data
    with st.spinner("Loading data..."):
        df_raw = load_data(uploaded_file)
        
    if df_raw is None:
        st.warning("⚠️ No data loaded. Please upload the Zomato CSV file.")
        st.info("""
        ### How to get the dataset:
        1. Visit [Kaggle Zomato Dataset](https://www.kaggle.com/datasets/rajeshrampure/zomato-dataset)
        2. Download the CSV file
        3. Upload it using the sidebar uploader
        """)
        st.stop()
    
    df = preprocess_data(df_raw)
    
    # Display the selected page
    if page == "📊 Overview":
        show_overview(df)
    elif page == "📍 Location Analysis":
        show_location_analysis(df)
    elif page == "⭐ Rating Analysis":
        show_rating_analysis(df)
    elif page == "💰 Cost Analysis":
        show_cost_analysis(df)
    elif page == "🍜 Cuisine Analysis":
        show_cuisine_analysis(df)
    elif page == "🤖 Machine Learning Models":
        show_ml_models(df)
    elif page == "🎯 Clustering & Segmentation":
        show_clustering(df)
    elif page == "💡 Recommendations":
        show_recommendations(df)

def show_overview(df):
    """Display dataset overview"""
    st.markdown('<p class="sub-header">Dataset Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Restaurants", f"{len(df):,}")
    with col2:
        st.metric("Total Locations", df['location'].nunique() if 'location' in df.columns else 'N/A')
    with col3:
        st.metric("Avg Rating", f"{df['rate'].mean():.2f}" if 'rate' in df.columns else 'N/A')
    with col4:
        st.metric("Total Cuisines", df['cuisines'].nunique() if 'cuisines' in df.columns else 'N/A')
    
    st.markdown("---")
    
    # Display first few records
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Dataset information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        st.write(df.describe())
    
    with col2:
        st.subheader("Missing Values")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_df = missing_data[missing_data['Missing Count'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values!")
    
    # Q5 & Q6: Online ordering and table booking
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Q5: Online Ordering Facility")
        if 'online_order' in df.columns:
            online_counts = df['online_order'].value_counts()
            fig = px.pie(values=online_counts.values, names=online_counts.index, 
                        title="Online Ordering Distribution",
                        color_discrete_sequence=['#E23744', '#2E3B4E'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show counts
            for idx, val in online_counts.items():
                st.write(f"**{idx}**: {val:,} restaurants ({val/len(df)*100:.1f}%)")
    
    with col2:
        st.subheader("Q6: Table Booking Facility")
        if 'book_table' in df.columns:
            table_counts = df['book_table'].value_counts()
            fig = px.pie(values=table_counts.values, names=table_counts.index,
                        title="Table Booking Distribution",
                        color_discrete_sequence=['#E23744', '#2E3B4E'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show counts
            for idx, val in table_counts.items():
                st.write(f"**{idx}**: {val:,} restaurants ({val/len(df)*100:.1f}%)")

def show_location_analysis(df):
    """Display location-based analysis"""
    st.markdown('<p class="sub-header">Location Analysis</p>', unsafe_allow_html=True)
    
    if 'location' not in df.columns:
        st.error("Location column not found in dataset")
        return
    
    # Q1: Restaurants per location
    st.subheader("Q1: Number of Restaurants in Each Location")
    location_counts = df['location'].value_counts()
    
    fig = px.bar(x=location_counts.index[:20], y=location_counts.values[:20],
                 labels={'x': 'Location', 'y': 'Number of Restaurants'},
                 title="Top 20 Locations by Restaurant Count",
                 color=location_counts.values[:20],
                 color_continuous_scale='Reds')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Q2: Top 10 locations
    st.subheader("Q2: Top 10 Locations with Highest Number of Restaurants")
    top_10_locations = location_counts.head(10)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(pd.DataFrame({
            'Location': top_10_locations.index,
            'Restaurant Count': top_10_locations.values
        }).reset_index(drop=True), use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=top_10_locations.index,
            values=top_10_locations.values,
            hole=0.4
        )])
        fig.update_layout(title="Top 10 Locations Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Q9: Average rating by location
    if 'rate' in df.columns:
        st.subheader("Q9: Average Restaurant Rating for Each Location")
        avg_rating_by_location = df.groupby('location')['rate'].mean().sort_values(ascending=False).head(15)
        
        fig = px.bar(x=avg_rating_by_location.index, y=avg_rating_by_location.values,
                     labels={'x': 'Location', 'y': 'Average Rating'},
                     title="Top 15 Locations by Average Rating",
                     color=avg_rating_by_location.values,
                     color_continuous_scale='Viridis')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_rating_analysis(df):
    """Display rating-based analysis"""
    st.markdown('<p class="sub-header">Rating Analysis</p>', unsafe_allow_html=True)
    
    if 'rate' not in df.columns:
        st.error("Rating column not found in dataset")
        return
    
    # Q3: Distribution of ratings
    st.subheader("Q3: Distribution of Restaurant Ratings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df.dropna(subset=['rate']), x='rate', nbins=30,
                          title="Rating Distribution",
                          labels={'rate': 'Rating', 'count': 'Frequency'},
                          color_discrete_sequence=['#E23744'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df.dropna(subset=['rate']), y='rate', title="Rating Box Plot",
                    color_discrete_sequence=['#E23744'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Q4: Relationship between votes and ratings
    if 'votes' in df.columns:
        st.subheader("Q4: Relationship Between Votes and Restaurant Ratings")
        
        # Sample data for better visualization
        df_sample = df.dropna(subset=['rate', 'votes']).sample(min(1000, len(df.dropna(subset=['rate', 'votes']))))
        
        # Try with trendline, fall back to without if statsmodels not available
        try:
            fig = px.scatter(df_sample, x='votes', y='rate',
                            title="Votes vs Rating",
                            labels={'votes': 'Number of Votes', 'rate': 'Rating'},
                            trendline="ols",
                            opacity=0.6,
                            color='rate',
                            color_continuous_scale='Viridis')
        except Exception:
            # Fallback without trendline if statsmodels not available
            fig = px.scatter(df_sample, x='votes', y='rate',
                            title="Votes vs Rating",
                            labels={'votes': 'Number of Votes', 'rate': 'Rating'},
                            opacity=0.6,
                            color='rate',
                            color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Q10: Restaurants with highest votes
    if 'votes' in df.columns and 'name' in df.columns:
        st.subheader("Q10: Restaurants with Highest Number of Votes")
        cols_to_show = ['name', 'location', 'votes', 'rate']
        available_cols = [col for col in cols_to_show if col in df.columns]
        top_voted = df.nlargest(10, 'votes')[available_cols]
        st.dataframe(top_voted.reset_index(drop=True), use_container_width=True)
    
    # Q18: Restaurants with rating >= 4
    st.subheader("Q18: High-Rated Restaurants (Rating ≥ 4)")
    high_rated = df[df['rate'] >= 4]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of High-Rated Restaurants", f"{len(high_rated):,}")
    with col2:
        st.metric("Percentage", f"{len(high_rated)/len(df)*100:.2f}%")
    
    # Q13 & Q14: Average rating by online ordering and table booking
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Q13: Avg Rating - Online Ordering")
        if 'online_order' in df.columns and 'rate' in df.columns:
            avg_rating_online = df.groupby('online_order')['rate'].mean()
            fig = px.bar(x=['No', 'Yes'], y=[avg_rating_online.get('No', 0), avg_rating_online.get('Yes', 0)],
                        labels={'x': 'Online Ordering', 'y': 'Average Rating'},
                        color=['No', 'Yes'],
                        color_discrete_sequence=['#2E3B4E', '#E23744'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Q14: Avg Rating - Table Booking")
        if 'book_table' in df.columns and 'rate' in df.columns:
            avg_rating_table = df.groupby('book_table')['rate'].mean()
            fig = px.bar(x=['No', 'Yes'], y=[avg_rating_table.get('No', 0), avg_rating_table.get('Yes', 0)],
                        labels={'x': 'Table Booking', 'y': 'Average Rating'},
                        color=['No', 'Yes'],
                        color_discrete_sequence=['#2E3B4E', '#E23744'])
            st.plotly_chart(fig, use_container_width=True)
    
    # Q23: Correlation between votes and ratings
    if 'votes' in df.columns:
        st.subheader("Q23: Correlation Between Votes and Ratings")
        correlation = df[['rate', 'votes']].corr().iloc[0, 1]
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        if correlation > 0.5:
            st.success("Strong positive correlation - Higher votes tend to correlate with higher ratings")
        elif correlation > 0.3:
            st.info("Moderate positive correlation")
        else:
            st.warning("Weak correlation")

def show_cost_analysis(df):
    """Display cost-based analysis"""
    st.markdown('<p class="sub-header">Cost Analysis</p>', unsafe_allow_html=True)
    
    cost_col = 'approx_cost(for two people)'
    
    if cost_col not in df.columns:
        st.error("Cost column not found in dataset")
        return
    
    # Q7: Distribution of cost
    st.subheader("Q7: Distribution of Approximate Cost for Two People")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df.dropna(subset=[cost_col]), x=cost_col, nbins=50,
                          title="Cost Distribution",
                          labels={cost_col: 'Cost for Two (₹)', 'count': 'Frequency'},
                          color_discrete_sequence=['#E23744'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df.dropna(subset=[cost_col]), y=cost_col, title="Cost Box Plot",
                    color_discrete_sequence=['#E23744'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Cost", f"₹{df[cost_col].mean():.0f}")
    with col2:
        st.metric("Median Cost", f"₹{df[cost_col].median():.0f}")
    with col3:
        st.metric("Min Cost", f"₹{df[cost_col].min():.0f}")
    with col4:
        st.metric("Max Cost", f"₹{df[cost_col].max():.0f}")
    
    # Q8: Relationship between cost and rating
    if 'rate' in df.columns:
        st.subheader("Q8: Relationship Between Cost and Rating")
        
        df_sample = df.dropna(subset=['rate', cost_col]).sample(min(1000, len(df.dropna(subset=['rate', cost_col]))))
        
        # Try with trendline, fall back to without if statsmodels not available
        try:
            fig = px.scatter(df_sample, x=cost_col, y='rate',
                            title="Cost vs Rating",
                            labels={cost_col: 'Cost for Two (₹)', 'rate': 'Rating'},
                            trendline="ols",
                            opacity=0.6,
                            color='rate',
                            color_continuous_scale='Viridis')
        except Exception:
            fig = px.scatter(df_sample, x=cost_col, y='rate',
                            title="Cost vs Rating",
                            labels={cost_col: 'Cost for Two (₹)', 'rate': 'Rating'},
                            opacity=0.6,
                            color='rate',
                            color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Q17: Correlation matrix
    if 'rate' in df.columns and 'votes' in df.columns:
        st.subheader("Q17: Correlation Between Rating, Votes, and Cost")
        
        corr_data = df[['rate', 'votes', cost_col]].dropna()
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(corr_matrix, 
                        text_auto='.2f',
                        title="Correlation Heatmap",
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    
    # Q19: Average cost by restaurant type
    if 'rest_type' in df.columns:
        st.subheader("Q19: Average Cost by Restaurant Type")
        avg_cost_by_type = df.groupby('rest_type')[cost_col].mean().sort_values(ascending=False).head(10)
        
        fig = px.bar(x=avg_cost_by_type.index, y=avg_cost_by_type.values,
                    labels={'x': 'Restaurant Type', 'y': 'Average Cost (₹)'},
                    title="Top 10 Restaurant Types by Average Cost",
                    color=avg_cost_by_type.values,
                    color_continuous_scale='Reds')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Q29: Most expensive restaurants
    if 'name' in df.columns:
        st.subheader("Q29: Top 10 Most Expensive Restaurants")
        cols_to_show = ['name', 'location', cost_col, 'rate']
        available_cols = [col for col in cols_to_show if col in df.columns]
        expensive = df.nlargest(10, cost_col)[available_cols]
        st.dataframe(expensive.reset_index(drop=True), use_container_width=True)
    
    # Q30: Cheapest high-rated restaurants
    if 'name' in df.columns and 'rate' in df.columns:
        st.subheader("Q30: Cheapest Restaurants with High Ratings (≥4)")
        cheap_high_rated = df[df['rate'] >= 4.0].nsmallest(10, cost_col)
        cols_to_show = ['name', 'location', cost_col, 'rate']
        available_cols = [col for col in cols_to_show if col in cheap_high_rated.columns]
        st.dataframe(cheap_high_rated[available_cols].reset_index(drop=True), use_container_width=True)

def show_cuisine_analysis(df):
    """Display cuisine-based analysis"""
    st.markdown('<p class="sub-header">Cuisine Analysis</p>', unsafe_allow_html=True)
    
    if 'cuisines' not in df.columns:
        st.warning("Cuisine information not available in the dataset")
        return
    
    # Q11 & Q12: Most common cuisines
    st.subheader("Q11 & Q12: Most Common Cuisines")
    
    # Split cuisines (as they might be comma-separated)
    cuisines_list = []
    for cuisine in df['cuisines'].dropna():
        cuisines_list.extend([c.strip() for c in str(cuisine).split(',')])
    
    cuisine_counts = pd.Series(cuisines_list).value_counts().head(20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(x=cuisine_counts.index, y=cuisine_counts.values,
                    labels={'x': 'Cuisine', 'y': 'Count'},
                    title="Top 20 Most Popular Cuisines",
                    color=cuisine_counts.values,
                    color_continuous_scale='Reds')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=cuisine_counts.head(10).index,
            values=cuisine_counts.head(10).values,
            hole=0.3
        )])
        fig.update_layout(title="Top 10 Cuisines Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Q48: Cuisine influence on ratings
    if 'rate' in df.columns:
        st.subheader("Q48: How Cuisine Type Influences Restaurant Ratings")
        
        # Get average rating for top cuisines
        cuisine_ratings = {}
        for cuisine in cuisine_counts.head(15).index:
            mask = df['cuisines'].str.contains(cuisine, na=False, case=False)
            cuisine_ratings[cuisine] = df[mask]['rate'].mean()
        
        cuisine_ratings_df = pd.DataFrame(list(cuisine_ratings.items()), 
                                         columns=['Cuisine', 'Avg Rating']).sort_values('Avg Rating', ascending=False)
        
        fig = px.bar(cuisine_ratings_df, x='Cuisine', y='Avg Rating',
                    title="Average Rating by Cuisine Type",
                    color='Avg Rating',
                    color_continuous_scale='Viridis')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_ml_models(df):
    """Display machine learning models"""
    st.markdown('<p class="sub-header">Machine Learning Models</p>', unsafe_allow_html=True)
    
    # Check required columns
    if 'rate' not in df.columns:
        st.error("Rating column required for ML models")
        return
    
    # Prepare data
    df_ml = df.dropna(subset=['rate'])
    if 'approx_cost(for two people)' in df.columns:
        df_ml = df_ml.dropna(subset=['approx_cost(for two people)'])
    if 'votes' in df.columns:
        df_ml = df_ml.dropna(subset=['votes'])
    
    if len(df_ml) < 100:
        st.error("Insufficient data for machine learning models")
        return
    
    model_type = st.selectbox("Select Model Type", [
        "Q31: Rating Prediction (Regression)",
        "Q32: High/Low Rating Classification",
        "Q33: Feature Importance Analysis",
        "Q34: Online Ordering Prediction",
        "Q35: Table Booking Prediction"
    ])
    
    if model_type == "Q31: Rating Prediction (Regression)":
        st.subheader("Predicting Restaurant Ratings")
        
        # Prepare features
        features = []
        if 'votes' in df_ml.columns:
            features.append('votes')
        if 'approx_cost(for two people)' in df_ml.columns:
            features.append('approx_cost(for two people)')
        if 'online_order_binary' in df_ml.columns:
            features.append('online_order_binary')
        if 'book_table_binary' in df_ml.columns:
            features.append('book_table_binary')
        
        if len(features) < 2:
            st.warning("Insufficient features for regression model")
            return
        
        X = df_ml[features]
        y = df_ml['rate']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        with st.spinner("Training Random Forest Regressor..."):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("RMSE", f"{np.sqrt(mse):.4f}")
            
            # Actual vs Predicted
            fig = px.scatter(x=y_test, y=y_pred,
                           labels={'x': 'Actual Rating', 'y': 'Predicted Rating'},
                           title="Actual vs Predicted Ratings",
                           opacity=0.6)
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                    y=[y_test.min(), y_test.max()],
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(color='red', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
    
    elif model_type == "Q32: High/Low Rating Classification":
        st.subheader("Classifying Restaurants as High-Rated or Low-Rated")
        
        # Create binary target (rating >= 4 is high)
        df_ml['high_rated'] = (df_ml['rate'] >= 4).astype(int)
        
        features = []
        if 'votes' in df_ml.columns:
            features.append('votes')
        if 'approx_cost(for two people)' in df_ml.columns:
            features.append('approx_cost(for two people)')
        if 'online_order_binary' in df_ml.columns:
            features.append('online_order_binary')
        if 'book_table_binary' in df_ml.columns:
            features.append('book_table_binary')
        
        if len(features) < 2:
            st.warning("Insufficient features for classification model")
            return
        
        X = df_ml[features]
        y = df_ml['high_rated']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with st.spinner("Training Random Forest Classifier..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            # Classification report
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, target_names=['Low Rated', 'High Rated']))
    
    elif model_type == "Q33: Feature Importance Analysis":
        st.subheader("Feature Importance in Predicting Ratings")
        
        features = []
        if 'votes' in df_ml.columns:
            features.append('votes')
        if 'approx_cost(for two people)' in df_ml.columns:
            features.append('approx_cost(for two people)')
        if 'online_order_binary' in df_ml.columns:
            features.append('online_order_binary')
        if 'book_table_binary' in df_ml.columns:
            features.append('book_table_binary')
        
        if len(features) < 2:
            st.warning("Insufficient features for feature importance analysis")
            return
        
        X = df_ml[features]
        y = df_ml['rate']
        
        with st.spinner("Analyzing feature importance..."):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h',
                        title="Feature Importance",
                        color='Importance',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    elif model_type in ["Q34: Online Ordering Prediction", "Q35: Table Booking Prediction"]:
        target_col = 'online_order_binary' if 'Q34' in model_type else 'book_table_binary'
        target_name = 'Online Ordering' if 'Q34' in model_type else 'Table Booking'
        
        if target_col not in df_ml.columns:
            st.warning(f"{target_name} information not available")
            return
        
        st.subheader(f"Predicting {target_name}")
        
        features = []
        if 'rate' in df_ml.columns:
            features.append('rate')
        if 'votes' in df_ml.columns:
            features.append('votes')
        if 'approx_cost(for two people)' in df_ml.columns:
            features.append('approx_cost(for two people)')
        
        if len(features) < 2:
            st.warning("Insufficient features for prediction")
            return
        
        df_temp = df_ml.dropna(subset=[target_col])
        X = df_temp[features]
        y = df_temp[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with st.spinner("Training classifier..."):
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

def show_clustering(df):
    """Display clustering analysis"""
    st.markdown('<p class="sub-header">Clustering & Market Segmentation</p>', unsafe_allow_html=True)
    
    # Check required columns
    required_cols = ['rate', 'approx_cost(for two people)', 'votes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return
    
    # Prepare data
    df_cluster = df.dropna(subset=required_cols)
    
    if len(df_cluster) < 100:
        st.error("Insufficient data for clustering")
        return
    
    # Q36: Cluster restaurants
    st.subheader("Q36-Q45: Restaurant Clustering Based on Cost and Rating")
    
    n_clusters = st.slider("Number of Clusters", 2, 10, 4)
    
    # Select features for clustering
    X = df_cluster[required_cols]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    with st.spinner("Performing K-Means clustering..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster = df_cluster.copy()
        df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Q37: Visualize clusters
        st.subheader("Cluster Visualization")
        
        # 3D visualization
        sample_size = min(1000, len(df_cluster))
        df_sample = df_cluster.sample(sample_size)
        
        fig = px.scatter_3d(df_sample,
                           x='approx_cost(for two people)', 
                           y='rate', 
                           z='votes',
                           color='cluster',
                           title="3D Cluster Visualization",
                           labels={'approx_cost(for two people)': 'Cost',
                                  'rate': 'Rating',
                                  'votes': 'Votes'})
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D visualization
        fig = px.scatter(df_cluster, 
                        x='approx_cost(for two people)', 
                        y='rate',
                        color='cluster',
                        title="Clusters: Cost vs Rating",
                        size='votes',
                        opacity=0.6,
                        color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Q38 & Q39: Cluster statistics
        st.subheader("Cluster Statistics")
        
        cluster_stats = df_cluster.groupby('cluster').agg({
            'rate': 'mean',
            'approx_cost(for two people)': 'mean',
            'votes': 'mean',
            'name': 'count' if 'name' in df_cluster.columns else 'size'
        }).round(2)
        
        cluster_stats.columns = ['Avg Rating', 'Avg Cost', 'Avg Votes', 'Count']
        cluster_stats = cluster_stats.reset_index()
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Q41: Highest-rated cluster
        st.subheader("Q41: Highest-Rated Cluster")
        best_cluster = cluster_stats.loc[cluster_stats['Avg Rating'].idxmax(), 'cluster']
        best_rating = cluster_stats.loc[cluster_stats['Avg Rating'].idxmax(), 'Avg Rating']
        st.success(f"Cluster {best_cluster} has the highest average rating: {best_rating:.2f}")
        
        # Q42: Budget-friendly cluster
        st.subheader("Q42: Budget-Friendly Cluster")
        budget_cluster = cluster_stats.loc[cluster_stats['Avg Cost'].idxmin(), 'cluster']
        budget_cost = cluster_stats.loc[cluster_stats['Avg Cost'].idxmin(), 'Avg Cost']
        st.success(f"Cluster {budget_cluster} represents budget-friendly restaurants with avg cost: ₹{budget_cost:.0f}")
        
        # Q43: Cluster distribution
        st.subheader("Q43: Restaurant Distribution Across Clusters")
        
        fig = px.pie(cluster_stats, 
                    values='Count', 
                    names='cluster',
                    title="Cluster Distribution",
                    hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Q44: Votes by cluster
        st.subheader("Q44: Votes Distribution by Cluster")
        
        fig = px.box(df_cluster, x='cluster', y='votes',
                    title="Votes Distribution by Cluster",
                    color='cluster',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Q45: Market segmentation insights
        st.subheader("Q45: Market Segmentation Insights")
        
        median_cost = df['approx_cost(for two people)'].median()
        
        for idx, row in cluster_stats.iterrows():
            cluster_id = row['cluster']
            
            with st.expander(f"Cluster {cluster_id} Profile"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Rating", f"{row['Avg Rating']:.2f}")
                with col2:
                    st.metric("Avg Cost", f"₹{row['Avg Cost']:.0f}")
                with col3:
                    st.metric("Restaurants", f"{int(row['Count']):,}")
                
                # Characterize cluster
                if row['Avg Rating'] >= 4 and row['Avg Cost'] >= median_cost:
                    st.info("**Premium Segment**: High-rated, expensive restaurants")
                elif row['Avg Rating'] >= 4 and row['Avg Cost'] < median_cost:
                    st.success("**Value Segment**: High-rated, affordable restaurants")
                elif row['Avg Rating'] < 4 and row['Avg Cost'] >= median_cost:
                    st.warning("**Overpriced Segment**: Low-rated, expensive restaurants")
                else:
                    st.error("**Budget Segment**: Low-rated, cheap restaurants")

def show_recommendations(df):
    """Display recommendation system"""
    st.markdown('<p class="sub-header">Restaurant Recommendation System</p>', unsafe_allow_html=True)
    
    # Q40, Q46, Q47: Simple recommendation system
    st.subheader("Q40 & Q46: Find Your Perfect Restaurant")
    
    # Check required columns
    if 'approx_cost(for two people)' not in df.columns or 'rate' not in df.columns:
        st.error("Required columns (cost, rating) not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_cost = st.slider("Maximum Budget (₹)", 
                            int(df['approx_cost(for two people)'].min()),
                            int(df['approx_cost(for two people)'].max()),
                            int(df['approx_cost(for two people)'].median()))
        
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    
    with col2:
        if 'location' in df.columns:
            location = st.selectbox("Preferred Location (Optional)", 
                                   ['Any'] + sorted(df['location'].unique().tolist()))
        else:
            location = 'Any'
        
        if 'cuisines' in df.columns:
            # Get unique cuisines
            all_cuisines = set()
            for cuisine in df['cuisines'].dropna():
                all_cuisines.update([c.strip() for c in str(cuisine).split(',')])
            
            cuisine_pref = st.selectbox("Preferred Cuisine (Optional)",
                                       ['Any'] + sorted(list(all_cuisines))[:50])
        else:
            cuisine_pref = 'Any'
    
    # Filter recommendations
    recommendations = df[
        (df['approx_cost(for two people)'] <= max_cost) &
        (df['rate'] >= min_rating)
    ].copy()
    
    if location != 'Any' and 'location' in df.columns:
        recommendations = recommendations[recommendations['location'] == location]
    
    if cuisine_pref != 'Any' and 'cuisines' in df.columns:
        recommendations = recommendations[recommendations['cuisines'].str.contains(cuisine_pref, na=False, case=False)]
    
    # Sort by rating and votes
    if 'votes' in recommendations.columns:
        recommendations = recommendations.sort_values(['rate', 'votes'], ascending=False)
    else:
        recommendations = recommendations.sort_values('rate', ascending=False)
    
    st.subheader(f"Top {min(10, len(recommendations))} Recommendations")
    
    if len(recommendations) > 0:
        # Select columns to display
        display_cols = ['name', 'location', 'rate', 'approx_cost(for two people)']
        if 'votes' in recommendations.columns:
            display_cols.insert(3, 'votes')
        if 'cuisines' in recommendations.columns:
            display_cols.append('cuisines')
        
        available_cols = [col for col in display_cols if col in recommendations.columns]
        top_recommendations = recommendations.head(10)[available_cols]
        st.dataframe(top_recommendations.reset_index(drop=True), use_container_width=True)
        
        # Show on scatter plot
        st.subheader("Recommendation Distribution")
        
        sample_size = min(50, len(recommendations))
        plot_data = recommendations.head(sample_size)
        
        if 'votes' in plot_data.columns:
            fig = px.scatter(plot_data,
                            x='approx_cost(for two people)',
                            y='rate',
                            size='votes',
                            hover_data=['name', 'location'] if 'name' in plot_data.columns and 'location' in plot_data.columns else None,
                            title="Cost vs Rating of Recommended Restaurants",
                            color='rate',
                            color_continuous_scale='Viridis')
        else:
            fig = px.scatter(plot_data,
                            x='approx_cost(for two people)',
                            y='rate',
                            hover_data=['name', 'location'] if 'name' in plot_data.columns and 'location' in plot_data.columns else None,
                            title="Cost vs Rating of Recommended Restaurants",
                            color='rate',
                            color_continuous_scale='Viridis')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No restaurants match your criteria. Try adjusting your filters.")
    
    # Q49 & Q50: Insights
    st.subheader("Q49 & Q50: Key Insights from Restaurant Data")
    
    with st.expander("📊 Key Insights"):
        st.markdown(f"""
        ### Data Overview
        - **Total Restaurants Analyzed**: {len(df):,}
        - **Average Rating**: {df['rate'].mean():.2f}
        - **Average Cost for Two**: ₹{df['approx_cost(for two people)'].mean():.0f}
        """)
        
        if 'votes' in df.columns:
            st.markdown(f"- **Average Votes**: {df['votes'].mean():.0f}")
        
        if 'location' in df.columns:
            st.markdown(f"""
            ### Location Insights
            - **Most Popular Location**: {df['location'].value_counts().index[0]} ({df['location'].value_counts().values[0]} restaurants)
            - **Locations Covered**: {df['location'].nunique()}
            """)
        
        st.markdown(f"""
        ### Rating Insights
        - **High-Rated Restaurants (≥4)**: {len(df[df['rate'] >= 4]):,} ({len(df[df['rate'] >= 4])/len(df)*100:.1f}%)
        """)
        
        if 'votes' in df.columns:
            corr = df[['rate', 'votes']].corr().iloc[0,1]
            st.markdown(f"- **Correlation (Votes-Rating)**: {corr:.3f}")
        
        st.markdown(f"""
        ### Cost Insights
        - **Budget Range**: ₹{df['approx_cost(for two people)'].min():.0f} - ₹{df['approx_cost(for two people)'].max():.0f}
        - **Median Price**: ₹{df['approx_cost(for two people)'].median():.0f}
        """)
        
        if 'online_order' in df.columns:
            online_yes = (df['online_order'] == 'Yes').sum()
            online_pct = online_yes / len(df) * 100
            st.markdown(f"- **Online Ordering Available**: {online_pct:.1f}% of restaurants")
        
        if 'book_table' in df.columns:
            table_yes = (df['book_table'] == 'Yes').sum()
            table_pct = table_yes / len(df) * 100
            st.markdown(f"- **Table Booking Available**: {table_pct:.1f}% of restaurants")
        
        st.markdown("""
        ### Business Recommendations
        1. **Focus on Quality**: Higher ratings correlate with more customer engagement
        2. **Digital Presence**: Online ordering attracts more customers
        3. **Price Strategy**: Balance affordability with quality
        4. **Location Matters**: High-density areas offer more opportunities
        """)

if __name__ == "__main__":
    main()
