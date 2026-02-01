import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="OptiPrice Prognosticator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Load data function
@st.cache_data
def load_data(uploaded_file=None):
    """Load dataset from file or uploaded source"""
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            # Try to load from default location
            data = pd.read_csv("dynamic_pricing.csv")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Feature engineering function
def create_engineered_features(data):
    """Create engineered features for Model 4"""
    df = data.copy()
    df['Rider_to_Driver_Ratio'] = df['Number_of_Riders'] / df['Number_of_Drivers']
    df['Cost_per_Minute'] = df['Historical_Cost_of_Ride'] / df['Expected_Ride_Duration']
    df['Riders_x_Duration'] = df['Number_of_Riders'] * df['Expected_Ride_Duration']
    df['Ratings_Squared'] = df['Average_Ratings'] ** 2
    df['Ratings_Binned'] = pd.cut(df['Average_Ratings'], bins=[0, 2, 4, 5], labels=['Low', 'Medium', 'High'])
    return df

# Train all models
def train_all_models(data):
    """Train all 4 models exactly as in your original files"""
    
    results = {}
    target = 'Historical_Cost_of_Ride'
    
    # ========== MODEL 1 ==========
    st.write("Training Model 1...")
    numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                         'Average_Ratings', 'Expected_Ride_Duration']
    
    X = data[numerical_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_1 = RandomForestRegressor(random_state=42)
    model_1.fit(X_train, y_train)
    y_pred_1 = model_1.predict(X_test)
    
    mse_1 = mean_squared_error(y_test, y_pred_1)
    r2_1 = r2_score(y_test, y_pred_1)
    
    results['Model 1'] = {
        'model': model_1,
        'scaler': None,
        'encoder': None,
        'y_test': y_test,
        'y_pred': y_pred_1,
        'mse': mse_1,
        'rmse': np.sqrt(mse_1),
        'mae': mean_absolute_error(y_test, y_pred_1),
        'r2': r2_1,
        'description': 'All numerical features',
        'numerical_features': numerical_features,
        'categorical_features': []
    }
    
    # ========== MODEL 2 ==========
    st.write("Training Model 2...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_2 = RandomForestRegressor(random_state=42)
    model_2.fit(X_train_scaled, y_train)
    y_pred_2 = model_2.predict(X_test_scaled)
    
    mse_2 = mean_squared_error(y_test, y_pred_2)
    r2_2 = r2_score(y_test, y_pred_2)
    
    results['Model 2'] = {
        'model': model_2,
        'scaler': scaler,
        'encoder': None,
        'y_test': y_test,
        'y_pred': y_pred_2,
        'mse': mse_2,
        'rmse': np.sqrt(mse_2),
        'mae': mean_absolute_error(y_test, y_pred_2),
        'r2': r2_2,
        'description': 'Numerical features + StandardScaler',
        'numerical_features': numerical_features,
        'categorical_features': []
    }
    
    # ========== MODEL 3 ==========
    st.write("Training Model 3...")
    categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
    
    X = data[numerical_features + categorical_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    encoder = OneHotEncoder(drop='first')
    
    X_train_num = scaler.fit_transform(X_train[numerical_features])
    X_test_num = scaler.transform(X_test[numerical_features])
    X_train_cat = encoder.fit_transform(X_train[categorical_features]).toarray()
    X_test_cat = encoder.transform(X_test[categorical_features]).toarray()
    
    X_train_full = np.hstack((X_train_num, X_train_cat))
    X_test_full = np.hstack((X_test_num, X_test_cat))
    
    model_3 = RandomForestRegressor(random_state=42)
    model_3.fit(X_train_full, y_train)
    y_pred_3 = model_3.predict(X_test_full)
    
    mse_3 = mean_squared_error(y_test, y_pred_3)
    r2_3 = r2_score(y_test, y_pred_3)
    
    results['Model 3'] = {
        'model': model_3,
        'scaler': scaler,
        'encoder': encoder,
        'y_test': y_test,
        'y_pred': y_pred_3,
        'mse': mse_3,
        'rmse': np.sqrt(mse_3),
        'mae': mean_absolute_error(y_test, y_pred_3),
        'r2': r2_3,
        'description': 'Numerical + StandardScaler + Categorical OneHotEncoding',
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    
    # ========== MODEL 4 ==========
    st.write("Training Model 4...")
    data_engineered = create_engineered_features(data)
    
    numerical_features_4 = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                           'Average_Ratings', 'Expected_Ride_Duration',
                           'Rider_to_Driver_Ratio', 'Cost_per_Minute', 
                           'Riders_x_Duration', 'Ratings_Squared']
    categorical_features_4 = ['Location_Category', 'Customer_Loyalty_Status', 
                             'Time_of_Booking', 'Vehicle_Type', 'Ratings_Binned']
    
    X = data_engineered[numerical_features_4 + categorical_features_4]
    y = data_engineered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    encoder = OneHotEncoder(drop='first')
    
    X_train_num = scaler.fit_transform(X_train[numerical_features_4])
    X_test_num = scaler.transform(X_test[numerical_features_4])
    X_train_cat = encoder.fit_transform(X_train[categorical_features_4]).toarray()
    X_test_cat = encoder.transform(X_test[categorical_features_4]).toarray()
    
    X_train_full = np.hstack((X_train_num, X_train_cat))
    X_test_full = np.hstack((X_test_num, X_test_cat))
    
    model_4 = RandomForestRegressor(random_state=42)
    model_4.fit(X_train_full, y_train)
    y_pred_4 = model_4.predict(X_test_full)
    
    mse_4 = mean_squared_error(y_test, y_pred_4)
    r2_4 = r2_score(y_test, y_pred_4)
    
    results['Model 4'] = {
        'model': model_4,
        'scaler': scaler,
        'encoder': encoder,
        'y_test': y_test,
        'y_pred': y_pred_4,
        'mse': mse_4,
        'rmse': np.sqrt(mse_4),
        'mae': mean_absolute_error(y_test, y_pred_4),
        'r2': r2_4,
        'description': 'Numerical + StandardScaler + Categorical OneHotEncoding + Feature Engineering',
        'numerical_features': numerical_features_4,
        'categorical_features': categorical_features_4
    }
    
    return results

# Make prediction function
def make_prediction(model_name, input_data):
    """Make prediction using the selected model"""
    model_info = st.session_state.results[model_name]
    
    # Prepare input DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Handle Model 4 feature engineering
    if model_name == 'Model 4':
        input_df['Rider_to_Driver_Ratio'] = input_df['Number_of_Riders'] / input_df['Number_of_Drivers']
        input_df['Cost_per_Minute'] = 10.0  # Placeholder
        input_df['Riders_x_Duration'] = input_df['Number_of_Riders'] * input_df['Expected_Ride_Duration']
        input_df['Ratings_Squared'] = input_df['Average_Ratings'] ** 2
        
        rating = input_df['Average_Ratings'].values[0]
        if rating <= 2:
            input_df['Ratings_Binned'] = 'Low'
        elif rating <= 4:
            input_df['Ratings_Binned'] = 'Medium'
        else:
            input_df['Ratings_Binned'] = 'High'
    
    # Extract features
    num_features = model_info['numerical_features']
    cat_features = model_info['categorical_features']
    
    # Scale numerical features
    if model_info['scaler'] is not None:
        X_num = model_info['scaler'].transform(input_df[num_features])
    else:
        X_num = input_df[num_features].values
    
    # Encode categorical features
    if model_info['encoder'] is not None and len(cat_features) > 0:
        X_cat = model_info['encoder'].transform(input_df[cat_features]).toarray()
        X_full = np.hstack((X_num, X_cat))
    else:
        X_full = X_num
    
    # Make prediction
    prediction = model_info['model'].predict(X_full)
    return prediction[0]

# ================ MAIN APP ================

# Header
st.markdown('<p class="main-header"> OptiPrice Prognosticator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dynamic Ride Pricing Prediction System</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title(" Navigation")
page = st.sidebar.radio("", [" Home", " Predictions", " Model Comparison", " Data Explorer"])

st.sidebar.markdown("---")
st.sidebar.markdown("###  Quick Info")
if st.session_state.data is not None:
    st.sidebar.success(f" Data Loaded: {len(st.session_state.data)} rows")
else:
    st.sidebar.warning(" No data loaded")

if st.session_state.models_trained:
    st.sidebar.success(" Models Trained")
else:
    st.sidebar.warning(" Models not trained")

# ========== HOME PAGE ==========
if page == " Home":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##  Welcome to OptiPrice Prognosticator")
        st.write("""
        This is an advanced machine learning system for predicting dynamic ride prices based on multiple factors.
        
        ###  How It Works:
        1. **Upload your data** (or use the default dataset)
        2. **Train all 4 models** with progressive improvements
        3. **Make predictions** with real-time pricing
        4. **Compare models** to see which performs best
        
        ###  The 4 Models:
        - **Model 1**: Basic numerical features only
        - **Model 2**: Adds StandardScaler for normalization
        - **Model 3**: Includes categorical features (Location, Loyalty, Time, Vehicle)
        - **Model 4**: Advanced feature engineering (ratios, interactions, binning)
        """)
    
    with col2:
        st.markdown("##  Load Data")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if data is not None:
                st.session_state.data = data
                st.success(f" Loaded {len(data)} rows")
        elif st.session_state.data is None:
            # Try default file
            data = load_data()
            if data is not None:
                st.session_state.data = data
                st.info(" Using default dataset")
    
    st.markdown("---")
    
    if st.session_state.data is not None:
        st.markdown("## Dataset Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rides", len(st.session_state.data))
        with col2:
            st.metric("Avg Price", f"${st.session_state.data['Historical_Cost_of_Ride'].mean():.2f}")
        with col3:
            st.metric("Min Price", f"${st.session_state.data['Historical_Cost_of_Ride'].min():.2f}")
        with col4:
            st.metric("Max Price", f"${st.session_state.data['Historical_Cost_of_Ride'].max():.2f}")
        
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        st.markdown("---")
        st.markdown("##  Train Models")
        
        if st.button(" Train All 4 Models", type="primary", use_container_width=True):
            with st.spinner("Training models... This may take a moment..."):
                progress_bar = st.progress(0)
                results = train_all_models(st.session_state.data)
                progress_bar.progress(100)
                
                st.session_state.results = results
                st.session_state.models_trained = True
                
                st.success(" All models trained successfully!")
                st.balloons()
                
                # Show quick results
                st.markdown("###  Quick Results")
                results_df = pd.DataFrame({
                    'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4'],
                    'R² Score': [results[f'Model {i}']['r2'] for i in range(1, 5)],
                    'RMSE': [results[f'Model {i}']['rmse'] for i in range(1, 5)]
                })
                st.dataframe(results_df, use_container_width=True)

# ========== PREDICTIONS PAGE ==========
elif page == " Predictions":
    
    if not st.session_state.models_trained:
        st.warning(" Please train the models first from the Home page!")
        st.stop()
    
    st.markdown("##  Make a Price Prediction")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["Model 1", "Model 2", "Model 3", "Model 4"],
        index=3,
        help="Model 4 is the most advanced with feature engineering"
    )
    
    st.markdown("---")
    st.markdown("###  Enter Ride Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(" Demand & Supply")
        num_riders = st.number_input("Number of Riders", min_value=1, max_value=100, value=10)
        num_drivers = st.number_input("Number of Drivers", min_value=1, max_value=100, value=15)
        num_past_rides = st.number_input("Number of Past Rides", min_value=0, max_value=1000, value=50)
    
    with col2:
        st.markdown("Quality & Duration")
        avg_rating = st.slider("Average Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        ride_duration = st.number_input("Expected Ride Duration (mins)", min_value=1, max_value=120, value=15)
    
    with col3:
        if model_choice in ["Model 3", "Model 4"]:
            st.markdown("**📍 Additional Details**")
            location = st.selectbox("Location Category", 
                                   st.session_state.data['Location_Category'].unique())
            loyalty = st.selectbox("Customer Loyalty", 
                                  st.session_state.data['Customer_Loyalty_Status'].unique())
            time_booking = st.selectbox("Time of Booking", 
                                       st.session_state.data['Time_of_Booking'].unique())
            vehicle_type = st.selectbox("Vehicle Type", 
                                       st.session_state.data['Vehicle_Type'].unique())
    
    st.markdown("---")
    
    if st.button(" Predict Price Now", type="primary", use_container_width=True):
        
        # Prepare input
        input_data = {
            'Number_of_Riders': num_riders,
            'Number_of_Drivers': num_drivers,
            'Number_of_Past_Rides': num_past_rides,
            'Average_Ratings': avg_rating,
            'Expected_Ride_Duration': ride_duration
        }
        
        # Add categorical features if needed
        if model_choice in ["Model 3", "Model 4"]:
            input_data['Location_Category'] = location
            input_data['Customer_Loyalty_Status'] = loyalty
            input_data['Time_of_Booking'] = time_booking
            input_data['Vehicle_Type'] = vehicle_type
        
        # Make prediction
        prediction = make_prediction(model_choice, input_data)
        
        # Display results
        st.markdown("---")
        st.markdown("##  Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("###  Predicted Price")
            st.markdown(f"<h1 style='text-align: center; color: green;'>${prediction:.2f}</h1>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Model Used", model_choice)
        
        with col3:
            r2 = st.session_state.results[model_choice]['r2']
            st.metric("Model Accuracy (R²)", f"{r2:.2%}")
        
        with col4:
            rmse = st.session_state.results[model_choice]['rmse']
            st.metric("Model Error (RMSE)", f"${rmse:.2f}")
        
        # Additional insights
        st.markdown("---")
        st.markdown("###  Prediction Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Input Summary:**")
            st.write(f"- Rider to Driver Ratio: {num_riders/num_drivers:.2f}")
            st.write(f"- Price per Minute: ${prediction/ride_duration:.2f}")
            st.write(f"- Customer Rating: {avg_rating} ")
        
        with col2:
            st.write("**Model Performance:**")
            st.write(f"- {st.session_state.results[model_choice]['description']}")
            st.write(f"- Mean Absolute Error: ${st.session_state.results[model_choice]['mae']:.2f}")

# ========== MODEL COMPARISON PAGE ==========
elif page == " Model Comparison":
    
    if not st.session_state.models_trained:
        st.warning(" Please train the models first from the Home page!")
        st.stop()
    
    st.markdown("##  Model Performance Comparison")
    
    results = st.session_state.results
    
    # Metrics table
    metrics_df = pd.DataFrame({
        'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4'],
        'Description': [results[f'Model {i}']['description'] for i in range(1, 5)],
        'R² Score': [results[f'Model {i}']['r2'] for i in range(1, 5)],
        'RMSE': [results[f'Model {i}']['rmse'] for i in range(1, 5)],
        'MAE': [results[f'Model {i}']['mae'] for i in range(1, 5)],
        'MSE': [results[f'Model {i}']['mse'] for i in range(1, 5)]
    })
    
    st.dataframe(
        metrics_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                       .highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='lightgreen'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  R² Score Comparison")
        fig1 = px.bar(
            metrics_df, x='Model', y='R² Score',
            color='R² Score',
            color_continuous_scale='Blues',
            text='R² Score'
        )
        fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("###  RMSE Comparison")
        fig2 = px.bar(
            metrics_df, x='Model', y='RMSE',
            color='RMSE',
            color_continuous_scale='Reds_r',
            text='RMSE'
        )
        fig2.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Prediction accuracy plots
    st.markdown("###  Prediction Accuracy Visualization")
    
    selected_model = st.selectbox("Select Model to Visualize", 
                                  ['Model 1', 'Model 2', 'Model 3', 'Model 4'])
    
    model_data = results[selected_model]
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=model_data['y_test'],
        y=model_data['y_pred'],
        mode='markers',
        name='Predictions',
        marker=dict(size=8, opacity=0.6, color='blue')
    ))
    
    fig3.add_trace(go.Scatter(
        x=[model_data['y_test'].min(), model_data['y_test'].max()],
        y=[model_data['y_test'].min(), model_data['y_test'].max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig3.update_layout(
        title=f'{selected_model}: Predicted vs Actual Prices (R² = {model_data["r2"]:.3f})',
        xaxis_title='Actual Price ($)',
        yaxis_title='Predicted Price ($)',
        hovermode='closest',
        height=500
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# ========== DATA EXPLORER PAGE ==========
elif page == " Data Explorer":
    
    if st.session_state.data is None:
        st.warning(" Please load data first from the Home page!")
        st.stop()
    
    data = st.session_state.data
    
    st.markdown("##  Dataset Explorer")
    
    # Quick stats
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", len(data))
    with col2:
        st.metric("Total Columns", len(data.columns))
    with col3:
        st.metric("Avg Price", f"${data['Historical_Cost_of_Ride'].mean():.2f}")
    with col4:
        st.metric("Min Price", f"${data['Historical_Cost_of_Ride'].min():.2f}")
    with col5:
        st.metric("Max Price", f"${data['Historical_Cost_of_Ride'].max():.2f}")
    
    st.markdown("---")
    
    # Data view options
    view_option = st.radio("View", ["First 100 Rows", "Random Sample", "Statistical Summary"], horizontal=True)
    
    if view_option == "First 100 Rows":
        st.dataframe(data.head(100), use_container_width=True)
    elif view_option == "Random Sample":
        sample_size = st.slider("Sample size", 10, 100, 50)
        st.dataframe(data.sample(sample_size), use_container_width=True)
    else:
        st.dataframe(data.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("###  Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(
            data, x='Historical_Cost_of_Ride',
            title='Price Distribution',
            nbins=50,
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = px.scatter(
            data, x='Number_of_Riders', y='Historical_Cost_of_Ride',
            title='Price vs Number of Riders',
            opacity=0.5
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig2 = px.box(
            data, y='Historical_Cost_of_Ride',
            title='Price Box Plot'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        fig4 = px.scatter(
            data, x='Expected_Ride_Duration', y='Historical_Cost_of_Ride',
            title='Price vs Ride Duration',
            opacity=0.5
        )
        st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>OptiPrice Prognosticator v1.0</strong></p>
    <p>Built with  using Streamlit | Advanced ML Price Prediction System</p>
</div>
""", unsafe_allow_html=True)