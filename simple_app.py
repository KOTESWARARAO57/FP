import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Medical AI Assistant")
st.markdown("**Upload medical data for AI-powered analysis and predictions**")

# Sidebar for navigation
st.sidebar.title("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Audio Analysis", "Image Analysis", "Data Prediction"]
)

# Medical conditions for prediction
CONDITIONS = [
    "Healthy",
    "Respiratory Issue", 
    "Cardiovascular Condition",
    "Neurological Concern",
    "Metabolic Disorder"
]

def generate_sample_data(n_samples=100):
    """Generate sample medical data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    heart_rate = np.random.normal(72, 12, n_samples)
    blood_pressure = np.random.normal(120, 15, n_samples)
    temperature = np.random.normal(98.6, 1.5, n_samples)
    oxygen_level = np.random.normal(98, 2, n_samples)
    
    # Create conditions based on feature ranges
    conditions = []
    for i in range(n_samples):
        if heart_rate[i] > 100 or blood_pressure[i] > 140:
            conditions.append("Cardiovascular Condition")
        elif temperature[i] > 100:
            conditions.append("Respiratory Issue")
        elif oxygen_level[i] < 95:
            conditions.append("Respiratory Issue")
        elif heart_rate[i] < 60:
            conditions.append("Neurological Concern")
        else:
            conditions.append("Healthy")
    
    return pd.DataFrame({
        'Heart Rate': heart_rate,
        'Blood Pressure': blood_pressure,
        'Temperature': temperature,
        'Oxygen Level': oxygen_level,
        'Condition': conditions
    })

def create_prediction_model(data):
    """Create and train a simple prediction model"""
    X = data[['Heart Rate', 'Blood Pressure', 'Temperature', 'Oxygen Level']]
    y = data['Condition']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Main content based on analysis type
if analysis_type == "Audio Analysis":
    st.header("🎵 Audio Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload an audio file", 
        type=['wav', 'mp3', 'm4a', 'flac']
    )
    
    if uploaded_file is not None:
        st.success("Audio file uploaded successfully!")
        
        # Simulate audio analysis
        with st.spinner("Analyzing audio features..."):
            # Mock audio features
            features = {
                "Duration": "45.2 seconds",
                "Sample Rate": "44.1 kHz",
                "Breathing Rate": "16 breaths/min",
                "Voice Quality": "Clear",
                "Background Noise": "Low"
            }
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Audio Features")
                for key, value in features.items():
                    st.metric(key, value)
            
            with col2:
                st.subheader("AI Prediction")
                prediction = np.random.choice(CONDITIONS, p=[0.4, 0.3, 0.1, 0.1, 0.1])
                confidence = np.random.uniform(0.75, 0.95)
                
                st.success(f"**Prediction:** {prediction}")
                st.info(f"**Confidence:** {confidence:.1%}")
                
                # Create confidence chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Image Analysis":
    st.header("🖼️ Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a medical image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        st.success("Image uploaded successfully!")
        
        # Display the image
        st.image(uploaded_file, caption="Uploaded Medical Image", use_column_width=True)
        
        with st.spinner("Analyzing image..."):
            # Mock image analysis
            analysis_results = {
                "Image Quality": "High",
                "Resolution": "1024x768",
                "Contrast": "Good",
                "Clarity": "Excellent"
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image Properties")
                for key, value in analysis_results.items():
                    st.metric(key, value)
            
            with col2:
                st.subheader("AI Analysis")
                prediction = np.random.choice(CONDITIONS, p=[0.3, 0.25, 0.2, 0.15, 0.1])
                confidence = np.random.uniform(0.70, 0.92)
                
                st.success(f"**Analysis Result:** {prediction}")
                st.info(f"**Confidence:** {confidence:.1%}")
                
                # Risk assessment
                risk_level = "Low" if prediction == "Healthy" else "Medium" if confidence > 0.8 else "High"
                color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                st.markdown(f"**Risk Level:** <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)

elif analysis_type == "Data Prediction":
    st.header("📊 Medical Data Prediction")
    
    # Generate sample data
    if 'medical_data' not in st.session_state:
        st.session_state.medical_data = generate_sample_data()
    
    data = st.session_state.medical_data
    
    # Show data overview
    st.subheader("Sample Medical Data")
    st.dataframe(data.head(10))
    
    # Create and train model
    model, scaler = create_prediction_model(data)
    
    # User input for prediction
    st.subheader("Enter Your Medical Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
    with col2:
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    with col3:
        temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6)
    with col4:
        oxygen_level = st.number_input("Oxygen Level (%)", min_value=85, max_value=100, value=98)
    
    if st.button("🔍 Analyze My Data", type="primary"):
        # Make prediction
        user_data = np.array([[heart_rate, blood_pressure, temperature, oxygen_level]])
        user_data_scaled = scaler.transform(user_data)
        
        prediction = model.predict(user_data_scaled)[0]
        probabilities = model.predict_proba(user_data_scaled)[0]
        
        # Display results
        st.subheader("🎯 AI Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Predicted Condition:** {prediction}")
            max_prob = max(probabilities)
            st.info(f"**Confidence:** {max_prob:.1%}")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction == "Healthy":
                st.write("✅ Your vitals appear normal. Continue healthy lifestyle.")
            elif prediction == "Cardiovascular Condition":
                st.write("⚠️ Consider consulting a cardiologist. Monitor blood pressure.")
            elif prediction == "Respiratory Issue":
                st.write("🫁 Respiratory concerns detected. Consider pulmonology consultation.")
            else:
                st.write("🏥 Please consult with a healthcare professional for proper evaluation.")
        
        with col2:
            # Probability chart
            condition_names = model.classes_
            prob_df = pd.DataFrame({
                'Condition': condition_names,
                'Probability': probabilities * 100
            })
            
            fig = px.bar(
                prob_df, 
                x='Condition', 
                y='Probability',
                title="Prediction Probabilities",
                color='Probability',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data visualization
    st.subheader("📈 Data Insights")
    
    tab1, tab2 = st.tabs(["Distribution", "Correlations"])
    
    with tab1:
        # Feature distributions
        feature = st.selectbox("Select feature to visualize", 
                              ['Heart Rate', 'Blood Pressure', 'Temperature', 'Oxygen Level'])
        
        fig = px.histogram(data, x=feature, color='Condition', 
                          title=f"Distribution of {feature} by Condition")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation matrix
        numeric_data = data[['Heart Rate', 'Blood Pressure', 'Temperature', 'Oxygen Level']]
        correlation_matrix = numeric_data.corr()
        
        fig = px.imshow(correlation_matrix, 
                       title="Feature Correlations",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**⚠️ Disclaimer:** This is for demonstration purposes only. Always consult healthcare professionals for medical advice.")