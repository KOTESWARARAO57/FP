import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List
import streamlit as st
import random

class MedicalAudioClassifier:
    """Mock medical audio classifier for demonstration purposes."""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.classes = [
            'Normal Breathing',
            'Asthma',
            'Pneumonia',
            'Bronchitis',
            'COPD',
            'COVID-19 Symptoms'
        ]
        self.is_trained = False
        self._train_mock_model()
    
    def _generate_mock_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for demonstration."""
        # Create features that somewhat mimic real audio features
        # In practice, this would be replaced with real medical audio data
        n_features = 39  # Typical number of features from audio processing
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(len(self.classes), n_samples)
        
        # Add some pattern to make predictions more realistic
        for i in range(len(self.classes)):
            mask = y == i
            X[mask] += np.random.randn(n_features) * 0.5  # Add class-specific bias
        
        return X, y
    
    def _train_mock_model(self):
        """Train the mock model with synthetic data."""
        try:
            X, y = self._generate_mock_training_data()
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
        except Exception as e:
            st.error(f"Error training audio model: {str(e)}")
    
    def predict(self, features: np.ndarray) -> Dict:
        """Make prediction on audio features."""
        try:
            if not self.is_trained or features.size == 0:
                return {
                    'predicted_class': 'Unknown',
                    'confidence': 0.0,
                    'all_probabilities': {}
                }
            
            # Ensure features have the right shape
            if features.shape[1] != 39:  # Expected number of features
                # Pad or truncate features to match expected size
                if features.shape[1] < 39:
                    padding = np.zeros((features.shape[0], 39 - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :39]
            
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class_idx = np.argmax(probabilities)
            
            results = {
                'predicted_class': self.classes[predicted_class_idx],
                'confidence': float(probabilities[predicted_class_idx]),
                'all_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.classes, probabilities)
                }
            }
            
            return results
        except Exception as e:
            st.error(f"Error making audio prediction: {str(e)}")
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'all_probabilities': {}
            }

class MedicalImageClassifier:
    """Mock medical image classifier for demonstration purposes."""
    
    def __init__(self):
        self.classes = [
            'Normal',
            'Melanoma',
            'Basal Cell Carcinoma',
            'Seborrheic Keratosis',
            'Actinic Keratosis',
            'Dermatofibroma',
            'Vascular Lesion'
        ]
        # Simulate a pre-trained model with some randomness but consistency
        self.model_weights = self._initialize_mock_weights()
    
    def _initialize_mock_weights(self) -> Dict:
        """Initialize mock model weights for consistent predictions."""
        np.random.seed(42)  # For reproducible results
        return {
            'feature_weights': np.random.randn(10, len(self.classes)),
            'bias': np.random.randn(len(self.classes))
        }
    
    def _extract_simple_features(self, image_array: np.ndarray) -> np.ndarray:
        """Extract simple features from preprocessed image."""
        try:
            if image_array.size == 0:
                return np.zeros(10)
            
            # Flatten image and extract basic statistical features
            flat_image = image_array.flatten()
            
            features = [
                np.mean(flat_image),
                np.std(flat_image),
                np.median(flat_image),
                np.min(flat_image),
                np.max(flat_image),
                np.percentile(flat_image, 25),
                np.percentile(flat_image, 75),
                len(flat_image[flat_image > 0.5]) / len(flat_image),  # High intensity ratio
                len(flat_image[flat_image < 0.2]) / len(flat_image),  # Low intensity ratio
                np.sum(np.diff(flat_image[:1000])**2) if len(flat_image) > 1000 else 0  # Texture measure
            ]
            
            return np.array(features)
        except Exception as e:
            st.error(f"Error extracting image features: {str(e)}")
            return np.zeros(10)
    
    def predict(self, image_array: np.ndarray) -> Dict:
        """Make prediction on preprocessed image."""
        try:
            if image_array.size == 0:
                return {
                    'predicted_class': 'Unknown',
                    'confidence': 0.0,
                    'all_probabilities': {}
                }
            
            # Extract features
            features = self._extract_simple_features(image_array)
            
            # Simulate model prediction
            logits = np.dot(features, self.model_weights['feature_weights']) + self.model_weights['bias']
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            probabilities = exp_logits / np.sum(exp_logits)
            
            predicted_class_idx = np.argmax(probabilities)
            
            results = {
                'predicted_class': self.classes[predicted_class_idx],
                'confidence': float(probabilities[predicted_class_idx]),
                'all_probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.classes, probabilities)
                }
            }
            
            return results
        except Exception as e:
            st.error(f"Error making image prediction: {str(e)}")
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'all_probabilities': {}
            }

class MedicalAISystem:
    """Main system that coordinates audio and image analysis."""
    
    def __init__(self):
        self.audio_classifier = MedicalAudioClassifier()
        self.image_classifier = MedicalImageClassifier()
    
    def get_audio_prediction(self, features: np.ndarray) -> Dict:
        """Get prediction for audio input."""
        return self.audio_classifier.predict(features)
    
    def get_image_prediction(self, image_array: np.ndarray) -> Dict:
        """Get prediction for image input."""
        return self.image_classifier.predict(image_array)
    
    def get_combined_analysis(self, audio_results: Dict, image_results: Dict) -> Dict:
        """Provide combined analysis when both audio and image data are available."""
        try:
            combined_confidence = (
                audio_results.get('confidence', 0) + image_results.get('confidence', 0)
            ) / 2
            
            analysis = {
                'audio_prediction': audio_results.get('predicted_class', 'Unknown'),
                'image_prediction': image_results.get('predicted_class', 'Unknown'),
                'audio_confidence': audio_results.get('confidence', 0),
                'image_confidence': image_results.get('confidence', 0),
                'combined_confidence': combined_confidence,
                'recommendation': self._generate_recommendation(audio_results, image_results)
            }
            
            return analysis
        except Exception as e:
            st.error(f"Error generating combined analysis: {str(e)}")
            return {}
    
    def _generate_recommendation(self, audio_results: Dict, image_results: Dict) -> str:
        """Generate a simple recommendation based on predictions."""
        audio_pred = audio_results.get('predicted_class', '').lower()
        image_pred = image_results.get('predicted_class', '').lower()
        
        recommendations = []
        
        # Audio-based recommendations
        if 'covid' in audio_pred or 'pneumonia' in audio_pred:
            recommendations.append("Consider consulting a pulmonologist for respiratory symptoms.")
        elif 'asthma' in audio_pred or 'copd' in audio_pred:
            recommendations.append("Monitor breathing patterns and consider respiratory therapy.")
        
        # Image-based recommendations
        if 'melanoma' in image_pred or 'carcinoma' in image_pred:
            recommendations.append("Seek immediate dermatological evaluation for skin lesion.")
        elif image_pred != 'normal':
            recommendations.append("Consider dermatological consultation for skin condition assessment.")
        
        if not recommendations:
            recommendations.append("Results appear normal. Continue regular health monitoring.")
        
        return " ".join(recommendations)
