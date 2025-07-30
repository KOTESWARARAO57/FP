import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa
import streamlit as st
from typing import Tuple, Optional, List, Dict, Any
import os
import tempfile
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

class FacialMicroExpressionAnalyzer:
    """Analyzes facial micro-expressions for neuropsychiatric indicators."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.expression_features = []
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def extract_facial_landmarks(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict:
        """Extract facial landmark features from detected face region."""
        x, y, w, h = face_coords
        face_region = frame[y:y+h, x:x+w]
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        
        # Extract basic facial features
        features = {
            'face_area': w * h,
            'aspect_ratio': w / h if h > 0 else 0,
            'eye_region_intensity': np.mean(gray_face[:h//3, :]) if h > 0 else 0,
            'mouth_region_intensity': np.mean(gray_face[2*h//3:, :]) if h > 0 else 0,
            'face_symmetry': self._calculate_symmetry(gray_face),
            'texture_variance': np.var(gray_face),
            'edge_density': self._calculate_edge_density(gray_face)
        }
        
        return features
    
    def _calculate_symmetry(self, face_region: np.ndarray) -> float:
        """Calculate facial symmetry score."""
        if face_region.size == 0:
            return 0.0
        
        h, w = face_region.shape
        left_half = face_region[:, :w//2]
        right_half = np.fliplr(face_region[:, w//2:])
        
        # Resize to match if dimensions differ
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate correlation as symmetry measure
        if left_half.size > 0 and right_half.size > 0:
            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        return 0.0
    
    def _calculate_edge_density(self, face_region: np.ndarray) -> float:
        """Calculate edge density in face region."""
        if face_region.size == 0:
            return 0.0
        
        edges = cv2.Canny(face_region, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def analyze_micro_expressions(self, frames: List[np.ndarray]) -> Dict:
        """Analyze micro-expressions across video frames."""
        frame_features = []
        face_tracking = []
        
        for i, frame in enumerate(frames):
            faces = self.detect_faces(frame)
            
            if len(faces) > 0:
                # Use the largest face detected
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                features = self.extract_facial_landmarks(frame, largest_face)
                features['frame_number'] = i
                frame_features.append(features)
                face_tracking.append(largest_face)
        
        if not frame_features:
            return {'error': 'No faces detected in video frames'}
        
        # Calculate temporal dynamics
        analysis = self._calculate_temporal_features(frame_features)
        analysis['total_frames_analyzed'] = len(frame_features)
        analysis['face_detection_rate'] = len(frame_features) / len(frames)
        
        return analysis
    
    def _calculate_temporal_features(self, frame_features: List[Dict]) -> Dict:
        """Calculate temporal dynamics of facial features."""
        if not frame_features:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(frame_features)
        
        temporal_features = {}
        
        # Calculate variability metrics for each feature
        feature_columns = ['face_area', 'aspect_ratio', 'eye_region_intensity', 
                          'mouth_region_intensity', 'face_symmetry', 'texture_variance', 'edge_density']
        
        for feature in feature_columns:
            if feature in df.columns:
                temporal_features[f'{feature}_mean'] = df[feature].mean()
                temporal_features[f'{feature}_std'] = df[feature].std()
                temporal_features[f'{feature}_range'] = df[feature].max() - df[feature].min()
                
                # Calculate micro-movement indicators
                if len(df) > 1:
                    diff = df[feature].diff().dropna()
                    temporal_features[f'{feature}_micro_variance'] = diff.var()
                    temporal_features[f'{feature}_change_frequency'] = (diff.abs() > diff.std()).sum()
        
        return temporal_features

class SpeechParalinguisticsAnalyzer:
    """Analyzes speech paralinguistics including prosody, fluency, and lexical content."""
    
    def __init__(self):
        self.supported_languages = ['english', 'telugu']
        self.prosody_features = []
        
    def extract_prosodic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Extract prosodic features from audio."""
        features = {}
        
        try:
            # Fundamental frequency (F0) analysis
            f0 = librosa.yin(audio_data, fmin=80, fmax=400, sr=sample_rate)
            f0_valid = f0[f0 > 0]  # Remove unvoiced segments
            
            if len(f0_valid) > 0:
                features['f0_mean'] = np.mean(f0_valid)
                features['f0_std'] = np.std(f0_valid)
                features['f0_range'] = np.max(f0_valid) - np.min(f0_valid)
                features['f0_median'] = np.median(f0_valid)
            else:
                features.update({'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 'f0_median': 0})
            
            # Energy and intensity features
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            features['energy_mean'] = np.mean(rms_energy)
            features['energy_std'] = np.std(rms_energy)
            features['energy_range'] = np.max(rms_energy) - np.min(rms_energy)
            
            # Tempo and rhythm analysis
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            features['tempo'] = tempo
            features['rhythm_regularity'] = self._calculate_rhythm_regularity(beats, sample_rate)
            
            # Spectral features for voice quality
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Zero crossing rate for voicing analysis
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
        except Exception as e:
            st.warning(f"Error extracting prosodic features: {str(e)}")
            features = {key: 0 for key in ['f0_mean', 'f0_std', 'f0_range', 'f0_median',
                                          'energy_mean', 'energy_std', 'energy_range',
                                          'tempo', 'rhythm_regularity', 'spectral_centroid_mean',
                                          'spectral_centroid_std', 'zcr_mean', 'zcr_std']}
        
        return features
    
    def _calculate_rhythm_regularity(self, beats: np.ndarray, sample_rate: int) -> float:
        """Calculate rhythm regularity score."""
        if len(beats) < 2:
            return 0.0
        
        # Convert beat frames to time
        beat_times = beats / sample_rate
        intervals = np.diff(beat_times)
        
        if len(intervals) == 0:
            return 0.0
        
        # Calculate coefficient of variation as regularity measure
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval > 0:
            return 1 - (std_interval / mean_interval)  # Higher score = more regular
        return 0.0
    
    def analyze_fluency_patterns(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze speech fluency patterns."""
        features = {}
        
        try:
            # Voice Activity Detection (simple energy-based)
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            # Calculate frame-wise energy
            frames = librosa.util.frame(audio_data, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            frame_energy = np.sum(frames**2, axis=0)
            
            # Simple VAD threshold
            energy_threshold = np.percentile(frame_energy, 30)
            voiced_frames = frame_energy > energy_threshold
            
            # Calculate speech-to-pause ratio
            total_frames = len(voiced_frames)
            voiced_frame_count = np.sum(voiced_frames)
            features['speech_to_pause_ratio'] = voiced_frame_count / total_frames if total_frames > 0 else 0
            
            # Calculate pause patterns
            pause_segments = self._find_pause_segments(voiced_frames)
            if pause_segments:
                pause_durations = [end - start for start, end in pause_segments]
                features['avg_pause_duration'] = np.mean(pause_durations) * hop_length / sample_rate
                features['pause_frequency'] = len(pause_segments) / (len(audio_data) / sample_rate)
            else:
                features['avg_pause_duration'] = 0
                features['pause_frequency'] = 0
            
            # Speech rate estimation
            speech_segments = self._find_speech_segments(voiced_frames)
            if speech_segments:
                total_speech_time = sum(end - start for start, end in speech_segments) * hop_length / sample_rate
                features['speech_rate'] = voiced_frame_count / total_speech_time if total_speech_time > 0 else 0
            else:
                features['speech_rate'] = 0
                
        except Exception as e:
            st.warning(f"Error analyzing fluency patterns: {str(e)}")
            features = {'speech_to_pause_ratio': 0, 'avg_pause_duration': 0, 
                       'pause_frequency': 0, 'speech_rate': 0}
        
        return features
    
    def _find_pause_segments(self, voiced_frames: np.ndarray) -> List[Tuple[int, int]]:
        """Find pause segments in voiced frame sequence."""
        pause_segments = []
        in_pause = False
        start_pause = 0
        
        for i, is_voiced in enumerate(voiced_frames):
            if not is_voiced and not in_pause:
                in_pause = True
                start_pause = i
            elif is_voiced and in_pause:
                in_pause = False
                pause_segments.append((start_pause, i))
        
        # Handle case where audio ends in a pause
        if in_pause:
            pause_segments.append((start_pause, len(voiced_frames)))
        
        return pause_segments
    
    def _find_speech_segments(self, voiced_frames: np.ndarray) -> List[Tuple[int, int]]:
        """Find speech segments in voiced frame sequence."""
        speech_segments = []
        in_speech = False
        start_speech = 0
        
        for i, is_voiced in enumerate(voiced_frames):
            if is_voiced and not in_speech:
                in_speech = True
                start_speech = i
            elif not is_voiced and in_speech:
                in_speech = False
                speech_segments.append((start_speech, i))
        
        # Handle case where audio ends in speech
        if in_speech:
            speech_segments.append((start_speech, len(voiced_frames)))
        
        return speech_segments

class NeuropsychiatricDiseaseClassifier:
    """Specialized classifier for neuropsychiatric and metabolic diseases."""
    
    def __init__(self):
        self.diseases = [
            'Healthy',
            'Depression',
            'Parkinson\'s Disease', 
            'Hypothyroidism',
            'Anxiety Disorder',
            'Bipolar Disorder',
            'Schizophrenia',
            'Dementia'
        ]
        
        self.facial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.speech_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.multimodal_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.facial_scaler = StandardScaler()
        self.speech_scaler = StandardScaler()
        self.multimodal_scaler = StandardScaler()
        
        self.is_trained = False
        self._train_models()
    
    def _generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data for demonstration."""
        np.random.seed(42)
        
        # Facial features (micro-expression analysis output)
        n_facial_features = 35  # Based on FacialMicroExpressionAnalyzer output
        facial_X = np.random.randn(n_samples, n_facial_features)
        
        # Speech features (paralinguistics analysis output)  
        n_speech_features = 25  # Based on SpeechParalinguisticsAnalyzer output
        speech_X = np.random.randn(n_samples, n_speech_features)
        
        # Generate labels with realistic distribution
        labels = np.random.choice(len(self.diseases), n_samples, 
                                p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        
        # Add disease-specific patterns to make predictions more realistic
        for i in range(len(self.diseases)):
            mask = labels == i
            if np.any(mask):
                # Add characteristic patterns for each disease
                facial_X[mask] += np.random.randn(n_facial_features) * 0.3
                speech_X[mask] += np.random.randn(n_speech_features) * 0.3
        
        return facial_X, speech_X, labels, labels
    
    def _train_models(self):
        """Train the multimodal classification models."""
        try:
            facial_X, speech_X, facial_y, speech_y = self._generate_training_data()
            
            # Train individual modality models
            facial_X_scaled = self.facial_scaler.fit_transform(facial_X)
            self.facial_model.fit(facial_X_scaled, facial_y)
            
            speech_X_scaled = self.speech_scaler.fit_transform(speech_X)
            self.speech_model.fit(speech_X_scaled, speech_y)
            
            # Combine features for multimodal model
            combined_X = np.hstack([facial_X, speech_X])
            combined_X_scaled = self.multimodal_scaler.fit_transform(combined_X)
            self.multimodal_model.fit(combined_X_scaled, facial_y)  # Using same labels
            
            self.is_trained = True
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
    
    def predict_from_facial_features(self, facial_features: Dict) -> Dict:
        """Predict disease from facial micro-expression features."""
        if not self.is_trained:
            return {'predicted_class': 'Model not trained', 'confidence': 0.0}
        
        try:
            # Convert features dict to array
            feature_vector = self._dict_to_vector(facial_features, expected_size=35)
            if feature_vector.size == 0:
                return {'predicted_class': 'Invalid features', 'confidence': 0.0}
            
            feature_vector_scaled = self.facial_scaler.transform(feature_vector.reshape(1, -1))
            probabilities = self.facial_model.predict_proba(feature_vector_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            
            return {
                'predicted_class': self.diseases[predicted_idx],
                'confidence': float(probabilities[predicted_idx]),
                'all_probabilities': {
                    disease: float(prob) 
                    for disease, prob in zip(self.diseases, probabilities)
                },
                'modality': 'facial'
            }
        except Exception as e:
            st.error(f"Error in facial prediction: {str(e)}")
            return {'predicted_class': 'Error', 'confidence': 0.0}
    
    def predict_from_speech_features(self, speech_features: Dict) -> Dict:
        """Predict disease from speech paralinguistic features."""
        if not self.is_trained:
            return {'predicted_class': 'Model not trained', 'confidence': 0.0}
        
        try:
            # Convert features dict to array
            feature_vector = self._dict_to_vector(speech_features, expected_size=25)
            if feature_vector.size == 0:
                return {'predicted_class': 'Invalid features', 'confidence': 0.0}
            
            feature_vector_scaled = self.speech_scaler.transform(feature_vector.reshape(1, -1))
            probabilities = self.speech_model.predict_proba(feature_vector_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            
            return {
                'predicted_class': self.diseases[predicted_idx],
                'confidence': float(probabilities[predicted_idx]),
                'all_probabilities': {
                    disease: float(prob) 
                    for disease, prob in zip(self.diseases, probabilities)
                },
                'modality': 'speech'
            }
        except Exception as e:
            st.error(f"Error in speech prediction: {str(e)}")
            return {'predicted_class': 'Error', 'confidence': 0.0}
    
    def predict_multimodal(self, facial_features: Dict, speech_features: Dict) -> Dict:
        """Predict disease using combined multimodal features."""
        if not self.is_trained:
            return {'predicted_class': 'Model not trained', 'confidence': 0.0}
        
        try:
            # Convert features to vectors
            facial_vector = self._dict_to_vector(facial_features, expected_size=35)
            speech_vector = self._dict_to_vector(speech_features, expected_size=25)
            
            if facial_vector.size == 0 or speech_vector.size == 0:
                return {'predicted_class': 'Invalid features', 'confidence': 0.0}
            
            # Combine features
            combined_vector = np.concatenate([facial_vector, speech_vector])
            combined_vector_scaled = self.multimodal_scaler.transform(combined_vector.reshape(1, -1))
            
            probabilities = self.multimodal_model.predict_proba(combined_vector_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            
            return {
                'predicted_class': self.diseases[predicted_idx],
                'confidence': float(probabilities[predicted_idx]),
                'all_probabilities': {
                    disease: float(prob) 
                    for disease, prob in zip(self.diseases, probabilities)
                },
                'modality': 'multimodal'
            }
        except Exception as e:
            st.error(f"Error in multimodal prediction: {str(e)}")
            return {'predicted_class': 'Error', 'confidence': 0.0}
    
    def _dict_to_vector(self, features_dict: Dict, expected_size: int) -> np.ndarray:
        """Convert features dictionary to numpy vector."""
        if not features_dict:
            return np.zeros(expected_size)
        
        # Extract numeric values from dictionary
        values = []
        for value in features_dict.values():
            if isinstance(value, (int, float)):
                values.append(value)
            elif isinstance(value, np.ndarray):
                values.extend(value.flatten())
            elif isinstance(value, list):
                values.extend(value)
        
        # Ensure correct size
        if len(values) < expected_size:
            values.extend([0] * (expected_size - len(values)))
        elif len(values) > expected_size:
            values = values[:expected_size]
        
        return np.array(values, dtype=np.float32)