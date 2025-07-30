import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import streamlit as st
from typing import Tuple, Optional

class AudioProcessor:
    """Handles audio file processing and analysis for medical AI predictions."""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
    
    def validate_audio_file(self, file) -> bool:
        """Validate if the uploaded file is a supported audio format."""
        if file is None:
            return False
        
        file_extension = file.name.lower().split('.')[-1]
        return f'.{file_extension}' in self.supported_formats
    
    def load_audio(self, file) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file and return audio data and sample rate."""
        try:
            # Save the uploaded file temporarily
            audio_data, sample_rate = librosa.load(file, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")
            return None, None
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """Extract relevant audio features for medical analysis."""
        try:
            features = {}
            
            # Basic audio statistics
            features['duration'] = len(audio_data) / sample_rate
            features['sample_rate'] = sample_rate
            features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # MFCC features (commonly used in medical audio analysis)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma(y=audio_data, sr=sample_rate)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            return features
        except Exception as e:
            st.error(f"Error extracting audio features: {str(e)}")
            return {}
    
    def create_waveform_plot(self, audio_data: np.ndarray, sample_rate: int) -> plt.Figure:
        """Create a waveform visualization of the audio data."""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        ax.plot(time_axis, audio_data, color='blue', alpha=0.7)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_spectrogram_plot(self, audio_data: np.ndarray, sample_rate: int) -> plt.Figure:
        """Create a spectrogram visualization of the audio data."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram')
        plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        return fig
    
    def preprocess_for_prediction(self, features: dict) -> np.ndarray:
        """Preprocess extracted features for ML model input."""
        try:
            # Flatten and normalize features for model input
            feature_vector = []
            
            # Basic features
            feature_vector.extend([
                features.get('duration', 0),
                features.get('rms_energy', 0),
                features.get('zero_crossing_rate', 0),
                features.get('spectral_centroid_mean', 0),
                features.get('spectral_centroid_std', 0)
            ])
            
            # MFCC features
            if 'mfcc_mean' in features:
                feature_vector.extend(features['mfcc_mean'].flatten())
                feature_vector.extend(features['mfcc_std'].flatten())
            
            # Chroma features
            if 'chroma_mean' in features:
                feature_vector.extend(features['chroma_mean'].flatten())
            
            return np.array(feature_vector).reshape(1, -1)
        except Exception as e:
            st.error(f"Error preprocessing audio features: {str(e)}")
            return np.array([]).reshape(1, -1)
