import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from moviepy.editor import VideoFileClip
import io
import base64
from PIL import Image
import scipy.stats as stats

# Page configuration
st.set_page_config(
    page_title="Video Analysis AI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Advanced Video Analysis AI")
st.markdown("**Upload a video to extract and analyze audio features, visual frames, and get comprehensive AI insights**")

class VideoAnalyzer:
    def __init__(self):
        self.video_path = None
        self.audio_data = None
        self.sample_rate = None
        self.frames = []
        self.audio_features = {}
        self.visual_features = {}
        
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        try:
            with VideoFileClip(video_path) as video:
                audio = video.audio
                if audio is not None:
                    # Save audio to temporary file
                    temp_audio_path = "temp_audio.wav"
                    audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                    
                    # Load with librosa
                    self.audio_data, self.sample_rate = librosa.load(temp_audio_path, sr=None)
                    
                    # Clean up
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    
                    return True
                else:
                    st.warning("No audio track found in video")
                    return False
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            return False
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame interval to get evenly distributed frames
            frame_interval = max(1, total_frames // max_frames)
            
            frames = []
            frame_count = 0
            
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append({
                        'frame': frame_rgb,
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0
                    })
                
                frame_count += 1
            
            cap.release()
            self.frames = frames
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'extracted_frames': len(frames)
            }
            
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            return None
    
    def analyze_audio_features(self):
        """Extract comprehensive audio features"""
        if self.audio_data is None:
            return {}
        
        features = {}
        
        # Basic properties
        features['duration'] = len(self.audio_data) / self.sample_rate
        features['sample_rate'] = self.sample_rate
        
        # Spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sample_rate))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=self.audio_data, sr=self.sample_rate))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=self.audio_data, sr=self.sample_rate))
        
        # Rhythm features
        tempo, beats = librosa.beat.beat_track(y=self.audio_data, sr=self.sample_rate)
        features['tempo'] = tempo
        features['beat_count'] = len(beats)
        
        # Energy features
        features['rms_energy'] = np.mean(librosa.feature.rms(y=self.audio_data))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(self.audio_data))
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
        
        # Harmonic and percussive separation
        harmonic, percussive = librosa.effects.hpss(self.audio_data)
        features['harmonic_ratio'] = np.mean(harmonic) / (np.mean(harmonic) + np.mean(percussive))
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=self.audio_data, sr=self.sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['mean_pitch'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
        else:
            features['mean_pitch'] = 0
            features['pitch_std'] = 0
        
        self.audio_features = features
        return features
    
    def analyze_visual_features(self):
        """Extract visual features from frames"""
        if not self.frames:
            return {}
        
        features = {}
        all_brightness = []
        all_contrast = []
        all_colors = {'red': [], 'green': [], 'blue': []}
        motion_vectors = []
        
        prev_gray = None
        
        for i, frame_data in enumerate(self.frames):
            frame = frame_data['frame']
            
            # Convert to different color spaces
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            all_brightness.append(brightness)
            all_contrast.append(contrast)
            
            # Color analysis
            all_colors['red'].append(np.mean(frame[:,:,0]))
            all_colors['green'].append(np.mean(frame[:,:,1]))
            all_colors['blue'].append(np.mean(frame[:,:,2]))
            
            # Motion estimation (optical flow)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, 
                    np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                    None
                )
                if flow[0] is not None:
                    motion = np.linalg.norm(flow[0] - np.array([[100, 100]]))
                    motion_vectors.append(motion)
            
            prev_gray = gray.copy()
        
        # Aggregate features
        features['avg_brightness'] = np.mean(all_brightness)
        features['brightness_std'] = np.std(all_brightness)
        features['avg_contrast'] = np.mean(all_contrast)
        features['contrast_std'] = np.std(all_contrast)
        
        features['avg_red'] = np.mean(all_colors['red'])
        features['avg_green'] = np.mean(all_colors['green'])
        features['avg_blue'] = np.mean(all_colors['blue'])
        
        if motion_vectors:
            features['avg_motion'] = np.mean(motion_vectors)
            features['motion_std'] = np.std(motion_vectors)
        else:
            features['avg_motion'] = 0
            features['motion_std'] = 0
        
        # Scene consistency
        features['brightness_consistency'] = 1 / (1 + features['brightness_std'])
        features['color_balance'] = abs(features['avg_red'] - features['avg_green']) + abs(features['avg_green'] - features['avg_blue'])
        
        self.visual_features = features
        return features

def create_audio_visualizations(analyzer):
    """Create audio analysis visualizations"""
    if analyzer.audio_data is None:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Waveform', 'Spectrogram', 'MFCC Features', 'Spectral Features'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Waveform
    time_axis = np.linspace(0, len(analyzer.audio_data) / analyzer.sample_rate, len(analyzer.audio_data))
    fig.add_trace(
        go.Scatter(x=time_axis, y=analyzer.audio_data, name="Waveform", line=dict(color='blue')),
        row=1, col=1
    )
    
    # Spectrogram
    stft = librosa.stft(analyzer.audio_data)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    
    fig.add_trace(
        go.Heatmap(z=spectrogram_db[:50], colorscale='Viridis', name="Spectrogram"),
        row=1, col=2
    )
    
    # MFCC visualization
    mfcc_features = [f'mfcc_{i+1}' for i in range(13)]
    mfcc_values = [analyzer.audio_features[f] for f in mfcc_features]
    
    fig.add_trace(
        go.Bar(x=mfcc_features, y=mfcc_values, name="MFCC", marker_color='orange'),
        row=2, col=1
    )
    
    # Spectral features
    spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']
    spectral_values = [analyzer.audio_features[f] for f in spectral_features]
    
    fig.add_trace(
        go.Bar(x=spectral_features, y=spectral_values, name="Spectral", marker_color='green'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Comprehensive Audio Analysis")
    return fig

def create_visual_analysis(analyzer):
    """Create visual analysis dashboard"""
    if not analyzer.frames:
        return None, None
    
    # Frame grid
    cols = min(5, len(analyzer.frames))
    rows = (len(analyzer.frames) + cols - 1) // cols
    
    frame_fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Frame {i*len(analyzer.frames)//cols}" for i in range(min(cols*rows, len(analyzer.frames)))],
        specs=[[{"type": "scatter"}]*cols for _ in range(rows)]
    )
    
    # Visual features over time
    timestamps = [f['timestamp'] for f in analyzer.frames]
    brightness_values = []
    contrast_values = []
    
    for frame_data in analyzer.frames:
        frame = frame_data['frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness_values.append(np.mean(gray))
        contrast_values.append(np.std(gray))
    
    features_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Brightness Over Time', 'Contrast Over Time', 'Color Distribution', 'Motion Analysis')
    )
    
    # Brightness
    features_fig.add_trace(
        go.Scatter(x=timestamps, y=brightness_values, name="Brightness", line=dict(color='yellow')),
        row=1, col=1
    )
    
    # Contrast
    features_fig.add_trace(
        go.Scatter(x=timestamps, y=contrast_values, name="Contrast", line=dict(color='purple')),
        row=1, col=2
    )
    
    # Color distribution
    red_values = [np.mean(f['frame'][:,:,0]) for f in analyzer.frames]
    green_values = [np.mean(f['frame'][:,:,1]) for f in analyzer.frames]
    blue_values = [np.mean(f['frame'][:,:,2]) for f in analyzer.frames]
    
    features_fig.add_trace(go.Scatter(x=timestamps, y=red_values, name="Red", line=dict(color='red')), row=2, col=1)
    features_fig.add_trace(go.Scatter(x=timestamps, y=green_values, name="Green", line=dict(color='green')), row=2, col=1)
    features_fig.add_trace(go.Scatter(x=timestamps, y=blue_values, name="Blue", line=dict(color='blue')), row=2, col=1)
    
    features_fig.update_layout(height=600, title_text="Visual Feature Analysis")
    
    return features_fig, analyzer.frames

def perform_ai_analysis(analyzer):
    """Perform AI analysis on extracted features"""
    # Combine audio and visual features
    all_features = {}
    all_features.update(analyzer.audio_features)
    all_features.update(analyzer.visual_features)
    
    # Create feature vector
    feature_names = list(all_features.keys())
    feature_values = list(all_features.values())
    
    # Mock classification (in real app, use trained models)
    # Generate synthetic training data for demonstration
    np.random.seed(42)
    n_samples = 100
    n_features = len(feature_values)
    
    # Create synthetic dataset
    X_synthetic = np.random.randn(n_samples, n_features)
    conditions = ['Normal', 'Respiratory Issue', 'Speech Disorder', 'Movement Disorder', 'Emotional Distress']
    y_synthetic = np.random.choice(conditions, n_samples)
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_synthetic)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_synthetic)
    
    # Predict on current features
    current_features = np.array(feature_values).reshape(1, -1)
    current_scaled = scaler.transform(current_features)
    
    prediction = model.predict(current_scaled)[0]
    probabilities = model.predict_proba(current_scaled)[0]
    confidence = max(probabilities)
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': dict(zip(conditions, probabilities)),
        'feature_importance': feature_importance,
        'all_features': all_features
    }

# Main app interface
analyzer = VideoAnalyzer()

# File upload
uploaded_file = st.file_uploader(
    "Upload a video file", 
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv']
)

if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    st.success("Video uploaded successfully!")
    
    # Display video
    st.video(uploaded_file)
    
    # Analysis progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Extract audio
    status_text.text("Extracting audio from video...")
    audio_success = analyzer.extract_audio_from_video(video_path)
    progress_bar.progress(25)
    
    # Step 2: Extract frames
    status_text.text("Extracting frames from video...")
    video_info = analyzer.extract_frames(video_path)
    progress_bar.progress(50)
    
    # Step 3: Analyze audio
    if audio_success:
        status_text.text("Analyzing audio features...")
        audio_features = analyzer.analyze_audio_features()
        progress_bar.progress(75)
    
    # Step 4: Analyze visual features
    status_text.text("Analyzing visual features...")
    visual_features = analyzer.analyze_visual_features()
    progress_bar.progress(90)
    
    # Step 5: AI analysis
    status_text.text("Performing AI analysis...")
    ai_results = perform_ai_analysis(analyzer)
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Display results
    st.header("📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Video Duration", f"{video_info['duration']:.1f}s" if video_info else "N/A")
    with col2:
        st.metric("Audio Duration", f"{analyzer.audio_features.get('duration', 0):.1f}s" if audio_success else "No Audio")
    with col3:
        st.metric("Frames Extracted", video_info['extracted_frames'] if video_info else 0)
    with col4:
        st.metric("AI Confidence", f"{ai_results['confidence']:.1%}")
    
    # AI Prediction
    st.subheader("🎯 AI Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Prediction:** {ai_results['prediction']}")
        st.info(f"**Confidence:** {ai_results['confidence']:.1%}")
        
        # Probability distribution
        prob_df = pd.DataFrame(list(ai_results['probabilities'].items()), 
                              columns=['Condition', 'Probability'])
        fig_prob = px.bar(prob_df, x='Condition', y='Probability', 
                         title="Prediction Probabilities",
                         color='Probability', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with col2:
        # Feature importance
        importance_df = pd.DataFrame(list(ai_results['feature_importance'].items()), 
                                    columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                               orientation='h', title="Top 10 Important Features")
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Audio Analysis", "Visual Analysis", "Combined Features", "Raw Data"])
    
    with tab1:
        if audio_success:
            st.subheader("🎵 Audio Feature Analysis")
            audio_fig = create_audio_visualizations(analyzer)
            if audio_fig:
                st.plotly_chart(audio_fig, use_container_width=True)
            
            # Audio features table
            audio_df = pd.DataFrame(list(analyzer.audio_features.items()), 
                                   columns=['Feature', 'Value'])
            st.dataframe(audio_df)
        else:
            st.warning("No audio analysis available")
    
    with tab2:
        st.subheader("🖼️ Visual Feature Analysis")
        visual_fig, frames = create_visual_analysis(analyzer)
        if visual_fig:
            st.plotly_chart(visual_fig, use_container_width=True)
        
        # Frame gallery
        if frames:
            st.subheader("Extracted Frames")
            cols = st.columns(5)
            for i, frame_data in enumerate(frames[:10]):  # Show first 10 frames
                with cols[i % 5]:
                    st.image(frame_data['frame'], caption=f"Frame {frame_data['frame_number']}")
        
        # Visual features table
        visual_df = pd.DataFrame(list(analyzer.visual_features.items()), 
                               columns=['Feature', 'Value'])
        st.dataframe(visual_df)
    
    with tab3:
        st.subheader("🔄 Combined Multimodal Analysis")
        
        # Feature correlation
        feature_df = pd.DataFrame([ai_results['all_features']])
        if len(feature_df.columns) > 1:
            corr_matrix = feature_df.corr()
            fig_corr = px.imshow(corr_matrix, title="Feature Correlations")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # PCA analysis
        if len(ai_results['all_features']) > 2:
            feature_array = np.array(list(ai_results['all_features'].values())).reshape(1, -1)
            # Create synthetic data for PCA visualization
            synthetic_data = np.random.randn(50, len(ai_results['all_features']))
            combined_data = np.vstack([synthetic_data, feature_array])
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined_data)
            
            # Plot PCA
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Scatter(
                x=pca_result[:-1, 0], y=pca_result[:-1, 1],
                mode='markers', name='Reference Data',
                marker=dict(color='lightblue', size=8)
            ))
            fig_pca.add_trace(go.Scatter(
                x=[pca_result[-1, 0]], y=[pca_result[-1, 1]],
                mode='markers', name='Your Video',
                marker=dict(color='red', size=15, symbol='star')
            ))
            fig_pca.update_layout(title="PCA Analysis - Video Position in Feature Space")
            st.plotly_chart(fig_pca, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Raw Data Export")
        
        # All features table
        all_features_df = pd.DataFrame([ai_results['all_features']])
        st.dataframe(all_features_df)
        
        # Download button
        csv = all_features_df.to_csv(index=False)
        st.download_button(
            label="Download Feature Data as CSV",
            data=csv,
            file_name="video_analysis_features.csv",
            mime="text/csv"
        )
    
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)

else:
    st.info("Please upload a video file to begin analysis")
    
    # Sample features explanation
    st.subheader("📖 What This App Analyzes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Audio Features:**
        - Spectral analysis (centroid, bandwidth, rolloff)
        - Rhythm and tempo detection
        - Energy and pitch analysis
        - MFCC coefficients
        - Harmonic content
        """)
    
    with col2:
        st.markdown("""
        **Visual Features:**
        - Frame brightness and contrast
        - Color distribution analysis
        - Motion detection
        - Scene consistency
        - Temporal changes
        """)

# Footer
st.markdown("---")
st.markdown("**⚠️ Disclaimer:** This is for demonstration and research purposes only. Always consult healthcare professionals for medical advice.")