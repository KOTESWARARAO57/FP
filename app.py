import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List
import tempfile
import os
from utils.audio_processor import AudioProcessor
from utils.image_processor import ImageProcessor
from utils.video_processor import VideoProcessor
from utils.ml_models import MedicalAISystem
from utils.multimodal_processor import (
    FacialMicroExpressionAnalyzer, 
    SpeechParalinguisticsAnalyzer, 
    NeuropsychiatricDiseaseClassifier
)
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Medical AI Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize processors and models
@st.cache_resource
def load_medical_ai_system():
    """Load and cache the medical AI system."""
    return MedicalAISystem()

@st.cache_resource
def load_processors():
    """Load and cache the audio, image, and video processors."""
    return AudioProcessor(), ImageProcessor(), VideoProcessor()

@st.cache_resource
def load_neuropsychiatric_system():
    """Load and cache the neuropsychiatric analysis system."""
    return (FacialMicroExpressionAnalyzer(), 
            SpeechParalinguisticsAnalyzer(), 
            NeuropsychiatricDiseaseClassifier())

# Load systems
medical_ai = load_medical_ai_system()
audio_processor, image_processor, video_processor = load_processors()
facial_analyzer, speech_analyzer, neuropsych_classifier = load_neuropsychiatric_system()

def main():
    """Main application function."""
    st.title("🏥 Medical AI Diagnostic Assistant")
    st.markdown("""
    This application uses artificial intelligence to analyze audio and image data for medical diagnostic assistance.
    Please upload your audio files (cough sounds, breathing patterns) or medical images for analysis.
    
    **⚠️ Disclaimer**: This is a demonstration application. Results should not be used for actual medical diagnosis. 
    Always consult with healthcare professionals for medical concerns.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["Audio Analysis", "Image Analysis", "Video Analysis", "Neuropsychiatric Analysis", "Combined Analysis"]
    )
    
    # Create tabs based on selection
    if analysis_type == "Audio Analysis":
        audio_analysis_tab()
    elif analysis_type == "Image Analysis":
        image_analysis_tab()
    elif analysis_type == "Video Analysis":
        video_analysis_tab()
    elif analysis_type == "Neuropsychiatric Analysis":
        neuropsychiatric_analysis_tab()
    else:
        combined_analysis_tab()
    
    # Add information sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ Information")
        st.markdown("""
        **Supported Audio Formats:**
        - WAV, MP3, M4A, FLAC
        
        **Supported Image Formats:**
        - JPG, JPEG, PNG, BMP, TIFF
        
        **Supported Video Formats:**
        - MP4, AVI, MOV, MKV, WMV, FLV
        
        **Features:**
        - Audio waveform analysis
        - Spectrogram visualization
        - Medical image processing
        - Video frame extraction
        - Audio extraction from video
        - AI-powered predictions
        - Confidence scoring
        """)

def audio_analysis_tab():
    """Handle audio analysis functionality."""
    st.header("🎵 Audio Analysis")
    st.markdown("Upload audio files containing cough sounds, breathing patterns, or speech for medical analysis.")
    
    # File upload
    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Upload audio files in WAV, MP3, M4A, or FLAC format"
    )
    
    if uploaded_audio is not None:
        # Validate file
        if not audio_processor.validate_audio_file(uploaded_audio):
            st.error("Please upload a valid audio file (WAV, MP3, M4A, or FLAC)")
            return
        
        # Display file information
        st.success(f"File uploaded: {uploaded_audio.name}")
        file_size = len(uploaded_audio.getvalue()) / (1024 * 1024)
        st.info(f"File size: {file_size:.2f} MB")
        
        # Audio player
        st.audio(uploaded_audio, format='audio/wav')
        
        # Process audio
        with st.spinner("Processing audio file..."):
            audio_data, sample_rate = audio_processor.load_audio(uploaded_audio)
        
        if audio_data is not None:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Audio Information")
                duration = len(audio_data) / sample_rate
                st.metric("Duration", f"{duration:.2f} seconds")
                st.metric("Sample Rate", f"{sample_rate} Hz")
                st.metric("Channels", "1 (Mono)")
                
                # Extract and display features
                with st.spinner("Extracting audio features..."):
                    if audio_data is not None and sample_rate is not None:
                        features = audio_processor.extract_features(audio_data, sample_rate)
                    else:
                        features = {}
                
                if features:
                    st.subheader("🔍 Audio Features")
                    feature_df = pd.DataFrame([
                        {"Feature": "RMS Energy", "Value": f"{features.get('rms_energy', 0):.4f}"},
                        {"Feature": "Zero Crossing Rate", "Value": f"{features.get('zero_crossing_rate', 0):.4f}"},
                        {"Feature": "Spectral Centroid (Mean)", "Value": f"{features.get('spectral_centroid_mean', 0):.2f} Hz"},
                        {"Feature": "Spectral Centroid (Std)", "Value": f"{features.get('spectral_centroid_std', 0):.2f} Hz"},
                    ])
                    st.dataframe(feature_df, use_container_width=True)
            
            with col2:
                st.subheader("📈 Visualizations")
                
                # Waveform plot
                if audio_data is not None and sample_rate is not None:
                    with st.spinner("Creating waveform..."):
                        waveform_fig = audio_processor.create_waveform_plot(audio_data, sample_rate)
                        st.pyplot(waveform_fig)
                    
                    # Spectrogram plot
                    with st.spinner("Creating spectrogram..."):
                        spectrogram_fig = audio_processor.create_spectrogram_plot(audio_data, sample_rate)
                        st.pyplot(spectrogram_fig)
            
            # AI Prediction
            st.subheader("🤖 AI Medical Analysis")
            
            if st.button("Analyze Audio for Medical Conditions", type="primary"):
                with st.spinner("Analyzing audio with AI model..."):
                    # Simulate processing time
                    time.sleep(2)
                    
                    # Preprocess for prediction
                    feature_vector = audio_processor.preprocess_for_prediction(features)
                    
                    # Get prediction
                    prediction_results = medical_ai.get_audio_prediction(feature_vector)
                
                # Display results
                display_prediction_results(prediction_results, "audio")

def image_analysis_tab():
    """Handle image analysis functionality."""
    st.header("🖼️ Image Analysis")
    st.markdown("Upload medical images, skin condition photos, or diagnostic images for AI analysis.")
    
    # File upload
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload medical images in JPG, PNG, BMP, or TIFF format"
    )
    
    if uploaded_image is not None:
        # Validate file
        if not image_processor.validate_image_file(uploaded_image):
            st.error("Please upload a valid image file (JPG, PNG, BMP, or TIFF)")
            return
        
        # Load and display image
        with st.spinner("Loading image..."):
            image = image_processor.load_image(uploaded_image)
        
        if image is not None:
            # Display file information
            st.success(f"Image uploaded: {uploaded_image.name}")
            image_info = image_processor.extract_image_info(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Image Information")
                info_df = pd.DataFrame([
                    {"Property": "Width", "Value": f"{image_info.get('width', 0)} pixels"},
                    {"Property": "Height", "Value": f"{image_info.get('height', 0)} pixels"},
                    {"Property": "Format", "Value": image_info.get('format', 'Unknown')},
                    {"Property": "Mode", "Value": image_info.get('mode', 'Unknown')},
                    {"Property": "Size", "Value": f"{image_info.get('size_mb', 0):.2f} MB"},
                ])
                st.dataframe(info_df, use_container_width=True)
                
                # Extract features
                with st.spinner("Extracting image features..."):
                    features = image_processor.extract_features(image)
                
                if features:
                    st.subheader("🔍 Image Features")
                    feature_df = pd.DataFrame([
                        {"Feature": "Mean Intensity", "Value": f"{features.get('mean_intensity', 0):.2f}"},
                        {"Feature": "Contrast", "Value": f"{features.get('contrast', 0):.2f}"},
                        {"Feature": "Brightness", "Value": f"{features.get('brightness', 0):.2f}"},
                        {"Feature": "Edge Density", "Value": f"{features.get('edge_density', 0):.4f}"},
                    ])
                    st.dataframe(feature_df, use_container_width=True)
            
            with col2:
                st.subheader("📸 Original Image")
                st.image(image, caption="Uploaded Medical Image", use_container_width=True)
            
            # Enhanced visualization
            st.subheader("📈 Image Analysis Visualization")
            with st.spinner("Creating enhanced visualization..."):
                viz_fig, analysis = image_processor.create_enhanced_visualization(image)
                if viz_fig is not None:
                    st.pyplot(viz_fig)
            
            # AI Prediction
            st.subheader("🤖 AI Medical Analysis")
            
            if st.button("Analyze Image for Medical Conditions", type="primary"):
                with st.spinner("Analyzing image with AI model..."):
                    # Simulate processing time
                    time.sleep(2)
                    
                    # Preprocess for prediction
                    preprocessed_image = image_processor.preprocess_for_prediction(image)
                    
                    # Get prediction
                    prediction_results = medical_ai.get_image_prediction(preprocessed_image)
                
                # Display results
                display_prediction_results(prediction_results, "image")
                
                # Show image with prediction overlay
                if prediction_results.get('predicted_class') != 'Error':
                    st.subheader("📋 Prediction Overlay")
                    overlay_image = image_processor.create_prediction_overlay(image, prediction_results)
                    st.image(overlay_image, caption="Image with AI Predictions", use_container_width=True)

def video_analysis_tab():
    """Handle video analysis functionality - extract audio and frames."""
    st.header("🎬 Video Analysis")
    st.markdown("Process video files to extract audio and frames for medical analysis. Videos will be analyzed for both audio patterns and visual content.")
    
    # Option 1: Upload new video
    st.subheader("📤 Upload Video File")
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload video files for audio and frame extraction"
    )
    
    # Option 2: Use existing extracted videos
    st.subheader("📁 Use Extracted Video Data")
    st.markdown("Select from the extracted video files:")
    
    video_files = []
    data_dir = "data/Healthy"
    if os.path.exists(data_dir):
        video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
    
    if video_files:
        selected_video = st.selectbox(
            "Choose a video file:",
            ["None"] + video_files
        )
        
        if selected_video != "None":
            video_path = os.path.join(data_dir, selected_video)
            process_video_file(video_path, selected_video)
    else:
        st.info("No extracted video files found. Please upload a video or check if the data was extracted properly.")
    
    # Process uploaded video
    if uploaded_video is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.getvalue())
            temp_path = tmp_file.name
        
        try:
            process_video_file(temp_path, uploaded_video.name)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

def process_video_file(video_path: str, video_name: str):
    """Process a video file and extract audio and frames."""
    import os
    import tempfile
    
    if not video_processor.validate_video_file(video_path):
        st.error("Invalid video file format.")
        return
    
    st.success(f"Processing video: {video_name}")
    
    # Get video information
    with st.spinner("Analyzing video..."):
        video_info = video_processor.get_video_info(video_path)
    
    if video_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Video Information")
            info_df = pd.DataFrame([
                {"Property": "Duration", "Value": f"{video_info.get('duration', 0):.2f} seconds"},
                {"Property": "Frame Rate", "Value": f"{video_info.get('fps', 0):.2f} FPS"},
                {"Property": "Resolution", "Value": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"},
                {"Property": "Total Frames", "Value": f"{video_info.get('frame_count', 0)}"},
                {"Property": "File Size", "Value": f"{video_info.get('file_size_mb', 0):.2f} MB"},
            ])
            st.dataframe(info_df, use_container_width=True)
        
        with col2:
            st.subheader("🎬 Video Preview")
            # Display video player
            try:
                if os.path.exists(video_path):
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
            except Exception as e:
                st.warning(f"Could not display video preview: {str(e)}")
        
        # Frame extraction section
        st.subheader("🖼️ Frame Extraction")
        
        extraction_method = st.selectbox(
            "Choose frame extraction method:",
            ["Uniform Sampling", "Key Frame Detection"]
        )
        
        if extraction_method == "Uniform Sampling":
            num_frames = st.slider("Number of frames to extract:", 5, 20, 10)
            
            if st.button("Extract Frames", type="primary"):
                with st.spinner("Extracting frames..."):
                    frames = video_processor.extract_frames(video_path, num_frames, method='uniform')
                
                if frames:
                    st.success(f"Extracted {len(frames)} frames")
                    
                    # Display frame grid
                    fig = video_processor.create_frame_grid(frames)
                    if fig:
                        st.pyplot(fig)
                    
                    # Analyze frame quality
                    quality_metrics = video_processor.analyze_frame_quality(frames)
                    if quality_metrics:
                        st.subheader("📈 Frame Quality Analysis")
                        quality_df = pd.DataFrame([
                            {"Metric": "Average Brightness", "Value": f"{quality_metrics.get('mean_brightness', 0):.2f}"},
                            {"Metric": "Average Contrast", "Value": f"{quality_metrics.get('mean_contrast', 0):.2f}"},
                            {"Metric": "Average Blur Score", "Value": f"{quality_metrics.get('mean_blur_score', 0):.2f}"},
                        ])
                        st.dataframe(quality_df, use_container_width=True)
                    
                    # AI Analysis on frames
                    st.subheader("🤖 AI Analysis on Frames")
                    if st.button("Analyze Frames with AI", key="frame_analysis"):
                        with st.spinner("Analyzing frames..."):
                            preprocessed_frames = video_processor.preprocess_frames_for_prediction(frames)
                            
                            # Analyze each frame individually and aggregate results
                            frame_predictions = []
                            for i, frame in enumerate(frames):
                                frame_pred = medical_ai.get_image_prediction(frame.reshape(1, *frame.shape))
                                frame_predictions.append({
                                    'frame': i + 1,
                                    'prediction': frame_pred.get('predicted_class', 'Unknown'),
                                    'confidence': frame_pred.get('confidence', 0)
                                })
                            
                            # Display frame-by-frame results
                            st.write("**Frame-by-Frame Analysis:**")
                            frames_df = pd.DataFrame(frame_predictions)
                            st.dataframe(frames_df, use_container_width=True)
                            
                            # Aggregate analysis
                            if frame_predictions:
                                avg_confidence = np.mean([fp['confidence'] for fp in frame_predictions])
                                most_common_pred = max(set([fp['prediction'] for fp in frame_predictions]), 
                                                     key=[fp['prediction'] for fp in frame_predictions].count)
                                
                                st.metric("Overall Prediction", most_common_pred)
                                st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        else:  # Key Frame Detection
            threshold = st.slider("Scene change threshold:", 10.0, 50.0, 30.0)
            
            if st.button("Extract Key Frames", type="primary"):
                with st.spinner("Detecting key frames..."):
                    key_frames = video_processor.extract_key_frames(video_path, threshold)
                
                if key_frames:
                    st.success(f"Extracted {len(key_frames)} key frames")
                    
                    # Display key frames with frame numbers
                    for i, (frame_idx, frame) in enumerate(key_frames[:10]):  # Show max 10 frames
                        st.image(frame, caption=f"Key Frame at position {frame_idx}", width=300)
        
        # Audio extraction section
        st.subheader("🎵 Audio Extraction")
        
        if st.button("Extract Audio from Video", type="primary"):
            with st.spinner("Extracting audio..."):
                audio_data, sample_rate = video_processor.extract_audio_from_video(video_path)
            
            if audio_data is not None:
                st.success("Audio extracted successfully!")
                
                # Display audio information
                duration = len(audio_data) / sample_rate
                st.metric("Audio Duration", f"{duration:.2f} seconds")
                st.metric("Sample Rate", f"{sample_rate} Hz")
                
                # Audio player (create temporary audio file)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    import soundfile as sf
                    sf.write(tmp_audio.name, audio_data, sample_rate)
                    st.audio(tmp_audio.name, format='audio/wav')
                    os.unlink(tmp_audio.name)
                
                # Extract audio features
                with st.spinner("Analyzing audio features..."):
                    audio_features = audio_processor.extract_features(audio_data, sample_rate)
                
                if audio_features:
                    st.subheader("🔍 Audio Features")
                    feature_df = pd.DataFrame([
                        {"Feature": "RMS Energy", "Value": f"{audio_features.get('rms_energy', 0):.4f}"},
                        {"Feature": "Zero Crossing Rate", "Value": f"{audio_features.get('zero_crossing_rate', 0):.4f}"},
                        {"Feature": "Spectral Centroid (Mean)", "Value": f"{audio_features.get('spectral_centroid_mean', 0):.2f} Hz"},
                    ])
                    st.dataframe(feature_df, use_container_width=True)
                
                # AI Analysis on audio
                st.subheader("🤖 AI Analysis on Audio")
                if st.button("Analyze Audio with AI", key="audio_analysis"):
                    with st.spinner("Analyzing audio..."):
                        audio_vector = audio_processor.preprocess_for_prediction(audio_features)
                        audio_prediction = medical_ai.get_audio_prediction(audio_vector)
                    
                    display_prediction_results(audio_prediction, "audio")
            else:
                st.warning("No audio track found in the video or audio extraction failed.")

def neuropsychiatric_analysis_tab():
    """Handle advanced neuropsychiatric and metabolic disease analysis using multimodal features."""
    st.header("🧠 Neuropsychiatric & Metabolic Disease Analysis")
    st.markdown("""
    Advanced AI analysis for specific neuropsychiatric and metabolic conditions using multimodal behavioral patterns:
    - **Facial Micro-expressions**: Analyze subtle facial movements and expressions
    - **Speech Paralinguistics**: Examine prosody, fluency, and vocal patterns
    - **Supported Languages**: English and Telugu
    - **Target Conditions**: Depression, Parkinson's Disease, Hypothyroidism
    """)
    
    # Language selection
    st.subheader("🌐 Language Selection")
    language = st.selectbox(
        "Select language for analysis:",
        ["English", "Telugu", "Multilingual (Both)"]
    )
    
    # Input selection tabs
    input_tab1, input_tab2, input_tab3 = st.tabs(["📹 Video Analysis", "🎵 Audio Only", "🖼️ Image Only"])
    
    with input_tab1:
        st.subheader("📹 Multimodal Video Analysis")
        st.markdown("Upload a video containing both facial expressions and speech for comprehensive analysis.")
        
        # Video file selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Upload New Video:**")
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov'],
                key="neuro_video_upload"
            )
        
        with col2:
            st.write("**Use Extracted Videos:**")
            video_files = []
            data_dir = "data/Healthy"
            if os.path.exists(data_dir):
                video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
            
            if video_files:
                selected_video = st.selectbox(
                    "Choose from extracted videos:",
                    ["None"] + video_files,
                    key="neuro_video_select"
                )
                
                if selected_video != "None":
                    video_path = os.path.join(data_dir, selected_video)
                    # Auto-process the video immediately
                    auto_process_neuropsychiatric_video(video_path, selected_video, language)
        
        # Process uploaded video automatically
        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                temp_path = tmp_file.name
            
            try:
                # Auto-process immediately upon upload
                auto_process_neuropsychiatric_video(temp_path, uploaded_video.name, language)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    with input_tab2:
        st.subheader("🎵 Speech Paralinguistics Analysis")
        st.markdown("Analyze speech patterns for neuropsychiatric indicators.")
        
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            key="neuro_audio"
        )
        
        if uploaded_audio is not None:
            process_neuropsychiatric_audio(uploaded_audio, language)
    
    with input_tab3:
        st.subheader("🖼️ Facial Micro-expression Analysis")
        st.markdown("Analyze facial expressions and micro-movements for neuropsychiatric indicators.")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            key="neuro_image"
        )
        
        if uploaded_image is not None:
            process_neuropsychiatric_image(uploaded_image, language)

def auto_process_neuropsychiatric_video(video_path: str, video_name: str, language: str):
    """Automatically process video for neuropsychiatric analysis without button click."""
    st.success(f"Auto-processing {video_name} for neuropsychiatric analysis...")
    
    # Video information
    video_info = video_processor.get_video_info(video_path)
    if video_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{video_info.get('duration', 0):.1f}s")
        with col2:
            st.metric("Language", language)
        with col3:
            st.metric("Analysis Type", "Multimodal")
    
    # Automatically start analysis
    with st.spinner("Performing comprehensive neuropsychiatric analysis..."):
        # Extract frames for facial analysis
        frames = video_processor.extract_frames(video_path, num_frames=15, method='uniform')
        
        # Extract audio for speech analysis
        audio_data, sample_rate = video_processor.extract_audio_from_video(video_path)
        
        if frames and audio_data is not None:
            # Facial micro-expression analysis
            st.subheader("👤 Facial Micro-expression Analysis")
            facial_features = facial_analyzer.analyze_micro_expressions(frames)
            
            if 'error' not in facial_features:
                # Display facial analysis results
                facial_metrics = pd.DataFrame([
                    {"Metric": "Face Detection Rate", "Value": f"{facial_features.get('face_detection_rate', 0):.1%}"},
                    {"Metric": "Frames Analyzed", "Value": facial_features.get('total_frames_analyzed', 0)},
                    {"Metric": "Facial Symmetry (avg)", "Value": f"{facial_features.get('face_symmetry_mean', 0):.3f}"},
                    {"Metric": "Expression Variance", "Value": f"{facial_features.get('face_symmetry_std', 0):.3f}"},
                ])
                st.dataframe(facial_metrics, use_container_width=True)
                
                # Get facial prediction
                facial_prediction = neuropsych_classifier.predict_from_facial_features(facial_features)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Facial Analysis Result", facial_prediction.get('predicted_class', 'Unknown'))
                with col2:
                    st.metric("Confidence", f"{facial_prediction.get('confidence', 0):.1%}")
            
            # Speech paralinguistics analysis
            st.subheader("🗣️ Speech Paralinguistics Analysis")
            prosodic_features = speech_analyzer.extract_prosodic_features(audio_data, sample_rate)
            fluency_features = speech_analyzer.analyze_fluency_patterns(audio_data, sample_rate)
            
            # Combine speech features
            combined_speech_features = {**prosodic_features, **fluency_features}
            
            # Display speech analysis results
            speech_metrics = pd.DataFrame([
                {"Metric": "F0 Mean (Hz)", "Value": f"{prosodic_features.get('f0_mean', 0):.1f}"},
                {"Metric": "Speech Rate", "Value": f"{fluency_features.get('speech_rate', 0):.2f}"},
                {"Metric": "Pause Frequency", "Value": f"{fluency_features.get('pause_frequency', 0):.2f}"},
                {"Metric": "Rhythm Regularity", "Value": f"{prosodic_features.get('rhythm_regularity', 0):.3f}"},
            ])
            st.dataframe(speech_metrics, use_container_width=True)
            
            # Get speech prediction
            speech_prediction = neuropsych_classifier.predict_from_speech_features(combined_speech_features)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Speech Analysis Result", speech_prediction.get('predicted_class', 'Unknown'))
            with col2:
                st.metric("Confidence", f"{speech_prediction.get('confidence', 0):.1%}")
            
            # Multimodal prediction - Final Diagnosis
            if 'error' not in facial_features:
                st.subheader("🎯 FINAL PREDICTION RESULTS")
                multimodal_prediction = neuropsych_classifier.predict_multimodal(facial_features, combined_speech_features)
                
                # Show all possible labels and their predictions
                all_probs = multimodal_prediction.get('all_probabilities', {})
                predicted_disease = multimodal_prediction.get('predicted_class', 'Unknown')
                confidence_level = multimodal_prediction.get('confidence', 0)
                
                # Display prediction results for all 4 target labels
                st.markdown("### 📊 Label Predictions:")
                
                # Create prediction results table
                prediction_results = []
                for disease in ['Healthy', 'Depression', 'Parkinson\'s Disease', 'Hypothyroidism']:
                    probability = all_probs.get(disease, 0)
                    is_predicted = (disease == predicted_disease)
                    prediction_results.append({
                        'Label': disease,
                        'Probability': f"{probability:.1%}",
                        'Predicted': "✅ YES" if is_predicted else "❌ No",
                        'Status': "SELECTED" if is_predicted else ""
                    })
                
                pred_df = pd.DataFrame(prediction_results)
                st.dataframe(pred_df, use_container_width=True)
                
                # Highlight the final prediction prominently
                st.markdown("### 🎯 FINAL PREDICTION:")
                if predicted_disease != 'Healthy':
                    st.error(f"🚨 **PREDICTED LABEL: {predicted_disease.upper()}**")
                    st.warning(f"🔍 **CONFIDENCE: {confidence_level:.1%}**")
                    
                    # Add severity assessment
                    if confidence_level > 0.8:
                        st.error("⚠️ **HIGH CONFIDENCE DETECTION - IMMEDIATE MEDICAL ATTENTION RECOMMENDED**")
                    elif confidence_level > 0.6:
                        st.warning("⚠️ **MODERATE CONFIDENCE - MEDICAL CONSULTATION ADVISED**")
                    else:
                        st.info("⚠️ **LOW CONFIDENCE - MONITORING RECOMMENDED**")
                else:
                    st.success(f"✅ **PREDICTED LABEL: {predicted_disease.upper()}**")
                    st.info(f"🔍 **CONFIDENCE: {confidence_level:.1%}**")
                
                # Display final results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Final Diagnosis",
                        multimodal_prediction.get('predicted_class', 'Unknown'),
                        help="Based on combined facial and speech analysis"
                    )
                
                with col2:
                    st.metric(
                        "Overall Confidence", 
                        f"{multimodal_prediction.get('confidence', 0):.1%}",
                        help="Confidence in multimodal prediction"
                    )
                
                with col3:
                    # Calculate improvement over single modality
                    max_single = max(facial_prediction.get('confidence', 0), speech_prediction.get('confidence', 0))
                    improvement = multimodal_prediction.get('confidence', 0) - max_single
                    st.metric(
                        "Multimodal Gain",
                        f"{improvement:+.1%}",
                        help="Improvement over best single modality"
                    )
                
                # Detailed probability breakdown
                st.subheader("📊 Detailed Analysis Results")
                
                # Create comparison chart
                all_probs = multimodal_prediction.get('all_probabilities', {})
                if all_probs:
                    prob_df = pd.DataFrame([
                        {
                            'Condition': condition,
                            'Facial': facial_prediction.get('all_probabilities', {}).get(condition, 0),
                            'Speech': speech_prediction.get('all_probabilities', {}).get(condition, 0),
                            'Multimodal': prob
                        }
                        for condition, prob in all_probs.items()
                    ])
                    
                    # Display as heatmap-style table
                    st.dataframe(
                        prob_df.style.format({
                            'Facial': '{:.1%}',
                            'Speech': '{:.1%}',
                            'Multimodal': '{:.1%}'
                        }).background_gradient(subset=['Facial', 'Speech', 'Multimodal']),
                        use_container_width=True
                    )
                    
                    # Generate clinical recommendations
                    st.subheader("💡 Clinical Recommendations")
                    predicted_condition = multimodal_prediction.get('predicted_class', '').lower()
                    confidence = multimodal_prediction.get('confidence', 0)
                    
                    recommendations = generate_clinical_recommendations(predicted_condition, confidence, language)
                    for rec in recommendations:
                        st.info(rec)
        
        else:
            if not frames:
                st.error("Could not extract frames from video for facial analysis.")
            if audio_data is None:
                st.error("Could not extract audio from video for speech analysis.")

def process_neuropsychiatric_video(video_path: str, video_name: str, language: str):
    """Process video for neuropsychiatric analysis with manual button."""
    st.success(f"Processing {video_name} for neuropsychiatric analysis...")
    
    # Video information
    video_info = video_processor.get_video_info(video_path)
    if video_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{video_info.get('duration', 0):.1f}s")
        with col2:
            st.metric("Language", language)
        with col3:
            st.metric("Analysis Type", "Multimodal")
    
    if st.button("🧠 Perform Neuropsychiatric Analysis", type="primary", key="neuro_video_analyze"):
        with st.spinner("Performing comprehensive neuropsychiatric analysis..."):
            # Extract frames for facial analysis
            frames = video_processor.extract_frames(video_path, num_frames=15, method='uniform')
            
            # Extract audio for speech analysis
            audio_data, sample_rate = video_processor.extract_audio_from_video(video_path)
            
            if frames and audio_data is not None:
                # Facial micro-expression analysis
                st.subheader("👤 Facial Micro-expression Analysis")
                facial_features = facial_analyzer.analyze_micro_expressions(frames)
                
                if 'error' not in facial_features:
                    # Display facial analysis results
                    facial_metrics = pd.DataFrame([
                        {"Metric": "Face Detection Rate", "Value": f"{facial_features.get('face_detection_rate', 0):.1%}"},
                        {"Metric": "Frames Analyzed", "Value": facial_features.get('total_frames_analyzed', 0)},
                        {"Metric": "Facial Symmetry (avg)", "Value": f"{facial_features.get('face_symmetry_mean', 0):.3f}"},
                        {"Metric": "Expression Variance", "Value": f"{facial_features.get('face_symmetry_std', 0):.3f}"},
                    ])
                    st.dataframe(facial_metrics, use_container_width=True)
                    
                    # Get facial prediction
                    facial_prediction = neuropsych_classifier.predict_from_facial_features(facial_features)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Facial Analysis Result", facial_prediction.get('predicted_class', 'Unknown'))
                    with col2:
                        st.metric("Confidence", f"{facial_prediction.get('confidence', 0):.1%}")
                
                # Speech paralinguistics analysis
                st.subheader("🗣️ Speech Paralinguistics Analysis")
                prosodic_features = speech_analyzer.extract_prosodic_features(audio_data, sample_rate)
                fluency_features = speech_analyzer.analyze_fluency_patterns(audio_data, sample_rate)
                
                # Combine speech features
                combined_speech_features = {**prosodic_features, **fluency_features}
                
                # Display speech analysis results
                speech_metrics = pd.DataFrame([
                    {"Metric": "F0 Mean (Hz)", "Value": f"{prosodic_features.get('f0_mean', 0):.1f}"},
                    {"Metric": "Speech Rate", "Value": f"{fluency_features.get('speech_rate', 0):.2f}"},
                    {"Metric": "Pause Frequency", "Value": f"{fluency_features.get('pause_frequency', 0):.2f}"},
                    {"Metric": "Rhythm Regularity", "Value": f"{prosodic_features.get('rhythm_regularity', 0):.3f}"},
                ])
                st.dataframe(speech_metrics, use_container_width=True)
                
                # Get speech prediction
                speech_prediction = neuropsych_classifier.predict_from_speech_features(combined_speech_features)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Speech Analysis Result", speech_prediction.get('predicted_class', 'Unknown'))
                with col2:
                    st.metric("Confidence", f"{speech_prediction.get('confidence', 0):.1%}")
                
                # Multimodal prediction
                if 'error' not in facial_features:
                    st.subheader("🔄 Multimodal Integration")
                    multimodal_prediction = neuropsych_classifier.predict_multimodal(facial_features, combined_speech_features)
                    
                    # Display final results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Final Diagnosis",
                            multimodal_prediction.get('predicted_class', 'Unknown'),
                            help="Based on combined facial and speech analysis"
                        )
                    
                    with col2:
                        st.metric(
                            "Overall Confidence", 
                            f"{multimodal_prediction.get('confidence', 0):.1%}",
                            help="Confidence in multimodal prediction"
                        )
                    
                    with col3:
                        # Calculate improvement over single modality
                        max_single = max(facial_prediction.get('confidence', 0), speech_prediction.get('confidence', 0))
                        improvement = multimodal_prediction.get('confidence', 0) - max_single
                        st.metric(
                            "Multimodal Gain",
                            f"{improvement:+.1%}",
                            help="Improvement over best single modality"
                        )
                    
                    # Detailed probability breakdown
                    st.subheader("📊 Detailed Analysis Results")
                    
                    # Create comparison chart
                    all_probs = multimodal_prediction.get('all_probabilities', {})
                    if all_probs:
                        prob_df = pd.DataFrame([
                            {
                                'Condition': condition,
                                'Facial': facial_prediction.get('all_probabilities', {}).get(condition, 0),
                                'Speech': speech_prediction.get('all_probabilities', {}).get(condition, 0),
                                'Multimodal': prob
                            }
                            for condition, prob in all_probs.items()
                        ])
                        
                        # Display as heatmap-style table
                        st.dataframe(
                            prob_df.style.format({
                                'Facial': '{:.1%}',
                                'Speech': '{:.1%}',
                                'Multimodal': '{:.1%}'
                            }).background_gradient(subset=['Facial', 'Speech', 'Multimodal']),
                            use_container_width=True
                        )
                        
                        # Generate clinical recommendations
                        st.subheader("💡 Clinical Recommendations")
                        predicted_condition = multimodal_prediction.get('predicted_class', '').lower()
                        confidence = multimodal_prediction.get('confidence', 0)
                        
                        recommendations = generate_clinical_recommendations(predicted_condition, confidence, language)
                        for rec in recommendations:
                            st.info(rec)
            
            else:
                if not frames:
                    st.error("Could not extract frames from video for facial analysis.")
                if audio_data is None:
                    st.error("Could not extract audio from video for speech analysis.")

def process_neuropsychiatric_audio(uploaded_audio, language: str):
    """Process audio file for speech paralinguistics analysis."""
    st.success(f"Processing audio for speech analysis in {language}...")
    
    # Load and process audio
    audio_data, sample_rate = audio_processor.load_audio(uploaded_audio)
    
    if audio_data is not None and sample_rate is not None:
        if st.button("🗣️ Analyze Speech Patterns", type="primary", key="neuro_audio_analyze"):
            with st.spinner("Analyzing speech paralinguistics..."):
                # Extract speech features
                prosodic_features = speech_analyzer.extract_prosodic_features(audio_data, sample_rate)
                fluency_features = speech_analyzer.analyze_fluency_patterns(audio_data, sample_rate)
                combined_features = {**prosodic_features, **fluency_features}
                
                # Get prediction
                prediction = neuropsych_classifier.predict_from_speech_features(combined_features)
                
                # Display results
                display_neuropsychiatric_results(prediction, "Speech Analysis", combined_features)

def process_neuropsychiatric_image(uploaded_image, language: str):
    """Process image for facial micro-expression analysis."""
    st.success(f"Processing image for facial analysis...")
    
    # Load image
    image = image_processor.load_image(uploaded_image)
    
    if image is not None:
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("👤 Analyze Facial Expressions", type="primary", key="neuro_image_analyze"):
            with st.spinner("Analyzing facial micro-expressions..."):
                # Convert PIL image to numpy array
                image_array = np.array(image)
                
                # Analyze single frame (convert to list for compatibility)
                facial_features = facial_analyzer.analyze_micro_expressions([image_array])
                
                if 'error' not in facial_features:
                    # Get prediction
                    prediction = neuropsych_classifier.predict_from_facial_features(facial_features)
                    
                    # Display results
                    display_neuropsychiatric_results(prediction, "Facial Analysis", facial_features)
                else:
                    st.error(facial_features['error'])

def display_neuropsychiatric_results(prediction: Dict, analysis_type: str, features: Dict):
    """Display neuropsychiatric analysis results."""
    st.subheader(f"🧠 {analysis_type} Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Condition", prediction.get('predicted_class', 'Unknown'))
    
    with col2:
        confidence = prediction.get('confidence', 0)
        confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        st.metric("Confidence", f"{confidence:.1%}", delta=confidence_level)
    
    with col3:
        st.metric("Analysis Modality", prediction.get('modality', 'Unknown').title())
    
    # Probability breakdown
    all_probs = prediction.get('all_probabilities', {})
    if all_probs:
        st.subheader("📊 Condition Probabilities")
        prob_df = pd.DataFrame(list(all_probs.items()), columns=['Condition', 'Probability'])
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.1%}")
        prob_df = prob_df.sort_values('Probability', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
        st.dataframe(prob_df, use_container_width=True)

def generate_clinical_recommendations(condition: str, confidence: float, language: str) -> List[str]:
    """Generate clinical recommendations based on predicted condition."""
    recommendations = []
    
    # Always include disclaimer first
    base_rec = "🏥 These are AI-generated suggestions. Always consult healthcare professionals for medical decisions."
    recommendations.append(base_rec)
    
    if confidence < 0.4:
        recommendations.append("⚠️ Low confidence prediction. Additional clinical assessment recommended.")
        recommendations.append("🔄 Consider retesting with higher quality video/audio data.")
        return recommendations
    
    # Add urgency-based recommendations
    if confidence > 0.8:
        recommendations.append("🚨 HIGH CONFIDENCE DETECTION - Schedule immediate medical consultation.")
    elif confidence > 0.6:
        recommendations.append("⚠️ MODERATE CONFIDENCE - Schedule medical evaluation within 2 weeks.")
    else:
        recommendations.append("📅 MONITORING RECOMMENDED - Schedule routine check-up.")
    
    if 'depression' in condition:
        recommendations.extend([
            "🧘 Psychological counseling or therapy sessions recommended",
            "💊 Psychiatric evaluation for potential medication",
            "🏃‍♂️ Regular exercise and social activities beneficial",
            "📞 Crisis support: National suicide prevention lifeline available 24/7"
        ])
    elif 'parkinson' in condition:
        recommendations.extend([
            "🧠 Immediate neurological evaluation by movement disorder specialist",
            "💪 Physical therapy and structured exercise program essential",
            "🎯 Occupational therapy for daily living activities",
            "💊 Dopamine replacement therapy consultation",
            "👥 Patient and family support groups"
        ])
    elif 'hypothyroid' in condition:
        recommendations.extend([
            "🧪 Comprehensive thyroid function tests (TSH, Free T3, Free T4)",
            "💊 Endocrinology consultation for hormone replacement therapy",
            "🥗 Nutritional counseling for iodine and selenium intake",
            "📅 Regular monitoring every 6-8 weeks initially",
            "⚡ Monitor for cardiac and metabolic complications"
        ])
    
    if language.lower() in ['telugu', 'multilingual']:
        recommendations.append("🌐 Telugu language healthcare resources may be available in your area")
    
    return recommendations

def combined_analysis_tab():
    """Handle combined audio and image analysis."""
    st.header("🔄 Combined Analysis")
    st.markdown("Upload both audio and image files for comprehensive medical analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Audio Input")
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            key="combined_audio"
        )
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav')
            st.success(f"Audio file: {uploaded_audio.name}")
    
    with col2:
        st.subheader("🖼️ Image Input")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            key="combined_image"
        )
        
        if uploaded_image is not None:
            image = image_processor.load_image(uploaded_image)
            if image is not None:
                st.image(image, caption="Medical Image", use_container_width=True)
                st.success(f"Image file: {uploaded_image.name}")
    
    # Combined analysis
    if uploaded_audio is not None and uploaded_image is not None:
        if st.button("Perform Combined Medical Analysis", type="primary"):
            with st.spinner("Performing comprehensive medical analysis..."):
                # Process audio
                audio_data, sample_rate = audio_processor.load_audio(uploaded_audio)
                if audio_data is not None and sample_rate is not None:
                    audio_features = audio_processor.extract_features(audio_data, sample_rate)
                    audio_vector = audio_processor.preprocess_for_prediction(audio_features)
                    audio_results = medical_ai.get_audio_prediction(audio_vector)
                else:
                    audio_results = {'predicted_class': 'Error', 'confidence': 0.0}
                
                # Process image
                image = image_processor.load_image(uploaded_image)
                if image is not None:
                    preprocessed_image = image_processor.preprocess_for_prediction(image)
                    image_results = medical_ai.get_image_prediction(preprocessed_image)
                else:
                    image_results = {'predicted_class': 'Error', 'confidence': 0.0}
                
                # Combined analysis
                combined_results = medical_ai.get_combined_analysis(audio_results, image_results)
                
                # Simulate processing time
                time.sleep(3)
            
            # Display combined results
            st.subheader("🔍 Comprehensive Medical Analysis Results")
            
            # Create three columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Audio Prediction",
                    audio_results.get('predicted_class', 'Unknown'),
                    f"{audio_results.get('confidence', 0):.1%} confidence"
                )
            
            with col2:
                st.metric(
                    "Image Prediction", 
                    image_results.get('predicted_class', 'Unknown'),
                    f"{image_results.get('confidence', 0):.1%} confidence"
                )
            
            with col3:
                st.metric(
                    "Combined Confidence",
                    f"{combined_results.get('combined_confidence', 0):.1%}",
                    "Overall assessment"
                )
            
            # Recommendations
            if combined_results.get('recommendation'):
                st.subheader("💡 Medical Recommendations")
                st.info(combined_results['recommendation'])
            
            # Detailed probability breakdown
            st.subheader("📊 Detailed Analysis")
            
            # Create probability charts
            audio_probs = audio_results.get('all_probabilities', {})
            image_probs = image_results.get('all_probabilities', {})
            
            if audio_probs:
                st.write("**Audio Analysis Probabilities:**")
                audio_df = pd.DataFrame(list(audio_probs.items()), columns=['Condition', 'Probability'])
                audio_df['Probability'] = audio_df['Probability'].apply(lambda x: f"{x:.1%}")
                st.dataframe(audio_df, use_container_width=True)
            
            if image_probs:
                st.write("**Image Analysis Probabilities:**")
                image_df = pd.DataFrame(list(image_probs.items()), columns=['Condition', 'Probability'])
                image_df['Probability'] = image_df['Probability'].apply(lambda x: f"{x:.1%}")
                st.dataframe(image_df, use_container_width=True)

def display_prediction_results(results: dict, analysis_type: str):
    """Display prediction results in a formatted way."""
    if not results or results.get('predicted_class') == 'Error':
        st.error("Unable to generate prediction. Please try again with a different file.")
        return
    
    # Main prediction result
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Condition",
            results.get('predicted_class', 'Unknown')
        )
    
    with col2:
        confidence = results.get('confidence', 0)
        st.metric(
            "Confidence Score",
            f"{confidence:.1%}",
            delta=f"{'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'} confidence"
        )
    
    with col3:
        st.metric(
            "Analysis Type",
            analysis_type.title()
        )
    
    # Probability breakdown
    all_probs = results.get('all_probabilities', {})
    if all_probs:
        st.subheader("📊 All Condition Probabilities")
        
        # Create a horizontal bar chart
        prob_df = pd.DataFrame(list(all_probs.items()), columns=['Condition', 'Probability'])
        prob_df = prob_df.sort_values('Probability', ascending=True)
        
        fig = px.bar(
            prob_df, 
            x='Probability', 
            y='Condition',
            orientation='h',
            title=f"Probability Distribution - {analysis_type.title()} Analysis",
            labels={'Probability': 'Confidence Score', 'Condition': 'Medical Condition'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show as table as well
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.1%}")
        st.dataframe(prob_df.sort_values('Probability', ascending=False), use_container_width=True)
    
    # Confidence indicator
    confidence = results.get('confidence', 0)
    if confidence > 0.8:
        st.success("🟢 High confidence prediction - Results are likely reliable")
    elif confidence > 0.6:
        st.warning("🟡 Medium confidence prediction - Consider additional testing")
    else:
        st.error("🔴 Low confidence prediction - Recommend professional medical evaluation")
    
    # Medical disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Medical Disclaimer**: These results are generated by an AI model for demonstration purposes only. 
    This application should not be used for actual medical diagnosis or treatment decisions. 
    Always consult qualified healthcare professionals for medical concerns.
    """)

if __name__ == "__main__":
    main()
