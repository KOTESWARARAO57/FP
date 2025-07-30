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
from utils.speech_to_text_analyzer import SpeechToTextAnalyzer
from utils.whisper_processor import WhisperProcessor
from utils.deepface_processor import DeepFaceProcessor
from utils.translator import MultilingualTranslator
from utils.text_to_speech import TextToSpeechProcessor
from utils.fusion_model import FusionDeepLearningModel
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

@st.cache_resource
def load_speech_to_text_system():
    """Load and cache the speech-to-text analysis system."""
    return SpeechToTextAnalyzer()

@st.cache_resource
def load_advanced_multimodal_system():
    """Load and cache the advanced multimodal AI system."""
    return (WhisperProcessor(), 
            DeepFaceProcessor(), 
            MultilingualTranslator(), 
            TextToSpeechProcessor(),
            FusionDeepLearningModel())

# Load systems
medical_ai = load_medical_ai_system()
audio_processor, image_processor, video_processor = load_processors()
facial_analyzer, speech_analyzer, neuropsych_classifier = load_neuropsychiatric_system()
speech_to_text_analyzer = load_speech_to_text_system()
whisper_processor, deepface_processor, translator, tts_processor, fusion_model = load_advanced_multimodal_system()

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
        ["Audio Analysis", "Image Analysis", "Video Analysis", "Speech-to-Text Analysis", "Advanced Multimodal Analysis", "Neuropsychiatric Analysis", "Combined Analysis"]
    )
    
    # Create tabs based on selection
    if analysis_type == "Audio Analysis":
        audio_analysis_tab()
    elif analysis_type == "Image Analysis":
        image_analysis_tab()
    elif analysis_type == "Video Analysis":
        video_analysis_tab()
    elif analysis_type == "Speech-to-Text Analysis":
        speech_to_text_analysis_tab()
    elif analysis_type == "Advanced Multimodal Analysis":
        advanced_multimodal_analysis_tab()
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

def speech_to_text_analysis_tab():
    """Handle speech-to-text conversion and text-based medical analysis."""
    st.header("🗣️ Speech-to-Text Medical Analysis")
    st.markdown("""
    Convert speech to text and analyze linguistic patterns for medical diagnosis:
    - **Speech Recognition**: Convert audio to text using advanced speech recognition
    - **Text Analysis**: Analyze linguistic patterns, sentiment, and medical keywords
    - **Medical Prediction**: Predict conditions based on speech content and patterns
    - **Target Labels**: Healthy, Depression, Parkinson's Disease, Hypothyroidism
    """)
    
    # Language selection for speech recognition
    st.subheader("🌐 Language Selection")
    language_options = {
        "English": "en-US",
        "Telugu": "te-IN",
        "English (UK)": "en-GB",
        "English (Australia)": "en-AU"
    }
    selected_language = st.selectbox("Select speech language:", list(language_options.keys()))
    language_code = language_options[selected_language]
    
    # Input options
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.subheader("📁 Upload Audio File")
        uploaded_audio = st.file_uploader(
            "Choose an audio file for speech-to-text analysis",
            type=['wav', 'mp3', 'm4a', 'flac'],
            key="stt_audio_upload"
        )
        
        if uploaded_audio is not None:
            process_speech_to_text_file(uploaded_audio, language_code, selected_language)
    
    with input_col2:
        st.subheader("📝 Direct Text Analysis")
        st.markdown("Skip speech recognition and directly analyze text:")
        
        text_input = st.text_area(
            "Enter text for medical analysis:",
            height=150,
            placeholder="Type or paste text here to analyze for medical conditions..."
        )
        
        if st.button("🔍 Analyze Text", type="primary") and text_input.strip():
            analyze_text_directly(text_input, selected_language)

def process_speech_to_text_file(uploaded_audio, language_code: str, language_name: str):
    """Process uploaded audio file for speech-to-text analysis."""
    st.success(f"Processing audio for speech-to-text analysis in {language_name}...")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_audio.getvalue())
        temp_audio_path = tmp_file.name
    
    try:
        # Show audio player
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("🎤 Convert Speech to Text & Analyze", type="primary", key="stt_analyze"):
            with st.spinner("Converting speech to text and analyzing..."):
                # Perform speech-to-text analysis
                result = speech_to_text_analyzer.analyze_speech_to_text(temp_audio_path, language_code)
                
                if result['success']:
                    # Display transcription results
                    st.subheader("📝 Speech Transcription Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Recognition Method", result['speech_to_text_method'])
                    with col2:
                        st.metric("Speech Quality", result['speech_confidence'])
                    with col3:
                        st.metric("Language", language_name)
                    
                    # Show transcribed text
                    st.subheader("📋 Transcribed Text")
                    st.text_area("Extracted Text:", value=result['transcribed_text'], height=100, disabled=True)
                    
                    # Display text analysis results
                    display_text_analysis_results(result, result['transcribed_text'])
                    
                else:
                    st.error(f"Speech-to-text conversion failed: {result['error']}")
                    st.info("Try:")
                    st.info("• Using clearer audio with less background noise")
                    st.info("• Ensuring the audio contains speech in the selected language")
                    st.info("• Using WAV format for better recognition accuracy")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

def analyze_text_directly(text: str, language_name: str):
    """Analyze text directly without speech recognition."""
    st.success(f"Analyzing text for medical indicators...")
    
    with st.spinner("Performing text analysis..."):
        # Get prediction from text
        result = speech_to_text_analyzer.predict_from_text(text)
        
        if 'error' not in result:
            # Create result dictionary similar to speech-to-text format
            analysis_result = {
                'success': True,
                'transcribed_text': text,
                'speech_to_text_method': 'Direct Text Input',
                'speech_confidence': 'N/A',
                'predicted_class': result['predicted_class'],
                'prediction_confidence': result['confidence'],
                'all_probabilities': result.get('all_probabilities', {}),
                'text_features': result.get('text_features', {}),
                'language': language_name
            }
            
            display_text_analysis_results(analysis_result, text)
        else:
            st.error(f"Text analysis failed: {result['error']}")

def display_text_analysis_results(result: Dict, text: str):
    """Display comprehensive text analysis results."""
    
    # Text features analysis
    st.subheader("📊 Text Analysis Features")
    text_features = result.get('text_features', {})
    
    if text_features:
        # Create features dataframe
        feature_metrics = [
            {"Feature": "Word Count", "Value": text_features.get('word_count', 0)},
            {"Feature": "Sentence Count", "Value": text_features.get('sentence_count', 0)},
            {"Feature": "Average Word Length", "Value": f"{text_features.get('avg_word_length', 0):.1f}"},
            {"Feature": "Sentiment Polarity", "Value": f"{text_features.get('sentiment_polarity', 0):.3f}"},
            {"Feature": "Medical Keywords", "Value": text_features.get('medical_keywords_count', 0)},
            {"Feature": "Coherence Score", "Value": f"{text_features.get('coherence_score', 0):.3f}"}
        ]
        
        features_df = pd.DataFrame(feature_metrics)
        st.dataframe(features_df, use_container_width=True)
        
        # Disease-specific indicators
        st.subheader("🔍 Condition-Specific Indicators")
        indicators_data = [
            {"Condition": "Depression", "Keywords Found": text_features.get('depression_indicators', 0)},
            {"Condition": "Parkinson's Disease", "Keywords Found": text_features.get('parkinson_indicators', 0)},
            {"Condition": "Hypothyroidism", "Keywords Found": text_features.get('hypothyroid_indicators', 0)},
            {"Condition": "Healthy", "Keywords Found": text_features.get('healthy_indicators', 0)}
        ]
        
        indicators_df = pd.DataFrame(indicators_data)
        st.dataframe(indicators_df, use_container_width=True)
    
    # Prediction results
    st.subheader("🎯 MEDICAL PREDICTION FROM TEXT")
    
    predicted_condition = result['predicted_class']
    confidence = result['prediction_confidence']
    all_probs = result.get('all_probabilities', {})
    
    # Show prediction table for all labels
    st.markdown("### 📊 Label Predictions from Text Analysis:")
    
    prediction_results = []
    target_labels = ['Healthy', 'Depression', 'Parkinson\'s Disease', 'Hypothyroidism']
    
    for label in target_labels:
        probability = all_probs.get(label, 0)
        is_predicted = (label == predicted_condition)
        prediction_results.append({
            'Label': label,
            'Probability': f"{probability:.1%}",
            'Predicted': "✅ YES" if is_predicted else "❌ No",
            'Status': "SELECTED" if is_predicted else ""
        })
    
    pred_df = pd.DataFrame(prediction_results)
    st.dataframe(pred_df, use_container_width=True)
    
    # Highlight final prediction
    st.markdown("### 🎯 FINAL TEXT-BASED PREDICTION:")
    
    if predicted_condition != 'Healthy':
        st.error(f"🚨 **TEXT ANALYSIS INDICATES: {predicted_condition.upper()}**")
        st.warning(f"🔍 **CONFIDENCE: {confidence:.1%}**")
        
        # Confidence-based recommendations
        if confidence > 0.8:
            st.error("⚠️ **HIGH CONFIDENCE - Consider medical consultation based on speech patterns**")
        elif confidence > 0.6:
            st.warning("⚠️ **MODERATE CONFIDENCE - Monitor symptoms and consider evaluation**")
        else:
            st.info("⚠️ **LOW CONFIDENCE - Text patterns suggest possible indicators**")
    else:
        st.success(f"✅ **TEXT ANALYSIS INDICATES: {predicted_condition.upper()}**")
        st.info(f"🔍 **CONFIDENCE: {confidence:.1%}**")
    
    # Additional insights
    st.subheader("💡 Text Analysis Insights")
    
    insights = []
    
    if text_features:
        sentiment = text_features.get('sentiment_polarity', 0)
        if sentiment < -0.3:
            insights.append("🔍 Text shows negative sentiment patterns")
        elif sentiment > 0.3:
            insights.append("🔍 Text shows positive sentiment patterns")
        else:
            insights.append("🔍 Text shows neutral sentiment")
        
        coherence = text_features.get('coherence_score', 0)
        if coherence < 0.4:
            insights.append("🔍 Text structure shows some disorganization")
        elif coherence > 0.7:
            insights.append("🔍 Text structure is well-organized and coherent")
        
        if text_features.get('repetitive_words', 0) > 3:
            insights.append("🔍 Text contains repetitive word patterns")
        
        if text_features.get('incomplete_sentences', 0) > 0:
            insights.append("🔍 Text contains incomplete sentence structures")
    
    if not insights:
        insights.append("🔍 Text analysis completed - no significant linguistic anomalies detected")
    
    for insight in insights:
        st.info(insight)
    
    # Clinical note
    st.warning("⚠️ This analysis is based on linguistic patterns and should not replace professional medical assessment.")

def advanced_multimodal_analysis_tab():
    """Handle advanced multimodal AI analysis with Whisper, DeepFace, and Fusion models."""
    st.header("🤖 Advanced Multimodal AI Analysis")
    st.markdown("""
    **Comprehensive multilingual, multimodal AI system for neuropsychiatric disease detection:**
    - **🎤 Whisper Speech-to-Text**: Advanced speech recognition with multilingual support
    - **😊 DeepFace Emotion Analysis**: Facial micro-expression detection and emotion recognition
    - **🌐 Multilingual Translation**: Telugu-English translation with medical context
    - **🔊 Text-to-Speech Output**: Diagnosis results spoken in selected language
    - **🧠 Fusion Deep Learning**: CNN+LSTM model combining facial and audio features
    
    **Target Diseases:** Healthy, Depression, Parkinson's Disease, Hypothyroidism
    """)
    
    # Language selection
    st.subheader("🌐 Language Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        input_language = st.selectbox(
            "Input Language (for speech recognition):",
            ["English", "Telugu", "Hindi", "Tamil"],
            help="Language of the video/audio content"
        )
        
    with col2:
        output_language = st.selectbox(
            "Output Language (for results):",
            ["English", "Telugu", "Hindi", "Tamil"],
            help="Language for diagnosis results and speech output"
        )
    
    # Map language names to codes
    lang_codes = {
        "English": "en",
        "Telugu": "te", 
        "Hindi": "hi",
        "Tamil": "ta"
    }
    
    input_lang_code = lang_codes.get(input_language, "en")
    output_lang_code = lang_codes.get(output_language, "en")
    
    # Video upload and processing
    st.subheader("📹 Video Upload for Multimodal Analysis")
    uploaded_video = st.file_uploader(
        "Upload video for comprehensive multimodal AI analysis",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video containing speech and facial expressions"
    )
    
    if uploaded_video is not None:
        process_advanced_multimodal_video(uploaded_video, input_lang_code, output_lang_code, input_language, output_language)

def process_advanced_multimodal_video(uploaded_video, input_lang_code: str, output_lang_code: str, input_language: str, output_language: str):
    """Process video using advanced multimodal AI pipeline."""
    st.success(f"Processing video with advanced AI pipeline...")
    st.info(f"Input: {input_language} → Output: {output_language}")
    
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(uploaded_video.getvalue())
        video_path = tmp_video.name
    
    try:
        # Show video preview
        st.video(uploaded_video)
        
        if st.button("🚀 Start Advanced Multimodal Analysis", type="primary", key="advanced_analysis"):
            with st.spinner("Running comprehensive AI analysis..."):
                
                # Step 1: Extract audio from video
                st.subheader("🎵 Step 1: Audio Extraction")
                audio_info = video_processor.extract_audio_from_video(video_path)
                
                if audio_info and 'audio_path' in audio_info:
                    audio_path = audio_info['audio_path']
                    st.success("✅ Audio extracted successfully")
                    
                    # Step 2: Whisper speech-to-text
                    st.subheader("🎤 Step 2: Whisper Speech Recognition")
                    whisper_result = whisper_processor.transcribe_audio(audio_path, input_lang_code)
                    
                    if whisper_result['success']:
                        st.success(f"✅ Speech transcribed using Whisper ({whisper_result['model_used']})")
                        
                        # Display transcription
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Detected Language", whisper_result['language_name'])
                        with col2:
                            st.metric("Confidence", f"{whisper_result['confidence']:.1%}")
                        with col3:
                            st.metric("Duration", f"{whisper_result['duration']:.1f}s")
                        
                        st.text_area("Transcribed Text:", value=whisper_result['text'], height=100, disabled=True)
                        
                        # Step 3: Extract speech features
                        st.subheader("📊 Step 3: Advanced Speech Feature Extraction")
                        speech_features = whisper_processor.extract_speech_features(audio_path, whisper_result)
                        
                        if speech_features:
                            st.success("✅ Speech features extracted (prosody, fluency, lexical)")
                            
                            # Display key speech features
                            feature_cols = st.columns(4)
                            with feature_cols[0]:
                                st.metric("Speech Rate", f"{speech_features.get('speech_rate', 0):.1f} WPM")
                            with feature_cols[1]:
                                st.metric("Pitch Mean", f"{speech_features.get('pitch_mean', 0):.1f} Hz")
                            with feature_cols[2]:
                                st.metric("Lexical Diversity", f"{speech_features.get('lexical_diversity', 0):.3f}")
                            with feature_cols[3]:
                                st.metric("Medical Keywords", speech_features.get('medical_keywords', 0))
                        
                        # Step 4: Extract frames and analyze faces
                        st.subheader("😊 Step 4: DeepFace Facial Analysis")
                        video_info = video_processor.extract_frames(video_path, max_frames=10)
                        
                        if video_info['success'] and video_info['frames']:
                            # Analyze facial emotions in key frames
                            facial_analysis_results = []
                            
                            progress_bar = st.progress(0)
                            for i, frame_info in enumerate(video_info['frames'][:5]):  # Analyze first 5 frames
                                progress_bar.progress((i + 1) / 5)
                                
                                frame_path = frame_info['path']
                                emotion_result = deepface_processor.analyze_facial_emotions(frame_path)
                                
                                if emotion_result['success']:
                                    facial_analysis_results.append(emotion_result)
                            
                            progress_bar.empty()
                            
                            if facial_analysis_results:
                                st.success(f"✅ Facial emotions analyzed in {len(facial_analysis_results)} frames")
                                
                                # Aggregate facial features
                                aggregated_facial_features = aggregate_facial_results(facial_analysis_results)
                                
                                # Display dominant emotions
                                emotion_cols = st.columns(3)
                                with emotion_cols[0]:
                                    st.metric("Dominant Emotion", aggregated_facial_features.get('dominant_emotion', 'Unknown'))
                                with emotion_cols[1]:
                                    st.metric("Avg Confidence", f"{aggregated_facial_features.get('avg_confidence', 0):.1%}")
                                with emotion_cols[2]:
                                    st.metric("Faces Detected", f"{len(facial_analysis_results)}/5")
                                
                                # Step 5: Fusion Deep Learning Prediction
                                st.subheader("🧠 Step 5: Fusion Deep Learning Model")
                                
                                # Get medical-relevant features for fusion
                                facial_medical_features = deepface_processor.get_medical_facial_features(aggregated_facial_features)
                                
                                # Run fusion model
                                fusion_result = fusion_model.predict_multimodal_fusion(facial_medical_features, speech_features)
                                
                                if fusion_result['success']:
                                    st.success("✅ Fusion model prediction completed")
                                    
                                    # Display comprehensive results
                                    display_advanced_multimodal_results(fusion_result, whisper_result, facial_analysis_results, output_lang_code, output_language)
                                
                                else:
                                    st.error(f"Fusion model failed: {fusion_result['error']}")
                            else:
                                st.warning("No faces detected in video frames")
                        else:
                            st.warning("Failed to extract frames from video")
                    else:
                        st.error(f"Speech recognition failed: {whisper_result['error']}")
                else:
                    st.error("Failed to extract audio from video")
    
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.unlink(video_path)

def aggregate_facial_results(facial_results: List[Dict]) -> Dict:
    """Aggregate facial emotion results across multiple frames."""
    try:
        if not facial_results:
            return {}
        
        # Aggregate emotion scores
        all_emotions = {}
        total_confidence = 0.0
        dominant_emotions = []
        
        for result in facial_results:
            emotions = result.get('emotions', {})
            confidence = result.get('confidence', 0.0)
            dominant = result.get('dominant_emotion', 'neutral')
            
            for emotion, score in emotions.items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = []
                all_emotions[emotion].append(score)
            
            total_confidence += confidence
            dominant_emotions.append(dominant)
        
        # Calculate average emotions
        avg_emotions = {}
        for emotion, scores in all_emotions.items():
            avg_emotions[emotion] = sum(scores) / len(scores)
        
        # Find most common dominant emotion
        from collections import Counter
        emotion_counter = Counter(dominant_emotions)
        most_common_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else 'neutral'
        
        return {
            'emotions': avg_emotions,
            'dominant_emotion': most_common_emotion,
            'avg_confidence': total_confidence / len(facial_results),
            'face_detected': True,
            'frames_analyzed': len(facial_results)
        }
        
    except Exception as e:
        st.warning(f"Facial aggregation failed: {str(e)}")
        return {}

def display_advanced_multimodal_results(fusion_result: Dict, whisper_result: Dict, facial_results: List[Dict], output_lang_code: str, output_language: str):
    """Display comprehensive multimodal analysis results."""
    
    # Main prediction results
    st.subheader("🎯 FUSION AI PREDICTION RESULTS")
    
    predicted_condition = fusion_result['predicted_class']
    confidence = fusion_result['confidence']
    all_probs = fusion_result.get('all_probabilities', {})
    
    # Create comprehensive prediction table
    st.markdown("### 📊 Multi-Modal Label Predictions:")
    
    prediction_results = []
    target_labels = ['Healthy', 'Depression', 'Parkinson\'s Disease', 'Hypothyroidism']
    
    for label in target_labels:
        probability = all_probs.get(label, 0)
        is_predicted = (label == predicted_condition)
        prediction_results.append({
            'Label': label,
            'Fusion Probability': f"{probability:.1%}",
            'Facial Only': f"{fusion_result.get('facial_prediction', {}).get('probabilities', {}).get(label, 0):.1%}",
            'Audio Only': f"{fusion_result.get('audio_prediction', {}).get('probabilities', {}).get(label, 0):.1%}",
            'Final Prediction': "✅ SELECTED" if is_predicted else "❌ Not Selected"
        })
    
    pred_df = pd.DataFrame(prediction_results)
    st.dataframe(pred_df, use_container_width=True)
    
    # Highlight final prediction
    st.markdown("### 🧠 FINAL FUSION AI DIAGNOSIS:")
    
    if predicted_condition != 'Healthy':
        st.error(f"🚨 **FUSION AI INDICATES: {predicted_condition.upper()}**")
        st.warning(f"🔍 **CONFIDENCE: {confidence:.1%}**")
        
        # Confidence-based recommendations
        if confidence > 0.8:
            st.error("⚠️ **HIGH CONFIDENCE - PROFESSIONAL MEDICAL EVALUATION STRONGLY RECOMMENDED**")
        elif confidence > 0.6:
            st.warning("⚠️ **MODERATE CONFIDENCE - MEDICAL CONSULTATION ADVISED**")
        else:
            st.info("⚠️ **LOW CONFIDENCE - MONITORING AND FOLLOW-UP SUGGESTED**")
    else:
        st.success(f"✅ **FUSION AI INDICATES: {predicted_condition.upper()}**")
        st.info(f"🔍 **CONFIDENCE: {confidence:.1%}**")
    
    # Modality analysis
    st.subheader("🔍 Modality Analysis")
    modality_weights = fusion_result.get('modality_weights', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Facial Contribution", f"{modality_weights.get('facial_weight', 0):.1%}")
        st.metric("Facial Quality Score", f"{modality_weights.get('facial_quality', 0):.2f}")
    
    with col2:
        st.metric("Audio Contribution", f"{modality_weights.get('audio_weight', 0):.1%}")
        st.metric("Audio Quality Score", f"{modality_weights.get('audio_quality', 0):.2f}")
    
    # Translation and TTS
    st.subheader("🌐 Multilingual Output")
    
    # Create diagnosis text
    diagnosis_text = f"Medical analysis completed. The fusion AI system indicates {predicted_condition} with {confidence:.0%} confidence."
    
    if output_lang_code != 'en':
        # Translate diagnosis
        translation_result = translator.create_bilingual_output(diagnosis_text, output_lang_code)
        
        if translation_result['success']:
            st.text_area("English Diagnosis:", value=translation_result['english_text'], height=60, disabled=True)
            st.text_area(f"{output_language} Translation:", value=translation_result['translated_text'], height=60, disabled=True)
            
            # Text-to-Speech output
            if st.button(f"🔊 Speak Diagnosis in {output_language}", key="tts_diagnosis"):
                tts_result = tts_processor.speak_medical_diagnosis(fusion_result, output_lang_code, save_audio=True)
                
                if tts_result['success'] and tts_result['audio_file']:
                    st.success(f"✅ Diagnosis spoken in {output_language}")
                    
                    # Play audio
                    with open(tts_result['audio_file'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav')
                    
                    # Cleanup audio file
                    tts_processor.cleanup_audio_files([tts_result['audio_file']])
                else:
                    st.error(f"Text-to-speech failed: {tts_result.get('error', 'Unknown error')}")
        else:
            st.warning(f"Translation failed: {translation_result.get('error', 'Unknown error')}")
    else:
        st.text_area("Diagnosis:", value=diagnosis_text, height=60, disabled=True)
        
        # English TTS
        if st.button("🔊 Speak Diagnosis in English", key="tts_english"):
            tts_result = tts_processor.speak_medical_diagnosis(fusion_result, 'en', save_audio=True)
            
            if tts_result['success'] and tts_result['audio_file']:
                st.success("✅ Diagnosis spoken in English")
                
                with open(tts_result['audio_file'], 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                
                tts_processor.cleanup_audio_files([tts_result['audio_file']])
    
    # Technical details
    with st.expander("🔧 Technical Analysis Details"):
        st.markdown("**Whisper Speech Recognition:**")
        st.json({
            'model': whisper_result.get('model_used', 'Unknown'),
            'language_detected': whisper_result.get('language_name', 'Unknown'),
            'transcription_confidence': whisper_result.get('confidence', 0),
            'duration': whisper_result.get('duration', 0)
        })
        
        st.markdown("**DeepFace Facial Analysis:**")
        st.json({
            'frames_analyzed': len(facial_results),
            'analysis_method': facial_results[0].get('analysis_method', 'Unknown') if facial_results else 'None',
            'average_confidence': sum(r.get('confidence', 0) for r in facial_results) / len(facial_results) if facial_results else 0
        })
        
        st.markdown("**Fusion Model Performance:**")
        feature_importance = fusion_result.get('feature_importance', {})
        st.json({
            'fusion_method': fusion_result.get('fusion_method', 'CNN_LSTM_Fusion'),
            'facial_feature_importance': feature_importance.get('facial_importance', 0),
            'audio_feature_importance': feature_importance.get('audio_importance', 0),
            'fusion_layer_importance': feature_importance.get('fusion_importance', 0)
        })
    
    # Medical disclaimer
    st.warning("⚠️ **IMPORTANT MEDICAL DISCLAIMER:** This AI system is for research and demonstration purposes only. Results should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical concerns.")

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
