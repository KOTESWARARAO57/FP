import streamlit as st
import numpy as np
import pandas as pd
from utils.audio_processor import AudioProcessor
from utils.image_processor import ImageProcessor
from utils.video_processor import VideoProcessor
from utils.ml_models import MedicalAISystem
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

# Load systems
medical_ai = load_medical_ai_system()
audio_processor, image_processor, video_processor = load_processors()

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
        ["Audio Analysis", "Image Analysis", "Video Analysis", "Combined Analysis"]
    )
    
    # Create tabs based on selection
    if analysis_type == "Audio Analysis":
        audio_analysis_tab()
    elif analysis_type == "Image Analysis":
        image_analysis_tab()
    elif analysis_type == "Video Analysis":
        video_analysis_tab()
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
