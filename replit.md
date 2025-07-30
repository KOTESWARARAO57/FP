# Medical AI Diagnostic Assistant

## Overview

This is a Streamlit-based web application that provides AI-powered medical diagnostic assistance through multimodal analysis including audio, image, and video processing. The application now features advanced neuropsychiatric and metabolic disease detection using facial micro-expressions and speech paralinguistics analysis. The system supports both English and Telugu languages and can analyze conditions like depression, Parkinson's disease, hypothyroidism, anxiety disorders, and more. The system emphasizes that it's for research and demonstration purposes only and should not replace professional medical consultation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Layout**: Wide layout with expandable sidebar for navigation
- **Caching**: Utilizes Streamlit's `@st.cache_resource` decorator for efficient resource management
- **Visualization**: Plotly for interactive charts and graphs

### Backend Architecture
- **Modular Design**: Separate utility modules for different processing tasks
- **Object-Oriented**: Each processor and model is encapsulated in its own class
- **Mock Implementation**: Uses synthetic data generation for demonstration purposes

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Entry point and UI orchestration
- **Responsibilities**: Page configuration, system initialization, navigation, and user interaction
- **Features**: Multi-type analysis support (Audio, Image, Combined)

### 2. Audio Processing Module (`utils/audio_processor.py`)
- **Purpose**: Handles audio file processing and feature extraction
- **Technology**: Librosa for audio analysis, NumPy for numerical operations
- **Supported Formats**: WAV, MP3, M4A, FLAC
- **Features**: 
  - Audio validation and loading
  - Feature extraction (RMS energy, zero-crossing rate, spectral centroids, MFCCs)
  - Medical-focused audio analysis

### 3. Image Processing Module (`utils/image_processor.py`)
- **Purpose**: Handles medical image processing and analysis
- **Technology**: PIL for image manipulation, OpenCV for computer vision, NumPy for array operations
- **Supported Formats**: JPG, JPEG, PNG, BMP, TIFF
- **Features**:
  - Image validation and loading
  - Standard preprocessing (224x224 target size)
  - Medical image enhancement and analysis

### 4. Machine Learning Models (`utils/ml_models.py`)
- **Purpose**: Provides AI classification capabilities for basic medical conditions
- **Technology**: Scikit-learn for machine learning algorithms
- **Architecture**: 
  - Separate classifiers for audio and image analysis
  - RandomForestClassifier as the base algorithm
  - StandardScaler for feature normalization
- **Classes**: Supports 6 medical conditions including Normal Breathing, Asthma, Pneumonia, Bronchitis, COPD, and COVID-19 symptoms

### 5. Video Processing Module (`utils/video_processor.py`)
- **Purpose**: Handles video file processing for extracting audio and frames
- **Technology**: OpenCV for video processing, FFmpeg for audio extraction fallback
- **Features**:
  - Video information extraction
  - Frame extraction (uniform sampling and key frame detection)
  - Audio extraction from video files
  - Quality analysis of extracted frames

### 6. Multimodal Neuropsychiatric Analysis (`utils/multimodal_processor.py`)
- **Purpose**: Advanced analysis for neuropsychiatric and metabolic diseases
- **Technology**: OpenCV for facial analysis, Librosa for speech processing
- **Components**:
  - **FacialMicroExpressionAnalyzer**: Detects and analyzes facial micro-expressions
  - **SpeechParalinguisticsAnalyzer**: Analyzes prosody, fluency, and vocal patterns
  - **NeuropsychiatricDiseaseClassifier**: Multimodal classification system
- **Languages**: English and Telugu support
- **Conditions**: Depression, Parkinson's Disease, Hypothyroidism, Anxiety, Bipolar Disorder, Schizophrenia, Dementia

## Data Flow

1. **File Upload**: User uploads audio or image files through Streamlit interface
2. **Validation**: Appropriate processor validates file format and integrity
3. **Loading**: File is loaded and converted to appropriate format (numpy arrays for audio, PIL Images for images)
4. **Preprocessing**: Features are extracted and data is prepared for analysis
5. **Classification**: ML models process the features and generate predictions
6. **Visualization**: Results are displayed through interactive Plotly charts and Streamlit components

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Plotly**: Interactive visualization
- **Scikit-learn**: Machine learning algorithms

### Audio Processing
- **Librosa**: Audio analysis and feature extraction
- **Matplotlib**: Audio visualization

### Image Processing
- **PIL (Pillow)**: Image manipulation
- **OpenCV**: Computer vision operations

## Deployment Strategy

### Local Development
- Streamlit run configuration for local testing
- Modular structure allows for easy testing of individual components

### Production Considerations
- **Caching Strategy**: Uses Streamlit's resource caching for model loading
- **Memory Management**: Efficient handling of audio and image data
- **Error Handling**: Comprehensive error handling throughout the application
- **Scalability**: Modular design allows for easy addition of new analysis types

### Security and Compliance
- **Disclaimer Integration**: Clear messaging about demonstration-only usage
- **Data Privacy**: No data persistence or external transmission mentioned
- **Medical Compliance**: Explicitly states not for actual medical diagnosis

## Technical Decisions

### Mock Data Approach
- **Problem**: Need for medical data for demonstration
- **Solution**: Synthetic data generation for training models
- **Rationale**: Allows demonstration without requiring real medical datasets
- **Pros**: Quick setup, no data privacy concerns, reproducible results
- **Cons**: Not suitable for actual medical use

### Streamlit Framework Choice
- **Problem**: Need for rapid prototyping of ML application
- **Solution**: Streamlit for web interface
- **Rationale**: Fast development, built-in ML support, easy deployment
- **Pros**: Quick development, automatic UI generation, good for demos
- **Cons**: Limited customization compared to full web frameworks

### Modular Architecture
- **Problem**: Need for maintainable and extensible code
- **Solution**: Separate modules for different functionalities
- **Rationale**: Clear separation of concerns, easier testing and maintenance
- **Pros**: Maintainable, testable, extensible
- **Cons**: Slightly more complex initial setup