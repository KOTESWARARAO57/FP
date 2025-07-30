import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

class DeepFaceProcessor:
    """Advanced facial emotion recognition and micro-expression analysis using DeepFace."""
    
    def __init__(self):
        self.models_loaded = False
        self.detector_backend = 'opencv'  # Start with opencv, fallback available
        self.emotion_model = 'fer2013'
        self.face_analysis_models = {
            'emotion': True,
            'age': False,  # Enable if needed
            'gender': False,  # Enable if needed
            'race': False   # Disable for privacy
        }
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize DeepFace models with fallback options."""
        try:
            # Try to import DeepFace
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
                st.success("DeepFace library loaded successfully!")
                self.models_loaded = True
            except ImportError:
                st.info("DeepFace not available, using OpenCV-based emotion detection")
                self.models_loaded = False
                self._initialize_opencv_fallback()
                
        except Exception as e:
            st.warning(f"DeepFace initialization failed: {str(e)}, using fallback")
            self.models_loaded = False
            self._initialize_opencv_fallback()
    
    def _initialize_opencv_fallback(self):
        """Initialize OpenCV-based emotion detection as fallback."""
        try:
            # Use OpenCV's built-in face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            st.info("Using OpenCV-based facial analysis as fallback")
        except Exception as e:
            st.error(f"Failed to initialize face detection: {str(e)}")
    
    def analyze_facial_emotions(self, image_path: str) -> Dict:
        """
        Analyze facial emotions using DeepFace or fallback method.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with emotion analysis results
        """
        try:
            if self.models_loaded:
                return self._deepface_emotion_analysis(image_path)
            else:
                return self._opencv_emotion_analysis(image_path)
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Emotion analysis failed: {str(e)}',
                'emotions': {},
                'dominant_emotion': 'unknown',
                'confidence': 0.0
            }
    
    def _deepface_emotion_analysis(self, image_path: str) -> Dict:
        """Perform emotion analysis using DeepFace."""
        try:
            # Analyze emotions using DeepFace
            result = self.DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False  # Continue even if face detection fails
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                result = result[0]  # Take first face if multiple detected
            
            emotions = result.get('emotion', {})
            
            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            return {
                'success': True,
                'emotions': emotions,
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1] / 100.0,  # Convert to 0-1 scale
                'face_detected': True,
                'analysis_method': 'DeepFace',
                'region': result.get('region', {}),
                'all_scores': emotions
            }
            
        except Exception as e:
            st.warning(f"DeepFace analysis failed: {str(e)}, trying fallback")
            return self._opencv_emotion_analysis(image_path)
    
    def _opencv_emotion_analysis(self, image_path: str) -> Dict:
        """Fallback emotion analysis using OpenCV."""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'success': False,
                    'error': 'Could not load image',
                    'emotions': {},
                    'dominant_emotion': 'unknown',
                    'confidence': 0.0
                }
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    'success': False,
                    'error': 'No face detected',
                    'emotions': {},
                    'dominant_emotion': 'unknown',
                    'confidence': 0.0,
                    'face_detected': False
                }
            
            # Use simple heuristics for emotion estimation (for demo purposes)
            # In a real implementation, you would use a trained emotion recognition model
            mock_emotions = self._generate_emotion_estimate(gray, faces[0])
            
            dominant_emotion = max(mock_emotions.items(), key=lambda x: x[1])
            
            return {
                'success': True,
                'emotions': mock_emotions,
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'face_detected': True,
                'analysis_method': 'OpenCV-Heuristic',
                'faces_count': len(faces),
                'all_scores': mock_emotions
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'OpenCV analysis failed: {str(e)}',
                'emotions': {},
                'dominant_emotion': 'unknown',
                'confidence': 0.0
            }
    
    def _generate_emotion_estimate(self, gray_img: np.ndarray, face_rect: Tuple) -> Dict[str, float]:
        """
        Generate emotion estimates using basic image analysis.
        This is a simplified approach for demonstration.
        """
        try:
            x, y, w, h = face_rect
            face_roi = gray_img[y:y+h, x:x+w]
            
            # Basic image statistics for emotion estimation
            mean_intensity = np.mean(face_roi)
            std_intensity = np.std(face_roi)
            
            # Simple heuristic mapping (this would be replaced by a real model)
            emotions = {
                'neutral': 0.4,
                'happy': 0.2,
                'sad': 0.15,
                'angry': 0.1,
                'surprise': 0.08,
                'fear': 0.05,
                'disgust': 0.02
            }
            
            # Adjust based on image characteristics (very basic heuristic)
            if mean_intensity > 120:  # Brighter face might indicate happiness
                emotions['happy'] += 0.15
                emotions['neutral'] -= 0.1
            elif mean_intensity < 80:  # Darker might indicate sadness
                emotions['sad'] += 0.15
                emotions['neutral'] -= 0.1
            
            if std_intensity > 40:  # High contrast might indicate surprise or fear
                emotions['surprise'] += 0.1
                emotions['fear'] += 0.05
                emotions['neutral'] -= 0.1
            
            # Normalize to ensure sum = 1
            total = sum(emotions.values())
            emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception:
            # Return default neutral emotion if analysis fails
            return {
                'neutral': 1.0,
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'surprise': 0.0,
                'fear': 0.0,
                'disgust': 0.0
            }
    
    def extract_facial_micro_expressions(self, video_path: str, frame_interval: int = 30) -> Dict:
        """
        Extract facial micro-expressions from video frames.
        
        Args:
            video_path: Path to video file
            frame_interval: Analyze every Nth frame
            
        Returns:
            Dictionary with micro-expression analysis
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    'success': False,
                    'error': 'Could not open video file',
                    'micro_expressions': [],
                    'emotion_timeline': {}
                }
            
            frame_count = 0
            analyzed_frames = 0
            emotion_timeline = {}
            micro_expressions = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze every frame_interval frames
                if frame_count % frame_interval == 0:
                    # Save frame temporarily
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                        cv2.imwrite(temp_path, frame)
                    
                    try:
                        # Analyze emotions in this frame
                        emotion_result = self.analyze_facial_emotions(temp_path)
                        
                        if emotion_result['success']:
                            timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
                            emotion_timeline[timestamp] = emotion_result['emotions']
                            
                            # Detect potential micro-expressions
                            micro_expr = self._detect_micro_expression(emotion_result)
                            if micro_expr:
                                micro_expr['timestamp'] = timestamp
                                micro_expressions.append(micro_expr)
                            
                            analyzed_frames += 1
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                frame_count += 1
            
            cap.release()
            
            # Analyze emotion patterns
            pattern_analysis = self._analyze_emotion_patterns(emotion_timeline)
            
            return {
                'success': True,
                'total_frames': frame_count,
                'analyzed_frames': analyzed_frames,
                'emotion_timeline': emotion_timeline,
                'micro_expressions': micro_expressions,
                'pattern_analysis': pattern_analysis,
                'video_duration': frame_count / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Video analysis failed: {str(e)}',
                'micro_expressions': [],
                'emotion_timeline': {}
            }
    
    def _detect_micro_expression(self, emotion_result: Dict) -> Optional[Dict]:
        """
        Detect potential micro-expressions from emotion analysis.
        
        Args:
            emotion_result: Result from emotion analysis
            
        Returns:
            Micro-expression data or None
        """
        try:
            emotions = emotion_result.get('emotions', {})
            dominant_emotion = emotion_result.get('dominant_emotion', 'neutral')
            confidence = emotion_result.get('confidence', 0.0)
            
            # Look for indicators of micro-expressions
            # High confidence in non-neutral emotions
            if dominant_emotion != 'neutral' and confidence > 0.7:
                return {
                    'type': 'strong_emotion',
                    'emotion': dominant_emotion,
                    'confidence': confidence,
                    'intensity': 'high' if confidence > 0.85 else 'medium'
                }
            
            # Mixed emotions (multiple emotions with similar scores)
            emotion_values = list(emotions.values())
            if len(emotion_values) >= 2:
                sorted_emotions = sorted(emotion_values, reverse=True)
                if len(sorted_emotions) >= 2 and sorted_emotions[0] - sorted_emotions[1] < 0.15:
                    return {
                        'type': 'mixed_emotion',
                        'primary_emotion': dominant_emotion,
                        'confidence': confidence,
                        'intensity': 'subtle'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _analyze_emotion_patterns(self, emotion_timeline: Dict) -> Dict:
        """
        Analyze emotion patterns over time for medical indicators.
        
        Args:
            emotion_timeline: Timeline of emotions
            
        Returns:
            Pattern analysis results
        """
        try:
            if not emotion_timeline:
                return {'pattern_type': 'insufficient_data'}
            
            # Convert timeline to arrays for analysis
            timestamps = sorted(emotion_timeline.keys())
            emotion_names = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
            
            # Calculate emotion stability and variability
            emotion_stability = {}
            emotion_trends = {}
            
            for emotion in emotion_names:
                values = [emotion_timeline[t].get(emotion, 0) for t in timestamps]
                emotion_stability[emotion] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'decreasing' if len(values) > 1 and values[-1] < values[0] else 'stable'
                }
            
            # Detect medical-relevant patterns
            patterns = []
            
            # Check for depression indicators (sustained sadness, low happiness)
            if (emotion_stability['sad']['mean'] > 0.3 and 
                emotion_stability['happy']['mean'] < 0.2):
                patterns.append({
                    'condition': 'depression_indicators',
                    'evidence': 'sustained_negative_emotion',
                    'confidence': 0.6
                })
            
            # Check for anxiety indicators (high fear/surprise variability)
            if (emotion_stability['fear']['std'] > 0.15 or 
                emotion_stability['surprise']['std'] > 0.15):
                patterns.append({
                    'condition': 'anxiety_indicators',
                    'evidence': 'emotion_variability',
                    'confidence': 0.5
                })
            
            # Check for emotional blunting (low overall emotion intensity)
            total_emotion_intensity = sum(emotion_stability[e]['mean'] for e in emotion_names if e != 'neutral')
            if total_emotion_intensity < 0.4 and emotion_stability['neutral']['mean'] > 0.6:
                patterns.append({
                    'condition': 'emotional_blunting',
                    'evidence': 'reduced_emotional_expression',
                    'confidence': 0.4
                })
            
            return {
                'pattern_type': 'analyzed',
                'emotion_stability': emotion_stability,
                'detected_patterns': patterns,
                'analysis_duration': max(timestamps) - min(timestamps) if timestamps else 0,
                'data_points': len(timestamps)
            }
            
        except Exception as e:
            return {
                'pattern_type': 'analysis_failed',
                'error': str(e)
            }
    
    def get_medical_facial_features(self, emotion_result: Dict, pattern_analysis: Dict = None) -> Dict:
        """
        Extract medical-relevant features from facial emotion analysis.
        
        Args:
            emotion_result: Result from emotion analysis
            pattern_analysis: Optional pattern analysis from video
            
        Returns:
            Medical feature vector
        """
        try:
            features = {}
            
            # Basic emotion features
            emotions = emotion_result.get('emotions', {})
            for emotion, score in emotions.items():
                features[f'emotion_{emotion}'] = float(score)
            
            # Dominant emotion features
            features['dominant_emotion_confidence'] = emotion_result.get('confidence', 0.0)
            features['face_detected'] = 1.0 if emotion_result.get('face_detected', False) else 0.0
            
            # Medical indicators
            features['negative_emotion_dominance'] = float(
                emotions.get('sad', 0) + emotions.get('angry', 0) + emotions.get('fear', 0)
            )
            features['positive_emotion_presence'] = float(emotions.get('happy', 0))
            features['emotional_intensity'] = float(max(emotions.values()) if emotions else 0)
            
            # Pattern features (if available)
            if pattern_analysis and pattern_analysis.get('pattern_type') == 'analyzed':
                detected_patterns = pattern_analysis.get('detected_patterns', [])
                features['depression_pattern_detected'] = 1.0 if any(p['condition'] == 'depression_indicators' for p in detected_patterns) else 0.0
                features['anxiety_pattern_detected'] = 1.0 if any(p['condition'] == 'anxiety_indicators' for p in detected_patterns) else 0.0
                features['emotional_blunting_detected'] = 1.0 if any(p['condition'] == 'emotional_blunting' for p in detected_patterns) else 0.0
            else:
                features['depression_pattern_detected'] = 0.0
                features['anxiety_pattern_detected'] = 0.0
                features['emotional_blunting_detected'] = 0.0
            
            return features
            
        except Exception as e:
            st.warning(f"Feature extraction failed: {str(e)}")
            # Return default features
            emotion_names = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
            features = {f'emotion_{emotion}': 0.0 for emotion in emotion_names}
            features.update({
                'dominant_emotion_confidence': 0.0,
                'face_detected': 0.0,
                'negative_emotion_dominance': 0.0,
                'positive_emotion_presence': 0.0,
                'emotional_intensity': 0.0,
                'depression_pattern_detected': 0.0,
                'anxiety_pattern_detected': 0.0,
                'emotional_blunting_detected': 0.0
            })
            return features