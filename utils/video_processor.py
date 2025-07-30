import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import librosa
import streamlit as st
from typing import Tuple, Optional, List, Dict
import os
import tempfile
from PIL import Image
import io

class VideoProcessor:
    """Handles video file processing to extract audio and frames for medical AI predictions."""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        self.frame_size = (224, 224)  # Standard size for medical image models
        
    def validate_video_file(self, file_path: str) -> bool:
        """Validate if the file is a supported video format."""
        if not os.path.exists(file_path):
            return False
        
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.supported_formats
    
    def get_video_info(self, file_path: str) -> Dict:
        """Extract basic video information."""
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return {}
            
            info = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
            cap.release()
            return info
        except Exception as e:
            st.error(f"Error extracting video info: {str(e)}")
            return {}
    
    def extract_audio_from_video(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Extract audio from video file and return audio data and sample rate."""
        try:
            # Use moviepy to extract audio
            video = VideoFileClip(file_path)
            
            if video.audio is None:
                st.warning("No audio track found in the video file.")
                return None, None
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Load audio using librosa
            audio_data, sample_rate = librosa.load(temp_audio_path, sr=None)
            
            # Clean up
            os.unlink(temp_audio_path)
            video.close()
            
            return audio_data, sample_rate
        except Exception as e:
            st.error(f"Error extracting audio from video: {str(e)}")
            return None, None
    
    def extract_frames(self, file_path: str, num_frames: int = 10, method: str = 'uniform') -> List[np.ndarray]:
        """Extract frames from video file.
        
        Args:
            file_path: Path to video file
            num_frames: Number of frames to extract
            method: 'uniform' for evenly spaced frames, 'random' for random frames
        """
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                st.error("Cannot open video file")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            
            if method == 'uniform':
                # Extract evenly spaced frames
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:  # random
                # Extract random frames
                frame_indices = np.random.choice(total_frames, size=min(num_frames, total_frames), replace=False)
                frame_indices.sort()
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame
                    frame_resized = cv2.resize(frame_rgb, self.frame_size)
                    frames.append(frame_resized)
            
            cap.release()
            return frames
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            return []
    
    def extract_key_frames(self, file_path: str, threshold: float = 30.0) -> List[Tuple[int, np.ndarray]]:
        """Extract key frames based on scene changes.
        
        Args:
            file_path: Path to video file
            threshold: Threshold for scene change detection
        """
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return []
            
            key_frames = []
            prev_frame = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray_frame)
                    diff_score = np.mean(diff)
                    
                    # If difference is above threshold, it's a key frame
                    if diff_score > threshold:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, self.frame_size)
                        key_frames.append((frame_idx, frame_resized))
                
                prev_frame = gray_frame
                frame_idx += 1
            
            cap.release()
            
            # If no key frames found, extract first frame
            if not key_frames:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, self.frame_size)
                    key_frames.append((0, frame_resized))
                cap.release()
            
            return key_frames
        except Exception as e:
            st.error(f"Error extracting key frames: {str(e)}")
            return []
    
    def create_frame_grid(self, frames: List[np.ndarray], grid_size: Tuple[int, int] = (2, 5)) -> plt.Figure:
        """Create a grid visualization of extracted frames."""
        try:
            rows, cols = grid_size
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
            
            # Flatten axes for easy iteration
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, frame in enumerate(frames[:rows * cols]):
                if i < len(axes):
                    axes[i].imshow(frame)
                    axes[i].set_title(f'Frame {i + 1}')
                    axes[i].axis('off')
            
            # Hide remaining subplots
            for i in range(len(frames), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating frame grid: {str(e)}")
            return None
    
    def analyze_frame_quality(self, frames: List[np.ndarray]) -> Dict:
        """Analyze the quality metrics of extracted frames."""
        try:
            if not frames:
                return {}
            
            quality_metrics = {
                'total_frames': len(frames),
                'average_brightness': [],
                'average_contrast': [],
                'blur_scores': []
            }
            
            for frame in frames:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Brightness (mean pixel intensity)
                brightness = np.mean(gray)
                quality_metrics['average_brightness'].append(brightness)
                
                # Contrast (standard deviation of pixel intensities)
                contrast = np.std(gray)
                quality_metrics['average_contrast'].append(contrast)
                
                # Blur detection using Laplacian variance
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality_metrics['blur_scores'].append(blur_score)
            
            # Calculate summary statistics
            quality_metrics['mean_brightness'] = np.mean(quality_metrics['average_brightness'])
            quality_metrics['mean_contrast'] = np.mean(quality_metrics['average_contrast'])
            quality_metrics['mean_blur_score'] = np.mean(quality_metrics['blur_scores'])
            
            return quality_metrics
        except Exception as e:
            st.error(f"Error analyzing frame quality: {str(e)}")
            return {}
    
    def preprocess_frames_for_prediction(self, frames: List[np.ndarray]) -> np.ndarray:
        """Preprocess frames for ML model input."""
        try:
            if not frames:
                return np.array([])
            
            # Normalize frames
            processed_frames = []
            for frame in frames:
                # Convert to float and normalize to [0, 1]
                normalized_frame = frame.astype(np.float32) / 255.0
                processed_frames.append(normalized_frame)
            
            # Stack frames into a batch
            batch = np.array(processed_frames)
            return batch
        except Exception as e:
            st.error(f"Error preprocessing frames: {str(e)}")
            return np.array([])
    
    def create_video_summary(self, file_path: str) -> Dict:
        """Create a comprehensive summary of the video file."""
        try:
            summary = {}
            
            # Basic video information
            video_info = self.get_video_info(file_path)
            summary['video_info'] = video_info
            
            # Extract a few frames for preview
            preview_frames = self.extract_frames(file_path, num_frames=6, method='uniform')
            summary['preview_frames'] = preview_frames
            
            # Analyze frame quality
            if preview_frames:
                quality_metrics = self.analyze_frame_quality(preview_frames)
                summary['quality_metrics'] = quality_metrics
            
            # Extract audio information
            audio_data, sample_rate = self.extract_audio_from_video(file_path)
            if audio_data is not None:
                summary['audio_info'] = {
                    'has_audio': True,
                    'sample_rate': sample_rate,
                    'duration': len(audio_data) / sample_rate,
                    'rms_energy': np.sqrt(np.mean(audio_data**2))
                }
            else:
                summary['audio_info'] = {'has_audio': False}
            
            return summary
        except Exception as e:
            st.error(f"Error creating video summary: {str(e)}")
            return {}