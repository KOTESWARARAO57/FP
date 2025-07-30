import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import streamlit as st
from typing import Tuple, Optional, Dict

class ImageProcessor:
    """Handles image file processing and analysis for medical AI predictions."""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.target_size = (224, 224)  # Standard size for many medical image models
    
    def validate_image_file(self, file) -> bool:
        """Validate if the uploaded file is a supported image format."""
        if file is None:
            return False
        
        file_extension = file.name.lower().split('.')[-1]
        return f'.{file_extension}' in self.supported_formats
    
    def load_image(self, file) -> Optional[Image.Image]:
        """Load and return PIL Image object."""
        try:
            image = Image.open(file)
            return image.convert('RGB')  # Ensure consistent format
        except Exception as e:
            st.error(f"Error loading image file: {str(e)}")
            return None
    
    def extract_image_info(self, image: Image.Image) -> Dict:
        """Extract basic image information."""
        try:
            info = {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': getattr(image, 'format', 'Unknown'),
                'size_mb': len(image.tobytes()) / (1024 * 1024)
            }
            return info
        except Exception as e:
            st.error(f"Error extracting image info: {str(e)}")
            return {}
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for medical analysis."""
        try:
            # Resize image to target size
            resized_image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(resized_image, dtype=np.float32)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            return img_array
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return np.array([])
    
    def extract_features(self, image: Image.Image) -> Dict:
        """Extract relevant features from medical images."""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            features = {}
            
            # Basic image statistics
            features['mean_intensity'] = np.mean(img_array)
            features['std_intensity'] = np.std(img_array)
            features['min_intensity'] = np.min(img_array)
            features['max_intensity'] = np.max(img_array)
            
            # Color channel statistics (if RGB)
            if len(img_array.shape) == 3:
                for i, channel in enumerate(['red', 'green', 'blue']):
                    features[f'{channel}_mean'] = np.mean(img_array[:, :, i])
                    features[f'{channel}_std'] = np.std(img_array[:, :, i])
            
            # Texture features using OpenCV
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate histogram features
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            features['histogram_mean'] = np.mean(hist)
            features['histogram_std'] = np.std(hist)
            
            # Edge detection features
            edges = cv2.Canny(gray_image, 100, 200)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Contrast and brightness
            features['contrast'] = np.std(gray_image)
            features['brightness'] = np.mean(gray_image)
            
            return features
        except Exception as e:
            st.error(f"Error extracting image features: {str(e)}")
            return {}
    
    def create_enhanced_visualization(self, image: Image.Image) -> Tuple[plt.Figure, Dict]:
        """Create enhanced visualization of the medical image with analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Grayscale version
            gray_image = image.convert('L')
            axes[0, 1].imshow(gray_image, cmap='gray')
            axes[0, 1].set_title('Grayscale')
            axes[0, 1].axis('off')
            
            # Histogram
            img_array = np.array(gray_image)
            axes[1, 0].hist(img_array.flatten(), bins=50, alpha=0.7, color='blue')
            axes[1, 0].set_title('Intensity Histogram')
            axes[1, 0].set_xlabel('Intensity')
            axes[1, 0].set_ylabel('Frequency')
            
            # Edge detection
            edges = cv2.Canny(np.array(gray_image), 100, 200)
            axes[1, 1].imshow(edges, cmap='gray')
            axes[1, 1].set_title('Edge Detection')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Calculate analysis metrics
            analysis = {
                'mean_intensity': np.mean(img_array),
                'contrast': np.std(img_array),
                'edge_density': np.sum(edges > 0) / edges.size * 100
            }
            
            return fig, analysis
        except Exception as e:
            st.error(f"Error creating image visualization: {str(e)}")
            return None, {}
    
    def preprocess_for_prediction(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ML model input."""
        try:
            # Resize and normalize
            preprocessed = self.preprocess_image(image)
            
            # Reshape for model input (batch_size, height, width, channels)
            if len(preprocessed.shape) == 3:
                preprocessed = preprocessed.reshape(1, *preprocessed.shape)
            
            return preprocessed
        except Exception as e:
            st.error(f"Error preprocessing image for prediction: {str(e)}")
            return np.array([])
    
    def create_prediction_overlay(self, image: Image.Image, prediction_results: Dict) -> Image.Image:
        """Create an image with prediction results overlaid."""
        try:
            # Create a copy of the original image
            overlay_image = image.copy()
            
            # Convert to numpy array for drawing
            img_array = np.array(overlay_image)
            
            # Add text overlay with prediction results
            # This is a simplified version - in practice, you might highlight specific regions
            height, width = img_array.shape[:2]
            
            # Create a semi-transparent overlay for text background
            overlay = img_array.copy()
            cv2.rectangle(overlay, (10, 10), (width - 10, 100), (0, 0, 0), -1)
            img_array = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
            
            # Add prediction text
            if 'predicted_class' in prediction_results:
                text = f"Prediction: {prediction_results['predicted_class']}"
                cv2.putText(img_array, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if 'confidence' in prediction_results:
                confidence_text = f"Confidence: {prediction_results['confidence']:.2%}"
                cv2.putText(img_array, confidence_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return Image.fromarray(img_array)
        except Exception as e:
            st.error(f"Error creating prediction overlay: {str(e)}")
            return image
