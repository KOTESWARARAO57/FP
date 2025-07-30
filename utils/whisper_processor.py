import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

class WhisperProcessor:
    """Advanced speech-to-text using OpenAI Whisper with multilingual support."""
    
    def __init__(self):
        self.model = None
        self.model_size = "base"  # Start with base model for efficiency
        self.supported_languages = {
            'en': 'English',
            'te': 'Telugu',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese'
        }
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model with caching."""
        try:
            if not WHISPER_AVAILABLE:
                st.warning("Whisper not available, using fallback speech recognition")
                self.model = None
                return
                
            if self.model is None:
                st.info(f"Loading Whisper {self.model_size} model...")
                self.model = whisper.load_model(self.model_size)
                st.success("Whisper model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading Whisper model: {str(e)}")
            # Fallback to tiny model if base fails
            try:
                self.model_size = "tiny"
                self.model = whisper.load_model(self.model_size)
                st.warning("Loaded Whisper tiny model as fallback")
            except:
                st.warning("Whisper not available, using fallback speech recognition")
                self.model = None
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio using Whisper with enhanced language detection.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'te') or None for auto-detection
            
        Returns:
            Dictionary with transcription results and metadata
        """
        if self.model is None or not WHISPER_AVAILABLE:
            # Fallback to basic speech recognition
            return self._fallback_speech_recognition(audio_path, language)
        
        if self.model is None:
            return {
                'success': False,
                'error': 'Whisper model not loaded',
                'text': '',
                'language': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Transcribe with Whisper
            if language and language in self.supported_languages:
                result = self.model.transcribe(audio_path, language=language)
            else:
                # Auto-detect language
                result = self.model.transcribe(audio_path)
            
            # Extract transcription details
            transcribed_text = result.get('text', '').strip()
            detected_language = result.get('language', 'unknown')
            
            # Calculate average confidence from segments
            segments = result.get('segments', [])
            if segments:
                # Whisper doesn't provide direct confidence, estimate from segment data
                avg_confidence = self._estimate_confidence(segments)
            else:
                avg_confidence = 0.8 if transcribed_text else 0.0
            
            return {
                'success': True,
                'text': transcribed_text,
                'language': detected_language,
                'language_name': self.supported_languages.get(detected_language, 'Unknown'),
                'confidence': avg_confidence,
                'segments': segments,
                'model_used': f"whisper-{self.model_size}",
                'duration': result.get('duration', 0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Whisper transcription failed: {str(e)}',
                'text': '',
                'language': 'unknown',
                'confidence': 0.0
            }
    
    def _fallback_speech_recognition(self, audio_path: str, language: str = None) -> Dict:
        """Fallback speech recognition using SpeechRecognition library."""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # Convert to WAV if needed
            try:
                import librosa
                y, sr_rate = librosa.load(audio_path, sr=16000)
                
                # Save as WAV temporarily
                import soundfile as sf
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_wav.name, y, sr_rate)
                temp_wav.close()
                
                with sr.AudioFile(temp_wav.name) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)
                
                # Try Google Speech Recognition
                try:
                    lang_code = language if language else 'en'
                    text = recognizer.recognize_google(audio_data, language=lang_code)
                    
                    # Clean up temp file
                    os.unlink(temp_wav.name)
                    
                    return {
                        'success': True,
                        'text': text,
                        'language': lang_code,
                        'language_name': self.supported_languages.get(lang_code, 'Unknown'),
                        'confidence': 0.7,  # Estimate
                        'segments': [],
                        'model_used': 'speech-recognition-fallback',
                        'duration': len(y) / sr_rate
                    }
                    
                except sr.UnknownValueError:
                    # Clean up temp file
                    if os.path.exists(temp_wav.name):
                        os.unlink(temp_wav.name)
                    return {
                        'success': False,
                        'error': 'Could not understand audio',
                        'text': '',
                        'language': 'unknown',
                        'confidence': 0.0
                    }
                except sr.RequestError as e:
                    # Clean up temp file
                    if os.path.exists(temp_wav.name):
                        os.unlink(temp_wav.name)
                    return {
                        'success': False,
                        'error': f'Speech recognition service error: {str(e)}',
                        'text': '',
                        'language': 'unknown',
                        'confidence': 0.0
                    }
            
            except ImportError:
                return {
                    'success': False,
                    'error': 'Required audio libraries not available',
                    'text': '',
                    'language': 'unknown',
                    'confidence': 0.0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Fallback speech recognition failed: {str(e)}',
                'text': '',
                'language': 'unknown',
                'confidence': 0.0
            }
    
    def _estimate_confidence(self, segments: List[Dict]) -> float:
        """
        Estimate transcription confidence from segment data.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Estimated confidence score (0.0 to 1.0)
        """
        if not segments:
            return 0.0
        
        # Calculate confidence based on segment properties
        total_confidence = 0.0
        total_duration = 0.0
        
        for segment in segments:
            duration = segment.get('end', 0) - segment.get('start', 0)
            text_length = len(segment.get('text', '').strip())
            
            # Higher confidence for longer, more complete segments
            segment_confidence = min(1.0, (text_length / max(duration * 10, 1)))
            
            total_confidence += segment_confidence * duration
            total_duration += duration
        
        if total_duration > 0:
            return min(1.0, total_confidence / total_duration)
        else:
            return 0.5
    
    def extract_speech_features(self, audio_path: str, transcription_result: Dict) -> Dict:
        """
        Extract speech features for medical analysis.
        
        Args:
            audio_path: Path to audio file
            transcription_result: Result from Whisper transcription
            
        Returns:
            Dictionary of extracted speech features
        """
        try:
            import librosa
            
            # Load audio for feature extraction
            y, sr = librosa.load(audio_path, sr=None)
            
            # Basic audio features
            features = {
                'duration': len(y) / sr,
                'sample_rate': sr,
                'audio_length': len(y)
            }
            
            # Prosodic features
            features.update(self._extract_prosodic_features(y, sr))
            
            # Fluency features from transcription
            features.update(self._extract_fluency_features(transcription_result))
            
            # Lexical features from text
            features.update(self._extract_lexical_features(transcription_result.get('text', '')))
            
            return features
            
        except Exception as e:
            st.warning(f"Feature extraction partially failed: {str(e)}")
            # Return basic features from transcription
            return self._extract_basic_features(transcription_result)
    
    def _extract_prosodic_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract prosodic features (pitch, rhythm, stress patterns)."""
        try:
            import librosa
            
            features = {}
            
            # Fundamental frequency (pitch)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            
            if len(f0_clean) > 0:
                features['pitch_mean'] = float(np.mean(f0_clean))
                features['pitch_std'] = float(np.std(f0_clean))
                features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
            
            # Energy features
            rms = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['rhythm_regularity'] = float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0
            
            return features
            
        except Exception as e:
            st.warning(f"Prosodic feature extraction failed: {str(e)}")
            return {
                'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0,
                'energy_mean': 0.0, 'energy_std': 0.0,
                'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
                'tempo': 0.0, 'rhythm_regularity': 0.0
            }
    
    def _extract_fluency_features(self, transcription_result: Dict) -> Dict:
        """Extract fluency features from transcription segments."""
        try:
            segments = transcription_result.get('segments', [])
            text = transcription_result.get('text', '')
            
            if not segments:
                return {
                    'speech_rate': 0.0,
                    'pause_frequency': 0.0,
                    'avg_pause_duration': 0.0,
                    'segment_count': 0,
                    'words_per_segment': 0.0
                }
            
            # Calculate speech rate (words per minute)
            total_duration = transcription_result.get('duration', 0)
            word_count = len(text.split())
            speech_rate = (word_count / max(total_duration / 60, 0.1)) if total_duration > 0 else 0.0
            
            # Analyze pauses between segments
            pause_durations = []
            for i in range(len(segments) - 1):
                pause = segments[i+1]['start'] - segments[i]['end']
                if pause > 0.1:  # Consider pauses > 100ms
                    pause_durations.append(pause)
            
            pause_frequency = len(pause_durations) / max(total_duration, 0.1)
            avg_pause_duration = np.mean(pause_durations) if pause_durations else 0.0
            
            # Segment analysis
            words_per_segment = word_count / len(segments) if segments else 0.0
            
            return {
                'speech_rate': float(speech_rate),
                'pause_frequency': float(pause_frequency),
                'avg_pause_duration': float(avg_pause_duration),
                'segment_count': len(segments),
                'words_per_segment': float(words_per_segment)
            }
            
        except Exception as e:
            st.warning(f"Fluency feature extraction failed: {str(e)}")
            return {
                'speech_rate': 0.0, 'pause_frequency': 0.0, 'avg_pause_duration': 0.0,
                'segment_count': 0, 'words_per_segment': 0.0
            }
    
    def _extract_lexical_features(self, text: str) -> Dict:
        """Extract lexical features from transcribed text."""
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'word_count': 0, 'unique_words': 0, 'lexical_diversity': 0.0,
                    'avg_word_length': 0.0, 'sentence_count': 0,
                    'medical_keywords': 0, 'emotional_words': 0
                }
            
            words = text.lower().split()
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Basic lexical measures
            word_count = len(words)
            unique_words = len(set(words))
            lexical_diversity = unique_words / max(word_count, 1)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
            
            # Medical and emotional keyword detection
            medical_keywords = [
                'pain', 'tired', 'sleep', 'energy', 'mood', 'stress', 'anxiety',
                'depression', 'memory', 'concentration', 'movement', 'balance',
                'voice', 'speech', 'tremor', 'stiff', 'slow', 'weak'
            ]
            
            emotional_words = [
                'sad', 'happy', 'angry', 'frustrated', 'hopeless', 'worried',
                'confused', 'lost', 'empty', 'lonely', 'scared', 'nervous'
            ]
            
            medical_count = sum(1 for word in words if word in medical_keywords)
            emotional_count = sum(1 for word in words if word in emotional_words)
            
            return {
                'word_count': word_count,
                'unique_words': unique_words,
                'lexical_diversity': float(lexical_diversity),
                'avg_word_length': float(avg_word_length),
                'sentence_count': len(sentences),
                'medical_keywords': medical_count,
                'emotional_words': emotional_count
            }
            
        except Exception as e:
            st.warning(f"Lexical feature extraction failed: {str(e)}")
            return {
                'word_count': 0, 'unique_words': 0, 'lexical_diversity': 0.0,
                'avg_word_length': 0.0, 'sentence_count': 0,
                'medical_keywords': 0, 'emotional_words': 0
            }
    
    def _extract_basic_features(self, transcription_result: Dict) -> Dict:
        """Extract basic features when full feature extraction fails."""
        text = transcription_result.get('text', '')
        duration = transcription_result.get('duration', 0)
        
        return {
            'duration': duration,
            'word_count': len(text.split()),
            'character_count': len(text),
            'language': transcription_result.get('language', 'unknown'),
            'confidence': transcription_result.get('confidence', 0.0)
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Return supported languages dictionary."""
        return self.supported_languages.copy()
    
    def change_model_size(self, size: str):
        """
        Change Whisper model size.
        
        Args:
            size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        if size in valid_sizes:
            self.model_size = size
            self.model = None  # Force reload
            self._load_model()
        else:
            st.error(f"Invalid model size. Choose from: {valid_sizes}")