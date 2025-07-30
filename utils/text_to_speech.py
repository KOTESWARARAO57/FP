import streamlit as st
import tempfile
import os
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")

class TextToSpeechProcessor:
    """Text-to-speech system for multilingual medical diagnosis output."""
    
    def __init__(self):
        self.tts_available = False
        self.engine = None
        self.supported_languages = {
            'en': {'name': 'English', 'voice_id': 'en'},
            'te': {'name': 'Telugu', 'voice_id': 'te'},
            'hi': {'name': 'Hindi', 'voice_id': 'hi'},
            'ta': {'name': 'Tamil', 'voice_id': 'ta'},
            'es': {'name': 'Spanish', 'voice_id': 'es'},
            'fr': {'name': 'French', 'voice_id': 'fr'}
        }
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine with fallback options."""
        try:
            # Try to import and initialize pyttsx3
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.tts_available = True
                st.success("Text-to-speech engine initialized successfully!")
                self._configure_voice_settings()
                
            except ImportError:
                st.info("pyttsx3 not available, TTS features disabled")
                self.tts_available = False
            except Exception as e:
                st.info(f"TTS engine initialization failed: {str(e)}")
                self.tts_available = False
                
        except Exception as e:
            st.warning(f"TTS system initialization failed: {str(e)}")
            self.tts_available = False
    
    def _configure_voice_settings(self):
        """Configure voice settings for optimal speech output."""
        try:
            if self.engine:
                # Get available voices
                voices = self.engine.getProperty('voices')
                
                # Set speech rate (words per minute)
                self.engine.setProperty('rate', 150)  # Slower for medical content
                
                # Set volume
                self.engine.setProperty('volume', 0.9)
                
                # Try to set a clear voice
                if voices:
                    # Prefer female voices for medical content (often perceived as more calming)
                    female_voices = [v for v in voices if 'female' in v.name.lower() or 'woman' in v.name.lower()]
                    if female_voices:
                        self.engine.setProperty('voice', female_voices[0].id)
                    else:
                        self.engine.setProperty('voice', voices[0].id)
                
        except Exception as e:
            st.warning(f"Voice configuration failed: {str(e)}")
    
    def speak_text(self, text: str, language: str = 'en', save_audio: bool = False) -> Dict:
        """
        Convert text to speech and optionally save as audio file.
        
        Args:
            text: Text to convert to speech
            language: Language code for speech
            save_audio: Whether to save audio file
            
        Returns:
            Dictionary with speech generation results
        """
        try:
            if not self.tts_available:
                return {
                    'success': False,
                    'error': 'Text-to-speech not available',
                    'audio_file': None,
                    'text': text,
                    'language': language
                }
            
            if not text or len(text.strip()) == 0:
                return {
                    'success': False,
                    'error': 'No text provided',
                    'audio_file': None,
                    'text': text,
                    'language': language
                }
            
            # Configure voice for language if possible
            self._set_voice_for_language(language)
            
            if save_audio:
                # Save to temporary file
                audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                audio_path = audio_file.name
                audio_file.close()
                
                # Generate speech to file
                self.engine.save_to_file(text, audio_path)
                self.engine.runAndWait()
                
                return {
                    'success': True,
                    'audio_file': audio_path,
                    'text': text,
                    'language': language,
                    'method': 'pyttsx3_file',
                    'duration_estimate': len(text.split()) * 0.5  # Rough estimate
                }
            else:
                # Speak directly
                self.engine.say(text)
                self.engine.runAndWait()
                
                return {
                    'success': True,
                    'audio_file': None,
                    'text': text,
                    'language': language,
                    'method': 'pyttsx3_direct',
                    'duration_estimate': len(text.split()) * 0.5
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Speech generation failed: {str(e)}',
                'audio_file': None,
                'text': text,
                'language': language
            }
    
    def _set_voice_for_language(self, language: str):
        """Set appropriate voice for the given language."""
        try:
            if not self.engine:
                return
            
            voices = self.engine.getProperty('voices')
            if not voices:
                return
            
            # Language-specific voice selection
            language_keywords = {
                'en': ['english', 'en-us', 'en-gb'],
                'te': ['telugu', 'te'],
                'hi': ['hindi', 'hi'],
                'ta': ['tamil', 'ta'],
                'es': ['spanish', 'es'],
                'fr': ['french', 'fr']
            }
            
            keywords = language_keywords.get(language, ['english'])
            
            # Find voice that matches language
            for voice in voices:
                voice_name = voice.name.lower()
                if any(keyword in voice_name for keyword in keywords):
                    self.engine.setProperty('voice', voice.id)
                    return
            
            # Fallback to first available voice
            self.engine.setProperty('voice', voices[0].id)
            
        except Exception as e:
            st.warning(f"Voice selection failed: {str(e)}")
    
    def create_medical_announcement(self, diagnosis_result: Dict, language: str = 'en') -> Dict:
        """
        Create structured medical announcement text for TTS.
        
        Args:
            diagnosis_result: Medical diagnosis result
            language: Target language for announcement
            
        Returns:
            Structured announcement text
        """
        try:
            predicted_condition = diagnosis_result.get('predicted_class', 'Unknown')
            confidence = diagnosis_result.get('confidence', 0.0)
            
            # Create structured announcement
            if language == 'en':
                announcement = self._create_english_announcement(predicted_condition, confidence)
            elif language == 'te':
                announcement = self._create_telugu_announcement(predicted_condition, confidence)
            else:
                # Fallback to English
                announcement = self._create_english_announcement(predicted_condition, confidence)
            
            return {
                'success': True,
                'announcement_text': announcement,
                'language': language,
                'condition': predicted_condition,
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Announcement creation failed: {str(e)}',
                'announcement_text': 'Medical analysis completed.',
                'language': language
            }
    
    def _create_english_announcement(self, condition: str, confidence: float) -> str:
        """Create English medical announcement."""
        confidence_text = f"{confidence:.0%}"
        
        if condition.lower() == 'healthy':
            return f"Medical analysis completed. The system indicates a healthy condition with {confidence_text} confidence. This analysis is for informational purposes only and should not replace professional medical consultation."
        else:
            return f"Medical analysis completed. The system indicates potential signs of {condition} with {confidence_text} confidence. Please consult with a healthcare professional for proper evaluation and diagnosis. This analysis is for research and demonstration purposes only."
    
    def _create_telugu_announcement(self, condition: str, confidence: float) -> str:
        """Create Telugu medical announcement (basic version)."""
        confidence_text = f"{confidence:.0%}"
        
        # Basic Telugu translation (would be enhanced with proper translation)
        condition_telugu = {
            'Healthy': 'ఆరోగ్యకరమైన',
            'Depression': 'డిప్రెషన్',
            'Parkinson\'s Disease': 'పార్కిన్సన్ వ్యాధి',
            'Hypothyroidism': 'హైపోథైరాయిడిజం'
        }.get(condition, condition)
        
        if condition.lower() == 'healthy':
            return f"వైద్య విశ్లేషణ పూర్తయింది. సిస్టమ్ {confidence_text} విశ్వాసంతో ఆరోగ్యకరమైన పరిస్థితిని సూచిస్తుంది. ఈ విశ్లేషణ కేవలం సమాచార ప్రయోజనాల కోసం మాత్రమే."
        else:
            return f"వైద్య విశ్లేషణ పూర్తయింది. సిస్టమ్ {confidence_text} విశ్వాసంతో {condition_telugu} లక్షణాలను సూచిస్తుంది. దయచేసి సరైన మూల్యాంకనం మరియు నిర్ధారణ కోసం వైద్య నిపుణులను సంప్రదించండి."
    
    def speak_medical_diagnosis(self, diagnosis_result: Dict, language: str = 'en', save_audio: bool = True) -> Dict:
        """
        Convert medical diagnosis to speech with appropriate formatting.
        
        Args:
            diagnosis_result: Medical diagnosis result
            language: Target language for speech
            save_audio: Whether to save audio file
            
        Returns:
            Speech generation result
        """
        try:
            # Create medical announcement
            announcement_result = self.create_medical_announcement(diagnosis_result, language)
            
            if not announcement_result['success']:
                return announcement_result
            
            # Convert to speech
            speech_result = self.speak_text(
                announcement_result['announcement_text'],
                language,
                save_audio
            )
            
            # Combine results
            if speech_result['success']:
                speech_result.update({
                    'announcement_text': announcement_result['announcement_text'],
                    'medical_condition': announcement_result.get('condition', 'Unknown'),
                    'diagnosis_confidence': announcement_result.get('confidence', 0.0)
                })
            
            return speech_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Medical diagnosis speech failed: {str(e)}',
                'audio_file': None,
                'language': language
            }
    
    def get_available_voices(self) -> List[Dict]:
        """Get list of available voices."""
        try:
            if not self.tts_available or not self.engine:
                return []
            
            voices = self.engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', []),
                    'gender': getattr(voice, 'gender', 'unknown')
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            st.warning(f"Could not get voice list: {str(e)}")
            return []
    
    def cleanup_audio_files(self, audio_files: List[str]):
        """Clean up temporary audio files."""
        try:
            for file_path in audio_files:
                if os.path.exists(file_path):
                    os.unlink(file_path)
        except Exception as e:
            st.warning(f"Audio cleanup failed: {str(e)}")
    
    def test_tts_system(self) -> Dict:
        """Test TTS system functionality."""
        try:
            if not self.tts_available:
                return {
                    'success': False,
                    'error': 'TTS system not available',
                    'test_result': 'failed'
                }
            
            test_text = "Text to speech system test successful."
            
            result = self.speak_text(test_text, 'en', save_audio=False)
            
            return {
                'success': result['success'],
                'test_result': 'passed' if result['success'] else 'failed',
                'error': result.get('error', None),
                'engine_available': self.tts_available
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'TTS test failed: {str(e)}',
                'test_result': 'failed',
                'engine_available': False
            }