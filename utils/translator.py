import streamlit as st
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

class MultilingualTranslator:
    """Multilingual translation system for Telugu-English medical text."""
    
    def __init__(self):
        self.translator_available = False
        self.supported_languages = {
            'en': 'English',
            'te': 'Telugu',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'es': 'Spanish',
            'fr': 'French'
        }
        self.medical_dictionary = self._initialize_medical_dictionary()
        self._initialize_translator()
    
    def _initialize_translator(self):
        """Initialize translation system with fallback options."""
        try:
            # Try to import and initialize argostranslate
            try:
                import argostranslate.package
                import argostranslate.translate
                
                self.argos_package = argostranslate.package
                self.argos_translate = argostranslate.translate
                
                # Download required language packages if available
                self._setup_language_packages()
                self.translator_available = True
                st.success("Argos Translate initialized successfully!")
                
            except ImportError:
                st.info("Argos Translate not available, using built-in medical dictionary")
                self.translator_available = False
                
        except Exception as e:
            st.warning(f"Translation system initialization failed: {str(e)}")
            self.translator_available = False
    
    def _setup_language_packages(self):
        """Setup required language packages for translation."""
        try:
            # Update package index
            self.argos_package.update_package_index()
            
            # Get available packages
            available_packages = self.argos_package.get_available_packages()
            
            # Required language pairs
            required_pairs = [
                ('te', 'en'),  # Telugu to English
                ('en', 'te'),  # English to Telugu
                ('hi', 'en'),  # Hindi to English
                ('en', 'hi')   # English to Hindi
            ]
            
            installed_packages = []
            for from_code, to_code in required_pairs:
                package = next(
                    (pkg for pkg in available_packages 
                     if pkg.from_code == from_code and pkg.to_code == to_code), 
                    None
                )
                if package and not package.package.is_installed():
                    try:
                        self.argos_package.install_from_path(package.download())
                        installed_packages.append(f"{from_code}-{to_code}")
                    except Exception as e:
                        st.warning(f"Could not install {from_code}-{to_code} package: {str(e)}")
            
            if installed_packages:
                st.info(f"Installed translation packages: {', '.join(installed_packages)}")
                
        except Exception as e:
            st.warning(f"Package setup failed: {str(e)}")
    
    def _initialize_medical_dictionary(self) -> Dict[str, Dict[str, str]]:
        """Initialize medical term dictionary for fallback translation."""
        return {
            'te': {  # Telugu medical terms
                # Basic health terms
                'ఆరోగ్యం': 'health',
                'వైద్యుడు': 'doctor',
                'దవాఖానా': 'hospital',
                'మందు': 'medicine',
                'నొప్పి': 'pain',
                'జ్వరం': 'fever',
                'తలనొప్పి': 'headache',
                'కడుపునొప్పి': 'stomach pain',
                
                # Mental health terms
                'మానసిక': 'mental',
                'డిప్రెషన్': 'depression',
                'ఆందోళన': 'anxiety',
                'ఒత్తిడి': 'stress',
                'దుఃఖం': 'sadness',
                'చిరాకు': 'irritation',
                'నిరాశ': 'despair',
                'భయం': 'fear',
                
                # Neurological terms
                'కంపనలు': 'tremors',
                'వణుకు': 'shaking',
                'కదలిక': 'movement',
                'సమతుల్యత': 'balance',
                'మెమరీ': 'memory',
                'దృష్టి': 'vision',
                'వినికిడి': 'hearing',
                'మాట': 'speech',
                
                # Symptoms
                'అలసట': 'fatigue',
                'నిద్రలేకపోవడం': 'insomnia',
                'ఆకలిలేకపోవడం': 'loss of appetite',
                'బరువు తగ్గడం': 'weight loss',
                'బరువు పెరుగుట': 'weight gain',
                'మూలికలు': 'herbs',
                'వైద్యం': 'treatment'
            },
            'hi': {  # Hindi medical terms
                'स्वास्थ्य': 'health',
                'डॉक्टर': 'doctor',
                'अस्पताल': 'hospital',
                'दवा': 'medicine',
                'दर्द': 'pain',
                'बुखार': 'fever',
                'सिरदर्द': 'headache',
                'अवसाद': 'depression',
                'चिंता': 'anxiety',
                'तनाव': 'stress',
                'कांपना': 'tremor',
                'थकान': 'fatigue',
                'नींद': 'sleep',
                'भूख': 'appetite'
            }
        }
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """
        Translate text between languages with medical context awareness.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translation result dictionary
        """
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'success': False,
                    'error': 'No text provided',
                    'translated_text': '',
                    'source_language': source_lang,
                    'target_language': target_lang
                }
            
            # Check if translation is needed
            if source_lang == target_lang:
                return {
                    'success': True,
                    'translated_text': text,
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'method': 'no_translation_needed'
                }
            
            # Try Argos Translate first
            if self.translator_available:
                result = self._argos_translate_text(text, source_lang, target_lang)
                if result['success']:
                    return result
            
            # Fallback to dictionary-based translation
            return self._dictionary_translate_text(text, source_lang, target_lang)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Translation failed: {str(e)}',
                'translated_text': text,  # Return original text as fallback
                'source_language': source_lang,
                'target_language': target_lang
            }
    
    def _argos_translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """Translate using Argos Translate."""
        try:
            translated_text = self.argos_translate.translate(text, source_lang, target_lang)
            
            return {
                'success': True,
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'method': 'argos_translate',
                'confidence': 0.8  # Assume good quality for Argos
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Argos translation failed: {str(e)}',
                'translated_text': '',
                'source_language': source_lang,
                'target_language': target_lang
            }
    
    def _dictionary_translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """Fallback dictionary-based translation for medical terms."""
        try:
            if source_lang not in self.medical_dictionary:
                return {
                    'success': False,
                    'error': f'Dictionary not available for {source_lang}',
                    'translated_text': text,
                    'source_language': source_lang,
                    'target_language': target_lang
                }
            
            # Simple word-by-word replacement for medical terms
            translated_text = text.lower()
            dictionary = self.medical_dictionary[source_lang]
            
            for source_term, target_term in dictionary.items():
                translated_text = translated_text.replace(source_term.lower(), target_term)
            
            # Capitalize first letter
            if translated_text:
                translated_text = translated_text[0].upper() + translated_text[1:]
            
            return {
                'success': True,
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'method': 'dictionary_based',
                'confidence': 0.5,  # Lower confidence for dictionary method
                'note': 'Basic medical term translation only'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Dictionary translation failed: {str(e)}',
                'translated_text': text,
                'source_language': source_lang,
                'target_language': target_lang
            }
    
    def translate_medical_diagnosis(self, diagnosis_text: str, target_lang: str) -> Dict:
        """
        Translate medical diagnosis text with medical context preservation.
        
        Args:
            diagnosis_text: Medical diagnosis text to translate
            target_lang: Target language code
            
        Returns:
            Translation result with medical context
        """
        try:
            # Detect if text contains medical terminology
            medical_terms = self._detect_medical_terms(diagnosis_text)
            
            # Translate the text
            translation_result = self.translate_text(diagnosis_text, 'en', target_lang)
            
            if translation_result['success']:
                # Post-process to ensure medical accuracy
                processed_text = self._post_process_medical_translation(
                    translation_result['translated_text'], 
                    target_lang, 
                    medical_terms
                )
                
                translation_result['translated_text'] = processed_text
                translation_result['medical_terms_detected'] = medical_terms
                translation_result['medical_context'] = True
            
            return translation_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Medical translation failed: {str(e)}',
                'translated_text': diagnosis_text,
                'target_language': target_lang
            }
    
    def _detect_medical_terms(self, text: str) -> List[str]:
        """Detect medical terms in text."""
        medical_keywords = [
            'depression', 'anxiety', 'parkinson', 'hypothyroidism', 'disease',
            'condition', 'symptoms', 'diagnosis', 'treatment', 'therapy',
            'medication', 'health', 'medical', 'clinical', 'patient',
            'tremor', 'fatigue', 'mood', 'cognitive', 'neurological'
        ]
        
        text_lower = text.lower()
        detected_terms = [term for term in medical_keywords if term in text_lower]
        return detected_terms
    
    def _post_process_medical_translation(self, translated_text: str, target_lang: str, medical_terms: List[str]) -> str:
        """Post-process translated text to maintain medical accuracy."""
        try:
            # For now, return the translated text as-is
            # In a full implementation, this would include:
            # - Medical term verification
            # - Context-aware corrections
            # - Standardized medical terminology mapping
            
            return translated_text
            
        except Exception:
            return translated_text
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes and names."""
        return self.supported_languages.copy()
    
    def create_bilingual_output(self, english_text: str, target_lang: str) -> Dict:
        """
        Create bilingual output with both English and target language.
        
        Args:
            english_text: English text
            target_lang: Target language code
            
        Returns:
            Bilingual text output
        """
        try:
            if target_lang == 'en':
                return {
                    'success': True,
                    'english_text': english_text,
                    'translated_text': english_text,
                    'bilingual_text': english_text,
                    'language': target_lang
                }
            
            # Translate to target language
            translation_result = self.translate_medical_diagnosis(english_text, target_lang)
            
            if translation_result['success']:
                translated_text = translation_result['translated_text']
                bilingual_text = f"{english_text}\n\n{self.supported_languages.get(target_lang, target_lang)}: {translated_text}"
                
                return {
                    'success': True,
                    'english_text': english_text,
                    'translated_text': translated_text,
                    'bilingual_text': bilingual_text,
                    'language': target_lang,
                    'translation_method': translation_result.get('method', 'unknown')
                }
            else:
                return {
                    'success': False,
                    'error': translation_result['error'],
                    'english_text': english_text,
                    'translated_text': english_text,
                    'bilingual_text': english_text,
                    'language': target_lang
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Bilingual output creation failed: {str(e)}',
                'english_text': english_text,
                'translated_text': english_text,
                'bilingual_text': english_text,
                'language': target_lang
            }