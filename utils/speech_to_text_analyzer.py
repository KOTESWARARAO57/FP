import speech_recognition as sr
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
import os
import tempfile
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import librosa
from textblob import TextBlob
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class SpeechToTextAnalyzer:
    """Converts speech to text and analyzes text for medical diagnosis."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.target_labels = ['Healthy', 'Depression', 'Parkinson\'s Disease', 'Hypothyroidism']
        
        # Text analysis models
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.text_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.text_scaler = StandardScaler()
        
        # Language patterns for diseases
        self.disease_patterns = self._initialize_disease_patterns()
        
        # Train text analysis model
        self._train_text_model()
    
    def _initialize_disease_patterns(self) -> Dict[str, List[str]]:
        """Initialize text patterns associated with different medical conditions."""
        return {
            'Depression': [
                'sad', 'hopeless', 'worthless', 'tired', 'exhausted', 'empty', 'lonely',
                'depressed', 'down', 'low', 'unmotivated', 'helpless', 'guilty', 'shame',
                'crying', 'tears', 'sleep', 'insomnia', 'appetite', 'weight', 'concentration',
                'slow', 'sluggish', 'energy', 'interest', 'pleasure', 'death', 'suicide'
            ],
            'Parkinson\'s Disease': [
                'tremor', 'shaking', 'stiff', 'rigid', 'slow', 'movement', 'balance',
                'walking', 'shuffle', 'frozen', 'coordination', 'handwriting', 'small',
                'voice', 'quiet', 'mumble', 'slurred', 'speech', 'facial', 'expression',
                'mask', 'constipation', 'smell', 'olfactory', 'REM', 'sleep', 'disorder'
            ],
            'Hypothyroidism': [
                'tired', 'fatigue', 'cold', 'weight', 'gain', 'hair', 'loss', 'dry',
                'skin', 'constipation', 'memory', 'concentration', 'slow', 'heart',
                'rate', 'depression', 'muscle', 'weakness', 'joint', 'pain', 'heavy',
                'periods', 'fertility', 'puffy', 'face', 'hoarse', 'voice', 'swelling'
            ],
            'Healthy': [
                'good', 'well', 'fine', 'healthy', 'normal', 'active', 'energy',
                'strong', 'regular', 'exercise', 'sleep', 'appetite', 'mood', 'happy',
                'positive', 'motivated', 'clear', 'focused', 'balanced', 'stable'
            ]
        }
    
    def convert_speech_to_text(self, audio_file_path: str, language: str = 'en-US') -> Dict:
        """Convert speech audio to text using speech recognition."""
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise and record audio
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Convert audio to text
            try:
                # Try Google Speech Recognition first
                text = self.recognizer.recognize_google(audio_data, language=language)
                
                return {
                    'success': True,
                    'text': text,
                    'method': 'Google Speech Recognition',
                    'confidence': 'High',
                    'language': language
                }
            except sr.UnknownValueError:
                return {
                    'success': False,
                    'error': 'Could not understand audio',
                    'method': 'Google Speech Recognition'
                }
            except sr.RequestError as e:
                # Fallback to offline recognition if available
                try:
                    text = self.recognizer.recognize_sphinx(audio_data)
                    return {
                        'success': True,
                        'text': text,
                        'method': 'Offline Recognition',
                        'confidence': 'Medium',
                        'language': language
                    }
                except:
                    return {
                        'success': False,
                        'error': f'Speech recognition service error: {str(e)}',
                        'method': 'All methods failed'
                    }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Audio processing error: {str(e)}',
                'method': 'File processing'
            }
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract linguistic and semantic features from text."""
        if not text or len(text.strip()) == 0:
            return {}
        
        try:
            # Basic text statistics
            words = text.split()
            sentences = text.split('.')
            
            # TextBlob analysis
            blob = TextBlob(text)
            
            features = {
                # Basic metrics
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': len(words) / len(sentences) if len(sentences) > 0 else 0,
                
                # Sentiment analysis
                'sentiment_polarity': blob.sentiment.polarity,  # -1 to 1
                'sentiment_subjectivity': blob.sentiment.subjectivity,  # 0 to 1
                
                # Linguistic patterns
                'question_marks': text.count('?'),
                'exclamation_marks': text.count('!'),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                
                # Medical content analysis
                'medical_keywords_count': self._count_medical_keywords(text),
                'depression_indicators': self._count_condition_keywords(text, 'Depression'),
                'parkinson_indicators': self._count_condition_keywords(text, 'Parkinson\'s Disease'),
                'hypothyroid_indicators': self._count_condition_keywords(text, 'Hypothyroidism'),
                'healthy_indicators': self._count_condition_keywords(text, 'Healthy'),
                
                # Speech patterns (inferred from text)
                'repetitive_words': self._count_repetitive_words(text),
                'incomplete_sentences': self._count_incomplete_sentences(text),
                'coherence_score': self._calculate_coherence_score(text)
            }
            
            return features
            
        except Exception as e:
            st.error(f"Error extracting text features: {str(e)}")
            return {}
    
    def _count_medical_keywords(self, text: str) -> int:
        """Count medical-related keywords in text."""
        medical_terms = [
            'doctor', 'hospital', 'medicine', 'symptoms', 'pain', 'treatment',
            'medication', 'therapy', 'diagnosis', 'health', 'illness', 'disease',
            'condition', 'clinic', 'physician', 'nurse', 'specialist'
        ]
        text_lower = text.lower()
        return sum(1 for term in medical_terms if term in text_lower)
    
    def _count_condition_keywords(self, text: str, condition: str) -> int:
        """Count keywords specific to a medical condition."""
        keywords = self.disease_patterns.get(condition, [])
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    def _count_repetitive_words(self, text: str) -> int:
        """Count repetitive word patterns."""
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Count words that appear more than twice
        return sum(1 for count in word_counts.values() if count > 2)
    
    def _count_incomplete_sentences(self, text: str) -> int:
        """Count sentences that appear incomplete."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        incomplete = 0
        
        for sentence in sentences:
            # Check for incomplete patterns
            if len(sentence.split()) < 3:  # Very short sentences
                incomplete += 1
            elif sentence.endswith(('...', '--', '-')):  # Trailing patterns
                incomplete += 1
            elif not sentence[0].isupper():  # Doesn't start with capital
                incomplete += 1
                
        return incomplete
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate text coherence score (0-1, higher is more coherent)."""
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                return 1.0
            
            # Simple coherence based on sentence length variance
            sentence_lengths = [len(s.split()) for s in sentences]
            if not sentence_lengths:
                return 0.0
            
            # Lower variance in sentence length indicates better coherence
            variance = np.var(sentence_lengths)
            mean_length = np.mean(sentence_lengths)
            
            if mean_length == 0:
                return 0.0
            
            # Normalize coherence score
            coherence = 1 / (1 + variance / mean_length)
            return min(1.0, max(0.0, coherence))
            
        except:
            return 0.5  # Default neutral score
    
    def _generate_training_data(self, n_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for text-based classification."""
        np.random.seed(42)
        
        # Sample texts for each condition
        sample_texts = {
            'Healthy': [
                "I feel great today, full of energy and ready for anything",
                "My sleep has been good and I'm maintaining regular exercise",
                "Everything seems normal, my mood is stable and positive",
                "I have good appetite and clear thinking lately"
            ],
            'Depression': [
                "I feel so tired all the time, can't find motivation for anything",
                "Everything seems hopeless, I'm having trouble sleeping",
                "I feel empty inside, losing interest in things I used to enjoy",
                "I'm sad most days, feeling worthless and guilty about everything"
            ],
            'Parkinson\'s Disease': [
                "My hands shake sometimes, especially when I'm trying to write",
                "I notice my movements are getting slower and stiffer",
                "My voice seems quieter lately, people ask me to speak up",
                "I have trouble with balance when walking, shuffling my feet"
            ],
            'Hypothyroidism': [
                "I'm always cold and tired, even after sleeping for hours",
                "I've gained weight recently and my hair is thinning",
                "My skin feels dry and I have trouble concentrating",
                "I feel sluggish and my heart rate seems slow"
            ]
        }
        
        texts = []
        labels = []
        
        # Generate samples for each condition
        for label_idx, (condition, text_samples) in enumerate(sample_texts.items()):
            samples_per_condition = n_samples // len(self.target_labels)
            
            for _ in range(samples_per_condition):
                # Randomly select and modify base text
                base_text = np.random.choice(text_samples)
                
                # Add some variation
                if np.random.random() > 0.5:
                    base_text = base_text + " " + np.random.choice(text_samples)
                
                texts.append(base_text)
                labels.append(label_idx)
        
        return texts, np.array(labels)
    
    def _train_text_model(self):
        """Train the text classification model."""
        try:
            # Generate training data
            texts, labels = self._generate_training_data()
            
            # Extract features for all texts
            text_features = []
            for text in texts:
                features = self.extract_text_features(text)
                if features:
                    # Convert features dict to array
                    feature_vector = [
                        features.get('word_count', 0),
                        features.get('sentence_count', 0),
                        features.get('avg_word_length', 0),
                        features.get('avg_sentence_length', 0),
                        features.get('sentiment_polarity', 0),
                        features.get('sentiment_subjectivity', 0),
                        features.get('question_marks', 0),
                        features.get('exclamation_marks', 0),
                        features.get('uppercase_ratio', 0),
                        features.get('medical_keywords_count', 0),
                        features.get('depression_indicators', 0),
                        features.get('parkinson_indicators', 0),
                        features.get('hypothyroid_indicators', 0),
                        features.get('healthy_indicators', 0),
                        features.get('repetitive_words', 0),
                        features.get('incomplete_sentences', 0),
                        features.get('coherence_score', 0)
                    ]
                    text_features.append(feature_vector)
                else:
                    # Default feature vector if extraction fails
                    text_features.append([0] * 17)
            
            # Train the model
            X = np.array(text_features)
            X_scaled = self.text_scaler.fit_transform(X)
            self.text_classifier.fit(X_scaled, labels)
            
            st.success("Text analysis model trained successfully!")
            
        except Exception as e:
            st.error(f"Error training text model: {str(e)}")
    
    def predict_from_text(self, text: str) -> Dict:
        """Predict medical condition from text analysis."""
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'predicted_class': 'Unknown',
                    'confidence': 0.0,
                    'error': 'No text provided'
                }
            
            # Extract features
            features = self.extract_text_features(text)
            if not features:
                return {
                    'predicted_class': 'Unknown',
                    'confidence': 0.0,
                    'error': 'Feature extraction failed'
                }
            
            # Convert to feature vector
            feature_vector = np.array([
                features.get('word_count', 0),
                features.get('sentence_count', 0),
                features.get('avg_word_length', 0),
                features.get('avg_sentence_length', 0),
                features.get('sentiment_polarity', 0),
                features.get('sentiment_subjectivity', 0),
                features.get('question_marks', 0),
                features.get('exclamation_marks', 0),
                features.get('uppercase_ratio', 0),
                features.get('medical_keywords_count', 0),
                features.get('depression_indicators', 0),
                features.get('parkinson_indicators', 0),
                features.get('hypothyroid_indicators', 0),
                features.get('healthy_indicators', 0),
                features.get('repetitive_words', 0),
                features.get('incomplete_sentences', 0),
                features.get('coherence_score', 0)
            ]).reshape(1, -1)
            
            # Scale features and predict
            feature_vector_scaled = self.text_scaler.transform(feature_vector)
            probabilities = self.text_classifier.predict_proba(feature_vector_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            
            return {
                'predicted_class': self.target_labels[predicted_idx],
                'confidence': float(probabilities[predicted_idx]),
                'all_probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.target_labels, probabilities)
                },
                'text_features': features,
                'modality': 'text_analysis'
            }
            
        except Exception as e:
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'error': f'Prediction error: {str(e)}'
            }
    
    def analyze_speech_to_text(self, audio_file_path: str, language: str = 'en-US') -> Dict:
        """Complete pipeline: speech to text + text analysis + medical prediction."""
        # Step 1: Convert speech to text
        speech_result = self.convert_speech_to_text(audio_file_path, language)
        
        if not speech_result['success']:
            return {
                'success': False,
                'error': speech_result['error'],
                'stage': 'speech_to_text'
            }
        
        # Step 2: Analyze text and predict condition
        text = speech_result['text']
        prediction_result = self.predict_from_text(text)
        
        # Combine results
        return {
            'success': True,
            'transcribed_text': text,
            'speech_to_text_method': speech_result['method'],
            'speech_confidence': speech_result.get('confidence', 'Unknown'),
            'predicted_class': prediction_result['predicted_class'],
            'prediction_confidence': prediction_result['confidence'],
            'all_probabilities': prediction_result.get('all_probabilities', {}),
            'text_features': prediction_result.get('text_features', {}),
            'language': language
        }