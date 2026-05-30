"""
Domain-Specific Feature Engineering Module

Implements advanced feature engineering for enthymeme detection:
1. Argument structure detection (claim, evidence, conclusion patterns)
2. Rhetorical device identification (appeals to emotion, authority, etc.)
3. Implicit language markers (hedging, presupposition triggers)
4. Sentiment and polarity features
5. Debate domain features (vaccine/immigration specific)
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LINGUISTIC PATTERNS
# ============================================================================

class LinguisticPatterns:
    """Domain-specific linguistic patterns for enthymeme detection"""
    
    # Implicit premise markers (things often assumed)
    IMPLICIT_PREMISE_MARKERS = {
        'generalization': [
            r'\b(all|every|most|many|some|no one)\b',
            r'\b(always|never|ever|rarely)\b',
        ],
        'causal': [
            r'\b(because|since|as|due to|caused by|leads to|results in)\b',
            r'\b(therefore|thus|hence|consequently|so)\b',
        ],
        'comparison': [
            r'\b(than|like|unlike|similar|different)\b',
            r'\b(better|worse|same|opposite)\b',
        ],
        'assumption': [
            r'\b(must|should|ought|need|require)\b',
            r'\b(if|unless|provided that|in case)\b',
        ]
    }
    
    # Implicit conclusion markers
    IMPLICIT_CONCLUSION_MARKERS = {
        'evaluation': [
            r'\b(good|bad|right|wrong|better|worse)\b',
            r'\b(safe|dangerous|effective|useless)\b',
        ],
        'recommendation': [
            r'\b(should|must|ought to|need to|recommend)\b',
            r'\b(must not|should not|cannot)\b',
        ],
        'certainty': [
            r'\b(definitely|certainly|obviously|clearly|evidently)\b',
            r'\b(truly|really|actually|indeed)\b',
        ],
        'emotional': [
            r'\b(tragic|wonderful|horrifying|amazing|terrible)\b',
            r'\b(absurd|ridiculous|unacceptable|outrageous)\b',
        ]
    }
    
    # Rhetorical devices
    RHETORICAL_DEVICES = {
        'appeal_to_emotion': [
            r'\b(fear|afraid|scared|worried|concerned)\b',
            r'\b(love|hate|disgusting|beautiful)\b',
        ],
        'appeal_to_authority': [
            r'\b(doctor|scientist|expert|research|study|professor)\b',
            r'\b(government|official|agency|organization)\b',
        ],
        'appeal_to_common_sense': [
            r'\b(obvious|clear|common sense|everyone knows)\b',
            r'\b(naturally|obviously|of course)\b',
        ],
        'strawman': [
            r'\b(they say|critics claim|opponents argue)\b',
            r'\b(supposedly|allegedly|claims that)\b',
        ],
        'loaded_language': [
            r'\b(agenda|propaganda|censorship|brainwash)\b',
            r'\b(coverup|conspiracy|corruption)\b',
        ]
    }
    
    # Hedging and presupposition
    HEDGING_MARKERS = [
        r'\b(maybe|perhaps|probably|possibly|might|could|may)\b',
        r'\b(seem|appear|suggest|indicate|tend)\b',
        r'\b(somewhat|quite|rather|relatively)\b',
    ]
    
    PRESUPPOSITION_TRIGGERS = [
        r'\b(also|too|still|even)\b',
        r'\b(again|anymore|already|yet)\b',
        r'\b(stop|quit|continue|start)\b',
    ]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class DomainFeatureExtractor:
    """Extract domain-specific features from tweets"""
    
    def __init__(self):
        self.patterns = LinguisticPatterns()
    
    def extract_linguistic_markers(self, text: str) -> Dict[str, float]:
        """Extract linguistic marker features"""
        text_lower = text.lower()
        features = {}
        
        # Implicit premise markers
        for marker_type, patterns in self.patterns.IMPLICIT_PREMISE_MARKERS.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower))
            features[f'premise_marker_{marker_type}'] = count
        
        # Implicit conclusion markers
        for marker_type, patterns in self.patterns.IMPLICIT_CONCLUSION_MARKERS.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower))
            features[f'conclusion_marker_{marker_type}'] = count
        
        return features
    
    def extract_rhetorical_devices(self, text: str) -> Dict[str, float]:
        """Extract rhetorical device features"""
        text_lower = text.lower()
        features = {}
        
        for device_type, patterns in self.patterns.RHETORICAL_DEVICES.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower))
            features[f'rhetorical_{device_type}'] = count
        
        return features
    
    def extract_hedging_presupposition(self, text: str) -> Dict[str, float]:
        """Extract hedging and presupposition features"""
        text_lower = text.lower()
        
        hedging_count = 0
        for pattern in self.patterns.HEDGING_MARKERS:
            hedging_count += len(re.findall(pattern, text_lower))
        
        presupposition_count = 0
        for pattern in self.patterns.PRESUPPOSITION_TRIGGERS:
            presupposition_count += len(re.findall(pattern, text_lower))
        
        return {
            'hedging_markers': hedging_count,
            'presupposition_triggers': presupposition_count
        }
    
    def extract_argument_structure(self, text: str) -> Dict[str, float]:
        """Extract argument structure features"""
        text_lower = text.lower()
        
        # Claim indicators
        claim_count = len(re.findall(r'\b(claim|argue|assert|maintain|believe)\b', text_lower))
        
        # Evidence indicators
        evidence_count = len(re.findall(r'\b(because|evidence|proof|reason|fact|data|study)\b', text_lower))
        
        # Conclusion indicators
        conclusion_count = len(re.findall(r'\b(therefore|thus|hence|conclude|so|implies)\b', text_lower))
        
        # Counter-argument
        counter_count = len(re.findall(r'\b(but|however|yet|although|while|despite)\b', text_lower))
        
        return {
            'claim_indicators': claim_count,
            'evidence_indicators': evidence_count,
            'conclusion_indicators': conclusion_count,
            'counter_argument': counter_count
        }
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract basic linguistic features"""
        words = text.split()
        
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'average_word_length': np.mean([len(w) for w in words]) if words else 0,
            'question_mark': text.count('?'),
            'exclamation_mark': text.count('!'),
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        }
        
        return features
    
    def extract_sentiment_polarity(self, text: str) -> Dict[str, float]:
        """Extract sentiment and polarity indicators"""
        text_lower = text.lower()
        
        # Simple sentiment lexicons
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'safe', 'effective',
            'benefit', 'beneficial', 'protection', 'strong', 'better', 'best'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'dangerous', 'harmful', 'risk',
            'death', 'died', 'injury', 'side effect', 'worse', 'worst'
        ]
        
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        return {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'sentiment_polarity': positive_count - negative_count
        }
    
    def extract_domain_specific(self, text: str) -> Dict[str, float]:
        """Extract vaccine/immigration debate specific features"""
        text_lower = text.lower()
        
        # Vaccine-related terms
        vaccine_terms = ['vaccine', 'vaccinated', 'vaccination', 'immunize', 'covid', 'pfizer', 'moderna', 'jj']
        vaccine_count = sum(1 for term in vaccine_terms if term in text_lower)
        
        # Health policy terms
        health_terms = ['mandate', 'policy', 'requirement', 'law', 'government', 'regulation', 'rule']
        health_count = sum(1 for term in health_terms if term in text_lower)
        
        # Personal freedom terms
        freedom_terms = ['freedom', 'liberty', 'choice', 'choice', 'forced', 'force', 'mandate']
        freedom_count = sum(1 for term in freedom_terms if term in text_lower)
        
        # Scientific terms
        science_terms = ['study', 'research', 'data', 'evidence', 'scientific', 'prove', 'trial']
        science_count = sum(1 for term in science_terms if term in text_lower)
        
        return {
            'vaccine_terms': vaccine_count,
            'health_policy_terms': health_count,
            'freedom_terms': freedom_count,
            'scientific_terms': science_count
        }
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all features"""
        features = {}
        
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_linguistic_markers(text))
        features.update(self.extract_rhetorical_devices(text))
        features.update(self.extract_hedging_presupposition(text))
        features.update(self.extract_argument_structure(text))
        features.update(self.extract_sentiment_polarity(text))
        features.update(self.extract_domain_specific(text))
        
        return features

# ============================================================================
# FEATURE COMBINATION WITH TFIDF
# ============================================================================

class EnhancedFeatureExtractor:
    """Combine domain features with TF-IDF"""
    
    def __init__(self):
        self.domain_extractor = DomainFeatureExtractor()
        self.tfidf_vectorizer = None
        self.feature_names = None
    
    def fit_transform(self, texts: List[str]):
        """Fit and transform texts to feature matrix"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Domain features
        domain_features = []
        for text in texts:
            features = self.domain_extractor.extract_all_features(text)
            domain_features.append(list(features.values()))
        
        domain_matrix = np.array(domain_features)
        
        # Normalize domain features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        domain_matrix = scaler.fit_transform(domain_matrix)
        
        # Combine
        combined = np.hstack([
            tfidf_matrix.toarray(),
            domain_matrix
        ])
        
        return combined
    
    def transform(self, texts: List[str]):
        """Transform texts to feature matrix"""
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        domain_features = []
        for text in texts:
            features = self.domain_extractor.extract_all_features(text)
            domain_features.append(list(features.values()))
        
        domain_matrix = np.array(domain_features)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        domain_matrix = scaler.fit_transform(domain_matrix)
        
        combined = np.hstack([
            tfidf_matrix.toarray(),
            domain_matrix
        ])
        
        return combined

# ============================================================================
# DEMO AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("DOMAIN-SPECIFIC FEATURE ENGINEERING DEMO")
    print("="*80)
    
    extractor = DomainFeatureExtractor()
    
    # Example tweets
    examples = [
        "We should mandate vaccines for everyone",  # Implicit conclusion/premise
        "The vaccine is safe and effective",  # Implicit conclusion
        "If everyone gets vaccinated, herd immunity is achieved",  # Implicit premise
        "I got vaccinated today",  # No implicit argument
    ]
    
    for tweet in examples:
        print(f"\nTweet: {tweet}")
        features = extractor.extract_all_features(tweet)
        
        # Print key features
        print("Key features:")
        important_features = {k: v for k, v in features.items() if v > 0}
        for feature, value in sorted(important_features.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature}: {value}")
    
    print("\n✅ Domain feature extraction working!")
