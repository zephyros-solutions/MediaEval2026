"""
Ollama Integration Module

Provides easy interface for using local LLMs via Ollama for:
1. Classification (replacing or augmenting transformer models)
2. Proposition generation
3. Model management and setup

Models recommended for MediaEval 2026:
- mistral (7B): Best balance for both tasks
- neural-chat (7B): Good for understanding and classification
- llama2 (7B/13B): All-purpose, good baseline
- dolphin-mixtral (45B): Best quality if compute available

Installation:
    curl https://ollama.ai/install.sh | sh
    
Running Ollama:
    ollama serve
    
Pulling models:
    ollama pull mistral
    ollama pull neural-chat
    ollama pull llama2
"""

from typing import Optional, Dict, List
import config

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class OllamaConfig:
    """Configuration for Ollama integration"""

    _BASE_URLS = ("http://host.docker.internal:11434", "http://localhost:11434")

    @staticmethod
    def _detect():
        import socket
        for url in OllamaConfig._BASE_URLS:
            try:
                s = socket.create_connection(
                    url.replace("http://", "").split(":")[:2], timeout=1
                )
                s.close()
                return url
            except OSError:
                continue
        return OllamaConfig._BASE_URLS[-1]

    BASE_URL = None  # set below at module level
    
    # Model recommendations
    MODELS = {
        'mistral': {
            'size': '7B',
            'speed': 'Fast',
            'quality': 'High',
            'best_for': ['classification', 'generation'],
            'params': '7 billion',
            'memory': '~4GB'
        },
        'neural-chat': {
            'size': '7B',
            'speed': 'Fast',
            'quality': 'High',
            'best_for': ['classification'],
            'params': '7 billion',
            'memory': '~4GB'
        },
        'llama2': {
            'size': '7B/13B',
            'speed': 'Medium/Slow',
            'quality': 'High',
            'best_for': ['classification', 'generation'],
            'params': '7-13 billion',
            'memory': '~4-8GB'
        },
        'dolphin-mixtral': {
            'size': '45B',
            'speed': 'Slow',
            'quality': 'Very High',
            'best_for': ['generation'],
            'params': '45 billion',
            'memory': '~24GB'
        }
    }

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

# Module-level Ollama URL detection
OllamaConfig.BASE_URL = OllamaConfig._detect()

class OllamaClient:
    """Client for interacting with Ollama"""
    
    def __init__(self, base_url: str = OllamaConfig.BASE_URL, timeout: int = 120):
        if not REQUESTS_AVAILABLE:
            print("⚠️  Warning: 'requests' module not available. Install with: pip install requests")
        self.base_url = base_url
        self.timeout = timeout
        self._available_models = None
    
    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        if not REQUESTS_AVAILABLE:
            print("⚠️  Cannot check connection without 'requests' module")
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def get_available_models(self, refresh: bool = False) -> List[str]:
        """Get list of available models"""
        if not REQUESTS_AVAILABLE:
            print("⚠️  Cannot fetch models without 'requests' module")
            return []
        
        if self._available_models and not refresh:
            return self._available_models
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'].split(':')[0] for m in response.json().get('models', [])]
                self._available_models = models
                return models
        except Exception as e:
            print(f"Error fetching models: {e}")
        
        return []
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 100,
        stream: bool = False
    ) -> str:
        """Generate text using specified model"""
        if not REQUESTS_AVAILABLE:
            print("⚠️  Cannot generate without 'requests' module")
            return None
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': stream,
                    'temperature': temperature,
                    'top_p': top_p,
                    'num_predict': num_predict
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                print(f"Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Generation error: {e}")
            return None

# ============================================================================
# TASK 1: CLASSIFICATION WITH OLLAMA
# ============================================================================

class OllamaClassifier:
    """Classify tweets using Ollama"""
    
    CLASSIFICATION_PROMPT_TEMPLATE = """Analyze this tweet carefully. Does it contain an implicit argument that is NOT explicitly stated?

An IMPLICIT PREMISE is an assumption or unstated starting point.
An IMPLICIT CONCLUSION is an unstated ending or inference.
NONE means the tweet is just a factual statement with no implicit argument.

Tweet: "{tweet}"

Respond ONLY with one word: none, premise, or conclusion"""
    
    def __init__(self, model: str = 'mistral', client: Optional[OllamaClient] = None):
        self.model = model
        self.client = client or OllamaClient()
    
    def classify(self, tweet: str) -> Dict:
        """Classify a single tweet.

        Uses a heuristic confidence estimate based on raw response length.
        """
        prompt = self.CLASSIFICATION_PROMPT_TEMPLATE.format(tweet=tweet)

        response = self.client.generate(
            self.model,
            prompt,
            temperature=0.2,
            num_predict=5,
        )

        if not response:
            return {"label": "none", "confidence": 0.0, "label_idx": 0, "raw": None}

        response_lower = response.lower().strip()
        label = "none"
        for key in config.LABEL_TO_ID.keys():
            if response_lower.startswith(key):
                label = key
                break

        return {
            "label": label,
            "label_idx": config.LABEL_TO_ID[label],
            "confidence": self._estimate_confidence(response),
            "raw": response,
        }

    @staticmethod
    def _estimate_confidence(raw_response: str) -> float:
        """Heuristic confidence from raw response length."""
        resp_len = len(raw_response.split())
        if resp_len <= 1:
            return 0.95
        elif resp_len <= 3:
            return 0.85
        else:
            return 0.7
    
    def classify_batch(self, tweets: List[str], show_progress: bool = True) -> Dict:
        """Classify multiple tweets"""
        predictions = []
        
        iterator = tweets
        if show_progress:
            from tqdm import tqdm
            import sys as _sys
            iterator = tqdm(tweets, desc="Classifying", file=_sys.stdout)
        
        for tweet in iterator:
            result = self.classify(tweet)
            predictions.append(result)
        
        # Aggregate results
        hard_predictions = [p['label_idx'] for p in predictions]
        labels_array = [p['label'] for p in predictions]
        
        return {
            'hard_predictions': hard_predictions,
            'labels': labels_array,
            'detailed_results': predictions,
            'distribution': {
                label: sum(1 for l in labels_array if l == label)
                for label in config.LABEL_TO_ID.keys()
            }
        }

# ============================================================================
# TASK 2: GENERATION WITH OLLAMA
# ============================================================================

class OllamaGenerator:
    """Generate propositions using Ollama"""
    
    GENERATION_PROMPTS = {
        'premise': """Generate a short implicit premise (assumption) for the following tweet. 
The premise should be what is assumed but not explicitly stated.
Keep it concise (15-25 words).

Tweet: "{tweet}"

Implicit premise:""",
        
        'conclusion': """Generate a short implicit conclusion for the following tweet.
The conclusion should be what is implied but not explicitly stated.
Keep it concise (15-25 words).

Tweet: "{tweet}"

Implicit conclusion:"""
    }
    
    def __init__(self, model: str = 'mistral', client: Optional[OllamaClient] = None):
        self.model = model
        self.client = client or OllamaClient()
    
    def generate_proposition(self, tweet: str, label: str) -> str:
        """Generate a proposition for a tweet"""
        if label not in self.GENERATION_PROMPTS:
            return None
        
        prompt = self.GENERATION_PROMPTS[label].format(tweet=tweet)
        
        response = self.client.generate(
            self.model,
            prompt,
            temperature=0.7,
            num_predict=50
        )
        
        return response if response else None
    
    def generate_batch(
        self,
        tweets: List[str],
        labels: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """Generate propositions for multiple tweets"""
        results = []
        
        iterator = zip(tweets, labels)
        if show_progress:
            from tqdm import tqdm
            import sys as _sys
            iterator = tqdm(list(zip(tweets, labels)), desc="Generating", file=_sys.stdout)
        
        for tweet, label in iterator:
            if label == 'none':
                proposition = None
            else:
                proposition = self.generate_proposition(tweet, label)
            
            results.append({
                'tweet': tweet,
                'label': label,
                'generated_proposition': proposition
            })
        
        return results

# ============================================================================
# SETUP UTILITIES
# ============================================================================

def print_model_info():
    """Print information about available models"""
    print("="*80)
    print("OLLAMA MODELS FOR MEDIAEVAL 2026")
    print("="*80)
    
    for model_name, info in OllamaConfig.MODELS.items():
        print(f"\n{model_name.upper()} ({info['size']})")
        print(f"  Speed: {info['speed']}")
        print(f"  Quality: {info['quality']}")
        print(f"  Parameters: {info['params']}")
        print(f"  Memory: {info['memory']}")
        print(f"  Best for: {', '.join(info['best_for'])}")

def setup_ollama():
    """Print setup instructions for Ollama"""
    print("="*80)
    print("OLLAMA SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n1. Install Ollama:")
    print("   curl https://ollama.ai/install.sh | sh")
    
    print("\n2. Pull recommended model:")
    print("   ollama pull mistral")
    
    print("\n3. Start Ollama server (in separate terminal):")
    print("   ollama serve")
    
    print("\n4. Test connection:")
    print("   ollama list")
    
    print("\nNote: Ollama will download models on first use.")
    print("First run may take a few minutes depending on internet speed.")

def test_ollama_connection():
    """Test connection to Ollama"""
    client = OllamaClient()
    
    print("\nTesting Ollama connection...")
    if client.check_connection():
        print("✅ Connected to Ollama!")
        models = client.get_available_models()
        print(f"Available models: {', '.join(models)}")
        return True
    else:
        print("❌ Could not connect to Ollama")
        print("   Make sure Ollama is running: ollama serve")
        setup_ollama()
        return False

# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print_model_info()
    
    # Test connection
    if not test_ollama_connection():
        print("\nExiting: Ollama not available")
        exit(1)
    
    # Example usage
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    
    client = OllamaClient()
    
    # Classification example
    print("\n1. Classification with Ollama:")
    classifier = OllamaClassifier(model='mistral', client=client)
    
    example_tweets = [
        "We should mandate vaccines for everyone",
        "The vaccine is safe and effective",
        "I got vaccinated today"
    ]
    
    print("\nClassifying example tweets:")
    for tweet in example_tweets:
        result = classifier.classify(tweet)
        print(f"  Tweet: {tweet[:50]}...")
        print(f"  Result: {result['label']} (confidence: {result['confidence']:.2f})")
    
    # Generation example
    print("\n2. Generation with Ollama:")
    generator = OllamaGenerator(model='mistral', client=client)
    
    print("\nGenerating propositions:")
    for tweet, label in zip(example_tweets[:2], ['premise', 'conclusion']):
        proposition = generator.generate_proposition(tweet, label)
        print(f"  Tweet: {tweet[:40]}...")
        print(f"  Generated {label}: {proposition}")
    
    print("\n✅ Ollama integration working!")
