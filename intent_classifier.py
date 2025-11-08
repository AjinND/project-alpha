"""
Intent classification to filter actionable commands from transcribed audio
"""
import logging
from typing import List, Dict, Tuple
import requests
import json

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Classify and filter transcribed audio into actionable automation steps"""
    
    def __init__(self, llm_endpoint: str = "http://localhost:11434/api/generate",
                 llm_model: str = "llama3.2:3b"):
        """
        Initialize intent classifier
        
        Args:
            llm_endpoint: LLM API endpoint (Ollama/OpenAI/etc)
            llm_model: Model name (use smaller model for efficiency)
        """
        self.endpoint = llm_endpoint
        self.model = llm_model
        logger.info(f"IntentClassifier initialized with {llm_model}")
    
    def classify_segments(self, segments: List[Tuple[float, str]], 
                         available_fields: List[str]) -> List[Dict]:
        """
        Classify audio segments into actionable vs non-actionable
        
        Args:
            segments: List of (timestamp, text) from Whisper
            available_fields: List of JSON data fields available
            
        Returns:
            List of classified segments with metadata
        """
        classified = []
        
        # Build context for the LLM
        fields_context = ", ".join(available_fields)
        
        for timestamp, text in segments:
            classification = self._classify_single_segment(text, fields_context)
            
            if classification['is_actionable']:
                classified.append({
                    'timestamp': timestamp,
                    'original_text': text,
                    'cleaned_text': classification['cleaned_text'],
                    'intent_type': classification['intent_type'],
                    'confidence': classification['confidence'],
                    'target_element': classification.get('target_element'),
                    'data_field': classification.get('data_field')
                })
                logger.info(f"[{timestamp:.1f}s] V Actionable: {classification['intent_type']} - {text[:50]}")
            else:
                logger.debug(f"[{timestamp:.1f}s] X Filtered: {classification['reason']} - {text[:50]}")
        
        return classified
    
    def _classify_single_segment(self, text: str, fields_context: str) -> Dict:
        """
        Classify a single audio segment
        
        Returns:
            {
                'is_actionable': bool,
                'intent_type': str,  # navigate, click, input, verify, etc.
                'confidence': float,
                'cleaned_text': str,
                'target_element': str,
                'data_field': str or None,
                'reason': str  # if not actionable
            }
        """
        prompt = f"""You are an automation intent classifier. Analyze this transcribed voice command.
            Audio segment: "{text}"
            Available data fields: {fields_context}

            Task: Determine if this is an ACTIONABLE automation step or should be FILTERED OUT.

            Filter out (mark as NOT actionable):
            - Greetings, filler words, thinking noises ("um", "uh", "so", "okay")
            - Explanations or narration ("now I'm going to", "as you can see")
            - Meta-commentary ("this is how you", "let me show you")
            - Corrections or backtracking ("wait", "actually", "no")
            - Incomplete or unclear statements

            Keep (mark as actionable):
            - Direct UI interaction commands ("click submit", "enter email")
            - Navigation commands ("go to", "open page")
            - Data entry actions ("type", "fill in", "select")
            - Verification steps ("check if", "confirm that")

            Return ONLY valid JSON:
            {{
            "is_actionable": true/false,
            "intent_type": "navigate|click|input|select|wait|verify|scroll",
            "confidence": 0.0-1.0,
            "cleaned_text": "simplified command without filler",
            "target_element": "UI element description",
            "data_field": "matching_json_field or null",
            "reason": "why filtered (if not actionable)"
            }}"""

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"  # Force JSON mode if supported
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.warning(f"LLM returned {response.status_code}, defaulting to actionable")
                return self._default_classification(text)
            
            result = response.json()
            ai_response = result.get('response', '{}')
            
            # Clean JSON formatting
            ai_response = ai_response.strip()
            if ai_response.startswith("```json"):
                ai_response = ai_response[7:-3].strip()
            elif ai_response.startswith("```"):
                ai_response = ai_response[3:-3].strip()
            
            classification = json.loads(ai_response)
            
            # Validate required fields
            if not isinstance(classification.get('is_actionable'), bool):
                raise ValueError("Missing or invalid 'is_actionable' field")
            
            return classification
            
        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Classification error: {e}. Defaulting to actionable.")
            return self._default_classification(text)
    
    def _default_classification(self, text: str) -> Dict:
        """Fallback classification when LLM fails"""
        # Simple heuristic fallback
        action_keywords = ['click', 'type', 'enter', 'fill', 'select', 'go', 'open', 
                          'submit', 'press', 'choose', 'input', 'navigate']
        
        text_lower = text.lower()
        is_actionable = any(keyword in text_lower for keyword in action_keywords)
        
        return {
            'is_actionable': is_actionable,
            'intent_type': 'unknown',
            'confidence': 0.5,
            'cleaned_text': text,
            'target_element': None,
            'data_field': None,
            'reason': 'LLM classification failed, using heuristic'
        }
    
    def post_process_segments(self, classified_segments: List[Dict]) -> List[Dict]:
        """
        Post-process classified segments for better quality
        
        - Merge consecutive similar intents
        - Remove duplicates
        - Order verification steps appropriately
        """
        if not classified_segments:
            return []
        
        processed = []
        prev_segment = None
        
        for segment in classified_segments:
            # Skip near-duplicates (within 2 seconds)
            if prev_segment:
                time_diff = segment['timestamp'] - prev_segment['timestamp']
                text_similarity = self._text_similarity(
                    segment['cleaned_text'], 
                    prev_segment['cleaned_text']
                )
                
                if time_diff < 2.0 and text_similarity > 0.8:
                    logger.debug(f"Skipping duplicate: {segment['cleaned_text']}")
                    continue

                if prev_segment and text_similarity > 0.8:
                    if 'username and password' in segment['cleaned_text'].lower():  # Specific filter for redundant
                        continue
            
            processed.append(segment)
            prev_segment = segment
        
        logger.info(f"Post-processing: {len(classified_segments)} -> {len(processed)} segments")
        return processed
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


def integrate_intent_classifier(segments: List[Tuple], json_data: Dict) -> List[Dict]:
    """
    Convenience function to filter audio segments
    
    Args:
        segments: Raw Whisper transcription [(timestamp, text), ...]
        json_data: Available JSON fields
        
    Returns:
        Filtered and classified segments ready for action planning
    """
    classifier = IntentClassifier()
    
    logger.info(f"Classifying {len(segments)} audio segments...")
    classified = classifier.classify_segments(segments, list(json_data.keys()))
    
    logger.info(f"Post-processing {len(classified)} actionable segments...")
    filtered = classifier.post_process_segments(classified)
    
    logger.info(f"Final: {len(filtered)} actionable commands (filtered {len(segments) - len(filtered)})")
    
    return filtered


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulated Whisper output
    segments = [
        (0.5, "Okay, so let me show you how to fill this form"),
        (3.2, "First, click on the email field"),
        (5.8, "Um, and then type the email address"),
        (9.1, "Actually, let me correct that"),
        (11.4, "Enter john@example.com"),
        (14.7, "Now click submit button"),
        (17.2, "And that's how you do it")
    ]
    
    json_data = {"email": "john@example.com", "password": "secret123"}
    
    filtered = integrate_intent_classifier(segments, json_data)
    
    print("\n=== FILTERED ACTIONABLE COMMANDS ===")
    for cmd in filtered:
        print(f"[{cmd['timestamp']:.1f}s] {cmd['intent_type']}: {cmd['cleaned_text']}")