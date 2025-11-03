import logging
from typing import List, Dict, Tuple, Optional
from .vlm_interface import VLMInterface
from .visual_memory import VisualMemory
from visual_imitation_system import VisualImitationLearner
import cv2

logger = logging.getLogger(__name__)

class ActionPlanner:
    """Plans actions from video and segments."""
    
    def __init__(self, json_data: Dict, no_imitation: bool = False):
        self.json_data = json_data
        self.vlm = VLMInterface("http://localhost:11434/api/generate", "llama3.2-vision")
        self.visual_memory = VisualMemory()
        self.imitation_learner = VisualImitationLearner() if not no_imitation else None
    
    def parse_video_actions(self, segments: List[Tuple], video_path: str, 
                            intent_classifications: List[Dict] = None) -> List[Dict]:
        """
        Parse video into action plans with imitation learning.
        
        Args:
            segments: List of (timestamp, text) tuples from transcription
            video_path: Path to demonstration video
            intent_classifications: Optional pre-classified intents
            
        Returns:
            List of action plan dictionaries
        """
        # If no intent classifications provided, create basic ones
        if intent_classifications is None:
            intent_classifications = self._create_basic_intents(segments)
        
        # Try imitation learning first
        if self.imitation_learner:
            try:
                logger.info("🎯 Learning from demonstration video with visual imitation...")
                interaction_memories = self.imitation_learner.learn_from_demonstration(
                    video_path, segments, intent_classifications
                )
                
                if interaction_memories:
                    logger.info(f"✓ Learned {len(interaction_memories)} visual interaction patterns")
                    return self._build_action_plans_from_memories(interaction_memories)
                else:
                    logger.warning("Imitation learning found no interactions")
            except Exception as e:
                logger.error(f"Imitation learning failed: {e}", exc_info=True)
        
        # Fallback to VLM parsing
        logger.info("Using VLM fallback parsing...")
        return self._fallback_vlm_parsing(segments, video_path, intent_classifications)
    
    def _create_basic_intents(self, segments: List[Tuple]) -> List[Dict]:
        """Create basic intent classifications from transcription."""
        intent_classifications = []
        
        for timestamp, audio_text in segments:
            text_lower = audio_text.lower()
            
            # Simple keyword-based intent detection
            if any(word in text_lower for word in ['click', 'press', 'tap', 'button']):
                intent = 'click'
            elif any(word in text_lower for word in ['type', 'enter', 'input', 'fill']):
                intent = 'input'
            elif any(word in text_lower for word in ['go', 'open', 'navigate', 'visit']):
                intent = 'navigate'
            elif any(word in text_lower for word in ['wait', 'pause', 'sleep']):
                intent = 'wait'
            else:
                intent = 'click'  # Default
            
            intent_classifications.append({
                'timestamp': timestamp,
                'intent_type': intent,
                'target_element': audio_text,
                'data_field': self._guess_data_field(audio_text),
                'cleaned_text': audio_text
            })
        
        return intent_classifications
    
    def _guess_data_field(self, audio_text: str) -> Optional[str]:
        """Guess which JSON field matches the audio command."""
        text_lower = audio_text.lower()
        
        # Direct field name match
        for key in self.json_data.keys():
            key_lower = key.lower()
            if key_lower in text_lower or text_lower in key_lower:
                return key
        
        # Common field patterns
        field_patterns = {
            'email': ['email', 'e-mail', 'mail address'],
            'password': ['password', 'pass', 'pwd'],
            'username': ['username', 'user', 'login'],
            'name': ['name', 'full name'],
            'phone': ['phone', 'mobile', 'telephone'],
            'address': ['address', 'street'],
            'city': ['city', 'town'],
            'zip': ['zip', 'postal', 'postcode'],
        }
        
        for json_key in self.json_data.keys():
            json_key_lower = json_key.lower()
            for pattern, keywords in field_patterns.items():
                if pattern in json_key_lower:
                    if any(kw in text_lower for kw in keywords):
                        return json_key
        
        return None
    
    def _build_action_plans_from_memories(self, memories: List) -> List[Dict]:
        """Convert interaction memories to action plans."""
        action_plans = []
        
        for i, memory in enumerate(memories):
            action_plan = {
                'timestamp': memory.timestamp,
                'audio': memory.audio_command,
                'intent': memory.intent,
                'target_description': memory.audio_command,
                'data_key': memory.data_key,
                'expected_outcome': memory.expected_outcome or f"{memory.intent} completed",
                'confidence': 0.9,  # High confidence from imitation learning
                
                # Visual memory for execution phase
                'visual_memory': memory,
                'interaction_point_demo': memory.interaction_point,
                'reference_frame': memory.surrounding_context_crop,
                
                # Metadata
                'element_type': memory.element_type,
                'ocr_text': memory.ocr_text,
                'spatial_features': memory.relative_position
            }
            
            action_plans.append(action_plan)
            
            # Store in visual memory for backward compatibility
            if memory.surrounding_context_crop is not None:
                self.visual_memory.store_reference(
                    memory.surrounding_context_crop,
                    memory.audio_command,
                    {
                        'timestamp': memory.timestamp,
                        'intent': memory.intent,
                        'interaction_point': memory.interaction_point
                    }
                )
        
        return action_plans
    
    def _fallback_vlm_parsing(self, segments: List[Tuple], video_path: str,
                              intent_classifications: List[Dict]) -> List[Dict]:
        """Fallback VLM-based parsing when imitation learning fails."""
        action_plans = []
        
        for i, ((timestamp, audio_text), intent_data) in enumerate(zip(segments, intent_classifications)):
            logger.info(f"Parsing action {i+1}/{len(segments)}: {audio_text}")
            
            try:
                frame = self._extract_frame(video_path, timestamp)
                
                prompt = f"""Analyze this UI screenshot and voice command to determine the automation action.

                    Voice command: "{audio_text}"
                    Available data fields: {list(self.json_data.keys())}
                    Intent type: {intent_data.get('intent_type', 'unknown')}

                    Determine:
                    1. Action type: navigate, click, input, select, wait, verify, or scroll
                    2. Target element description (semantic, not exact text)
                    3. Data field mapping (which JSON key to use, if applicable)
                    4. Expected outcome after action

                    Return ONLY valid JSON:
                    {{
                    "intent": "action_type",
                    "target_description": "semantic description of UI element",
                    "data_key": "json_field_name or null",
                    "expected_outcome": "what should happen",
                    "confidence": 0.0-1.0,
                    "element_type": "button|input|link|..."
                    }}"""
                
                action_plan = self.vlm.query_vlm(prompt, frame)
                
                if action_plan and isinstance(action_plan, dict):
                    # Add missing fields
                    action_plan.setdefault('confidence', 0.5)
                    action_plan.setdefault('intent', intent_data.get('intent_type', 'click'))
                    action_plan.setdefault('target_description', audio_text)
                    
                    if action_plan.get('confidence', 0) > 0.4:
                        action_plan['timestamp'] = timestamp
                        action_plan['audio'] = audio_text
                        action_plan['reference_frame'] = frame
                        action_plans.append(action_plan)
                        
                        # Store reference
                        if action_plan.get('target_description'):
                            self.visual_memory.store_reference(
                                frame,
                                action_plan['target_description'],
                                {'timestamp': timestamp, 'intent': action_plan['intent']}
                            )
                    else:
                        logger.warning(f"Low confidence ({action_plan.get('confidence')}) for: {audio_text}")
                else:
                    logger.warning(f"Failed to parse: {audio_text}")
                    
            except Exception as e:
                logger.error(f"Error parsing action {i+1}: {e}")
                continue
        
        return action_plans
    
    def _extract_frame(self, video_path: str, timestamp: float) -> cv2.Mat:
        """Extract frame from video at timestamp."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to extract frame at {timestamp}s from {video_path}")
        
        return frame