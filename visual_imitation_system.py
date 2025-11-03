"""
Enhanced Visual Imitation Learning System
==========================================
Learns UI interactions from demonstration video by tracking actual cursor movements
and visual context, then replicates those interactions during automation.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import clip
from PIL import Image
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)


@dataclass
class InteractionMemory:
    """Stores a learned interaction from the demonstration video"""
    timestamp: float
    audio_command: str
    intent: str  # 'click', 'input', 'scroll'
    
    # Visual context (what the system should look for)
    target_element_crop: np.ndarray  # The actual element you interacted with
    surrounding_context_crop: np.ndarray  # Larger area around the element
    clip_embedding: torch.Tensor  # Semantic representation
    
    # Spatial metadata
    interaction_point: Tuple[int, int]  # Exact x,y where you clicked/typed
    element_bbox: Tuple[int, int, int, int]  # Bounding box of the element
    relative_position: Dict  # Position relative to screen edges, center, etc.
    
    # Visual features for robust matching
    visual_descriptors: Dict  # Color, texture, edge patterns around the element
    ocr_text: str  # Text inside or near the element
    element_type: str  # 'button', 'input_field', 'link', etc.
    
    # Data to input (if applicable)
    data_key: Optional[str] = None
    expected_outcome: str = ""


class CursorTracker:
    """
    Tracks cursor/mouse movements in the demonstration video
    Detects where the user actually clicked or typed
    """
    
    def __init__(self):
        self.prev_frame = None
        self.motion_threshold = 20  # Pixels of movement to detect interaction
        self.click_detection_method = 'motion_pause'
    
    def detect_interaction_point(self, 
                                 video_path: str, 
                                 timestamp: float,
                                 window_seconds: float = 2.0) -> Optional[Tuple[int, int]]:
        """
        Analyze video around timestamp to find where user actually interacted
        
        Strategy:
        1. Track cursor movement (find the moving white/black arrow)
        2. Detect click moment (cursor stops + visual feedback)
        3. Return exact coordinates
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int((timestamp - window_seconds/2) * fps)
        end_frame = int((timestamp + window_seconds/2) * fps)
        
        cursor_positions = []
        motion_scores = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Method 1: Detect cursor by finding the white arrow
            cursor_pos = self._find_cursor_arrow(frame)
            
            # Method 2: Track motion/optical flow
            if self.prev_frame is not None:
                motion_intensity = self._calculate_motion(self.prev_frame, frame)
                motion_scores.append(motion_intensity)
            
            if cursor_pos:
                cursor_positions.append((frame_num, cursor_pos))
            
            self.prev_frame = frame.copy()
        
        cap.release()
        
        # Find the moment of interaction (where cursor stops or clicks)
        if cursor_positions:
            interaction_point = self._identify_click_moment(cursor_positions, motion_scores)
            logger.info(f"Detected interaction at: {interaction_point}")
            return interaction_point
        
        logger.warning("Could not detect cursor/interaction point")
        return None
    
    def _find_cursor_arrow(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the cursor arrow in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for cursor-like shapes (small, high contrast regions)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cursor_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Cursor-sized
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 1.5:  # Arrow-ish shape
                    cursor_candidates.append((x + w//2, y + h//2))
        
        if cursor_candidates:
            # Return the topmost-leftmost (typical cursor tip position)
            return min(cursor_candidates, key=lambda p: p[0] + p[1])
        
        return None
    
    def _calculate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate motion intensity between frames"""
        diff = cv2.absdiff(prev_frame, curr_frame)
        motion = np.mean(diff)
        return motion
    
    def _identify_click_moment(self, 
                               cursor_positions: List[Tuple[int, Tuple[int, int]]], 
                               motion_scores: List[float]) -> Tuple[int, int]:
        """Identify the exact frame/position where click occurred"""
        if not cursor_positions:
            return None
        
        # Find moments where cursor stopped (velocity near zero)
        velocities = []
        for i in range(1, len(cursor_positions)):
            frame1, (x1, y1) = cursor_positions[i-1]
            frame2, (x2, y2) = cursor_positions[i]
            velocity = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            velocities.append((frame2, (x2, y2), velocity))
        
        # Find the longest pause (likely the click moment)
        if velocities:
            pauses = [v for v in velocities if v[2] < 5]  # Nearly stationary
            if pauses:
                # Return position where cursor paused longest
                return pauses[len(pauses)//2][1]  # Middle of pause
        
        # Fallback: return center position
        return cursor_positions[len(cursor_positions)//2][1]


class ContextualElementExtractor:
    """
    Extracts visual context around the interaction point
    Creates multiple crops at different scales for robust matching
    """
    
    def extract_element_context(self, 
                                frame: np.ndarray, 
                                interaction_point: Tuple[int, int],
                                intent: str) -> Dict:
        """Extract visual context at multiple scales"""
        x, y = interaction_point
        h, w = frame.shape[:2]
        
        # Define crop sizes based on intent
        if intent == 'click':
            tight_size = 60
            medium_size = 150
            wide_size = 300
        elif intent == 'input':
            tight_size = 80
            medium_size = 200
            wide_size = 400
        else:
            tight_size = 80
            medium_size = 200
            wide_size = 400
        
        crops = {}
        
        # Tight crop (the element itself)
        x1 = max(0, x - tight_size//2)
        y1 = max(0, y - tight_size//2)
        x2 = min(w, x + tight_size//2)
        y2 = min(h, y + tight_size//2)
        crops['tight_crop'] = frame[y1:y2, x1:x2].copy()
        crops['tight_bbox'] = (x1, y1, x2, y2)
        
        # Medium crop (element + surroundings)
        x1 = max(0, x - medium_size//2)
        y1 = max(0, y - medium_size//2)
        x2 = min(w, x + medium_size//2)
        y2 = min(h, y + medium_size//2)
        crops['medium_crop'] = frame[y1:y2, x1:x2].copy()
        crops['medium_bbox'] = (x1, y1, x2, y2)
        
        # Wide crop (full context)
        x1 = max(0, x - wide_size//2)
        y1 = max(0, y - wide_size//2)
        x2 = min(w, x + wide_size//2)
        y2 = min(h, y + wide_size//2)
        crops['wide_crop'] = frame[y1:y2, x1:x2].copy()
        crops['wide_bbox'] = (x1, y1, x2, y2)
        
        # Extract spatial features
        crops['spatial_features'] = self._extract_spatial_features(
            frame, interaction_point, crops
        )
        
        return crops
    
    def _extract_spatial_features(self, 
                                  frame: np.ndarray, 
                                  point: Tuple[int, int],
                                  crops: Dict) -> Dict:
        """Extract layout/positional features for robust matching"""
        h, w = frame.shape[:2]
        x, y = point
        
        return {
            'relative_x': x / w,
            'relative_y': y / h,
            'quadrant': self._get_quadrant(x, y, w, h),
            'distance_from_edges': {
                'left': x,
                'right': w - x,
                'top': y,
                'bottom': h - y
            },
            'nearby_edges': self._detect_nearby_edges(crops['medium_crop']),
            'dominant_colors': self._get_dominant_colors(crops['tight_crop'])
        }
    
    def _get_quadrant(self, x: int, y: int, w: int, h: int) -> str:
        """Determine which screen quadrant the point is in"""
        if x < w/2 and y < h/2:
            return 'top-left'
        elif x >= w/2 and y < h/2:
            return 'top-right'
        elif x < w/2 and y >= h/2:
            return 'bottom-left'
        else:
            return 'bottom-right'
    
    def _detect_nearby_edges(self, crop: np.ndarray) -> Dict:
        """Detect edge patterns around the element"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'edge_density': edge_density,
            'has_strong_border': edge_density > 0.1
        }
    
    def _get_dominant_colors(self, crop: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get top 3 dominant colors in the crop"""
        from sklearn.cluster import KMeans
        
        pixels = crop.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]


class VisualImitationLearner:
    """Main system that learns interactions from video and replicates them"""
    
    def __init__(self):
        self.cursor_tracker = CursorTracker()
        self.context_extractor = ContextualElementExtractor()
        
        # Initialize CLIP for semantic understanding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Storage for learned interactions
        self.interaction_memories: List[InteractionMemory] = []
        
        logger.info(f"Visual Imitation Learner initialized on {self.device}")
    
    def learn_from_demonstration(self, 
                                 video_path: str,
                                 audio_segments: List[Tuple[float, str]],
                                 intent_classifications: List[Dict]) -> List[InteractionMemory]:
        """Learn interactions from demonstration video"""
        memories = []
        
        for i, (timestamp, audio_text) in enumerate(audio_segments):
            intent_data = intent_classifications[i]
            intent = intent_data['intent_type']
            
            logger.info(f"Learning interaction {i+1}: '{audio_text}' ({intent})")
            
            # Step 1: Find where user actually clicked/typed
            interaction_point = self.cursor_tracker.detect_interaction_point(
                video_path, timestamp
            )
            
            if not interaction_point:
                logger.warning(f"Could not detect interaction point for: {audio_text}")
                continue
            
            # Step 2: Extract frame at that moment
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.warning(f"Could not extract frame at {timestamp}s")
                continue
            
            # Step 3: Extract visual context at multiple scales
            context = self.context_extractor.extract_element_context(
                frame, interaction_point, intent
            )
            
            # Step 4: Create CLIP embedding for semantic matching
            clip_embedding = self._create_clip_embedding(context['medium_crop'])
            
            # Step 5: Extract OCR text if present
            from advanced_ocr_matcher import AdvancedOCRMatcher
            ocr_matcher = AdvancedOCRMatcher()
            ocr_results = ocr_matcher.extract_text_with_positions(context['medium_crop'])
            ocr_text = " ".join([r['text'] for r in ocr_results[:3]])  # Top 3 text items
            
            # Step 6: Create interaction memory
            memory = InteractionMemory(
                timestamp=timestamp,
                audio_command=audio_text,
                intent=intent,
                target_element_crop=context['tight_crop'],
                surrounding_context_crop=context['medium_crop'],
                clip_embedding=clip_embedding,
                interaction_point=interaction_point,
                element_bbox=context['tight_bbox'],
                relative_position=context['spatial_features'],
                visual_descriptors=context['spatial_features'],
                ocr_text=ocr_text,
                element_type=intent_data.get('target_element', 'unknown'),
                data_key=intent_data.get('data_field'),
                expected_outcome=intent_data.get('cleaned_text', '')
            )
            
            memories.append(memory)
            logger.info(f"✓ Learned interaction at ({interaction_point[0]}, {interaction_point[1]})")
        
        self.interaction_memories = memories
        logger.info(f"Total interactions learned: {len(memories)}")
        return memories
    
    def _create_clip_embedding(self, image_crop: np.ndarray) -> torch.Tensor:
        """Create CLIP embedding for visual semantic matching"""
        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu()
    
    def find_matching_element(self, 
                             live_screenshot: np.ndarray,
                             memory: InteractionMemory,
                             search_strategy: str = 'multi_scale') -> Optional[Tuple[int, int]]:
        """Find the matching element in a live screenshot during automation"""
        h, w = live_screenshot.shape[:2]
        
        candidates = []
        
        # Strategy 1: Sliding window CLIP matching
        logger.debug("Searching with CLIP semantic matching...")
        clip_candidates = self._clip_sliding_window_search(
            live_screenshot, memory
        )
        candidates.extend(clip_candidates)
        
        # Strategy 2: OCR-guided search
        if memory.ocr_text.strip():
            logger.debug(f"Searching for OCR text: '{memory.ocr_text}'")
            ocr_candidates = self._ocr_guided_search(
                live_screenshot, memory
            )
            candidates.extend(ocr_candidates)
        
        # Strategy 3: Spatial position heuristic
        logger.debug("Trying spatial position heuristic...")
        spatial_candidate = self._spatial_heuristic_search(
            live_screenshot, memory
        )
        if spatial_candidate:
            candidates.append(spatial_candidate)
        
        # Rank candidates and return best match
        if candidates:
            best_candidate = max(candidates, key=lambda c: c['confidence'])
            
            if best_candidate['confidence'] > 0.7:
                logger.info(f"Found matching element with {best_candidate['confidence']:.2%} confidence")
                logger.info(f"Location: ({best_candidate['x']}, {best_candidate['y']})")
                logger.info(f"Strategy: {best_candidate['strategy']}")
                return (best_candidate['x'], best_candidate['y'])
        
        logger.warning("No confident match found")
        return None
    
    def _clip_sliding_window_search(self, 
                                   screenshot: np.ndarray,
                                   memory: InteractionMemory) -> List[Dict]:
        """Use CLIP embeddings to find semantically similar regions"""
        h, w = screenshot.shape[:2]
        ref_embedding = memory.clip_embedding.to(self.device)
        
        # Get crop size from memory
        crop_h, crop_w = memory.surrounding_context_crop.shape[:2]
        
        candidates = []
        stride = min(crop_h, crop_w) // 4  # 75% overlap
        
        for y in range(0, h - crop_h + 1, stride):
            for x in range(0, w - crop_w + 1, stride):
                crop = screenshot[y:y+crop_h, x:x+crop_w]
                
                # Create embedding for this crop
                crop_embedding = self._create_clip_embedding(crop)
                crop_embedding = crop_embedding.to(self.device)
                
                # Calculate similarity
                similarity = (ref_embedding @ crop_embedding.T).item()
                
                if similarity > 0.65:  # Threshold for candidates
                    center_x = x + crop_w // 2
                    center_y = y + crop_h // 2
                    
                    candidates.append({
                        'x': center_x,
                        'y': center_y,
                        'confidence': similarity,
                        'strategy': 'clip_semantic'
                    })
        
        return candidates
    
    def _ocr_guided_search(self,
                          screenshot: np.ndarray,
                          memory: InteractionMemory) -> List[Dict]:
        """Use OCR to find text-based elements"""
        from advanced_ocr_matcher import AdvancedOCRMatcher
        from fuzzywuzzy import fuzz
        
        ocr_matcher = AdvancedOCRMatcher()
        ocr_results = ocr_matcher.extract_text_with_positions(screenshot)
        
        candidates = []
        target_text = memory.ocr_text.lower().strip()
        
        for ocr_item in ocr_results:
            detected_text = ocr_item['text'].lower().strip()
            
            # Fuzzy match
            similarity = fuzz.ratio(target_text, detected_text) / 100.0
            
            if similarity > 0.7:
                x, y = ocr_item['center']
                candidates.append({
                    'x': x,
                    'y': y,
                    'confidence': similarity * 0.9,  # Slightly lower weight
                    'strategy': 'ocr_text'
                })
        
        return candidates
    
    def _spatial_heuristic_search(self,
                                 screenshot: np.ndarray,
                                 memory: InteractionMemory) -> Optional[Dict]:
        """Use relative spatial position as a fallback"""
        h, w = screenshot.shape[:2]
        
        # Get relative position from memory
        rel_x = memory.relative_position['relative_x']
        rel_y = memory.relative_position['relative_y']
        
        # Calculate approximate position
        approx_x = int(rel_x * w)
        approx_y = int(rel_y * h)
        
        # Lower confidence since this is just a guess
        return {
            'x': approx_x,
            'y': approx_y,
            'confidence': 0.5,
            'strategy': 'spatial_heuristic'
        }