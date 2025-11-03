"""
Advanced OCR with fuzzy matching for robust text-based element location
"""
import cv2
import numpy as np
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz
from typing import List, Tuple, Optional
import logging
import torch

logger = logging.getLogger(__name__)

class AdvancedOCRMatcher:
    """Enhanced OCR with fuzzy matching and context awareness"""
     
    def __init__(self, languages=['en']):
        self.ocr = None  # Lazy init for faster startup
        self.languages = languages
        logger.info("AdvancedOCRMatcher initialized (OCR loading deferred)")
    
    def _init_ocr(self):
        if self.ocr is None:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR lazily initialized")
    
    def extract_text_with_positions(self, image: np.ndarray) -> List[dict]:
        """
        Extract all text with precise positions
        
        Returns:
            List of dicts: [{'text': str, 'bbox': [x1,y1,x2,y2], 'confidence': float, 'center': (x,y)}]
        """
        self._init_ocr()
        result = self.ocr.ocr(image, cls=True)
        
        extractions = []
        for line in result[0] if result and result[0] else []:
            bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_info = line[1]  # (text, confidence)
            text = text_info[0]
            confidence = text_info[1]
            
            # Convert bbox to x1,y1,x2,y2
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            extractions.append({
                'text': text,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'center': (int(center[0]), int(center[1]))
            })
        
        return extractions
    
    def fuzzy_match(self, target: str, ocr_results: List[dict], 
                   threshold: int = 75) -> Optional[dict]:
        """
        Find best fuzzy match for target text
        
        Args:
            target: Target text to find
            ocr_results: OCR extraction results
            threshold: Minimum fuzzy match score (0-100)
            
        Returns:
            Best matching result or None
        """
        if not ocr_results:
            return None
        
        # Normalize target
        target_lower = target.lower().strip()
        target_words = set(target_lower.split())
        
        best_match = None
        best_score = 0
        
        for result in ocr_results:
            text = result['text']
            text_lower = text.lower().strip()
            text_words = set(text_lower.split())
            
            # Calculate multiple similarity scores
            scores = []
            
            # 1. Full string similarity
            scores.append(fuzz.ratio(target_lower, text_lower))
            
            # 2. Partial string similarity
            scores.append(fuzz.partial_ratio(target_lower, text_lower))
            
            # 3. Token sort (order-independent)
            scores.append(fuzz.token_sort_ratio(target_lower, text_lower))
            
            # 4. Token set (unique words)
            scores.append(fuzz.token_set_ratio(target_lower, text_lower))
            
            # 5. Word overlap
            if target_words and text_words:
                overlap = len(target_words & text_words) / len(target_words)
                scores.append(overlap * 100)
            
            # 6. Substring match
            if target_lower in text_lower or text_lower in target_lower:
                scores.append(90)
            
            # Take maximum score
            score = max(scores)
            
            # Boost score if confidence is high
            score = score * (0.8 + 0.2 * result['confidence'])
            
            if score > best_score:
                best_score = score
                best_match = result
        
        if best_score >= threshold:
            logger.info(f"Fuzzy matched '{target}' -> '{best_match['text']}' (score: {best_score:.1f})")
            best_match['match_score'] = best_score
            return best_match
        
        logger.warning(f"No match found for '{target}' (best score: {best_score:.1f})")
        return None
    
    def find_text_near_position(self, ocr_results: List[dict], 
                               reference_pos: Tuple[int, int], 
                               max_distance: int = 100) -> List[dict]:
        """Find text elements near a reference position"""
        nearby = []
        ref_x, ref_y = reference_pos
        
        for result in ocr_results:
            cx, cy = result['center']
            distance = np.sqrt((cx - ref_x)**2 + (cy - ref_y)**2)
            
            if distance <= max_distance:
                result['distance'] = distance
                nearby.append(result)
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance'])
        return nearby
    
    def find_by_context(self, target: str, context_words: List[str], 
                       ocr_results: List[dict]) -> Optional[dict]:
        """
        Find element using contextual clues
        
        Args:
            target: Target text to find
            context_words: List of words that should appear near target
            ocr_results: OCR extraction results
        """
        # First, find context elements
        context_positions = []
        for context_word in context_words:
            match = self.fuzzy_match(context_word, ocr_results, threshold=70)
            if match:
                context_positions.append(match['center'])
        
        if not context_positions:
            # No context found, fall back to direct match
            return self.fuzzy_match(target, ocr_results)
        
        # Find target near context
        avg_x = sum(pos[0] for pos in context_positions) / len(context_positions)
        avg_y = sum(pos[1] for pos in context_positions) / len(context_positions)
        
        nearby_results = self.find_text_near_position(
            ocr_results, (int(avg_x), int(avg_y)), max_distance=200
        )
        
        # Match within nearby results
        return self.fuzzy_match(target, nearby_results, threshold=70)
    
    def visualize_ocr(self, image: np.ndarray, ocr_results: List[dict], 
                     output_path: str = "ocr_visualization.png"):
        """Draw OCR results on image"""
        result = image.copy()
        
        for item in ocr_results:
            x1, y1, x2, y2 = item['bbox']
            text = item['text']
            conf = item['confidence']
            
            # Draw box
            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw text
            cv2.putText(result, f"{text} ({conf:.2f})", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(output_path, result)
        return result
    
    def cleanup(self):
        if self.ocr is not None:
            del self.ocr
            self.ocr = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("PaddleOCR resources freed")