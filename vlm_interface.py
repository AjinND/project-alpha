import numpy as np
import requests
import logging
import json
import time
import base64
import cv2
from io import BytesIO
from PIL import Image
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class VLMInterface:
    """Interface for Vision-Language Model queries."""
    
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model
        self.request_timeout = 300
        logger.info(f"VLM initialized: {model} at {endpoint}")
    
    def query_vlm(self, prompt: str, image: np.ndarray = None, max_retries: int = 3) -> dict:
        """Query VLM with optional image."""
        try:
            # Prepare payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            # Add image if provided
            if image is not None:
                img_base64 = self._encode_image(image)
                if img_base64:
                    payload["images"] = [img_base64]
            
            # Retry logic
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.endpoint,
                        json=payload,
                        timeout=self.request_timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result.get('response', '')
                        
                        if ai_response:
                            return self._parse_json_response(ai_response)
                        else:
                            logger.warning("Empty response from VLM")
                            return {}
                    
                    elif response.status_code == 500:
                        logger.warning(f"VLM 500 error (attempt {attempt+1}/{max_retries})")
                        time.sleep(2 ** attempt)
                        last_error = f"Server error: {response.status_code}"
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        response.raise_for_status()
                
                except requests.exceptions.Timeout:
                    logger.warning(f"VLM timeout (attempt {attempt+1}/{max_retries})")
                    last_error = "Request timeout"
                    time.sleep(2)
                except requests.exceptions.RequestException as e:
                    logger.warning(f"VLM request error (attempt {attempt+1}/{max_retries}): {e}")
                    last_error = str(e)
                    time.sleep(2)
            
            logger.error(f"VLM query failed after {max_retries} attempts: {last_error}")
            return {}
            
        except Exception as e:
            logger.error(f"VLM query failed: {e}", exc_info=True)
            return {}
    
    def locate_element(self, screenshot: np.ndarray, element_description: str,
                      context: Dict = None) -> Optional[Dict]:
        """Use VLM to locate UI element with bounding box."""
        h, w = screenshot.shape[:2]
        
        prompt = f"""Analyze this screenshot ({w}x{h} pixels) and locate the UI element.

            Element to find: "{element_description}"
            Context: {json.dumps(context, indent=2) if context else 'None'}

            Return ONLY valid JSON with this structure:
            {{
            "found": true/false,
            "element_type": "button|input|link|dropdown|checkbox|...",
            "bounding_box": [x1, y1, x2, y2],
            "confidence": 0.0-1.0,
            "description": "visual characteristics"
            }}

            Rules:
            - Coordinates must be in pixels within image bounds (0-{w}, 0-{h})
            - Set found=false if element is not visible
            - Higher confidence (>0.8) for clear matches only"""
        
        result = self.query_vlm(prompt, screenshot)
        
        if result and isinstance(result, dict):
            # Validate and clamp bounding box
            if 'bounding_box' in result:
                bbox = result['bounding_box']
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(w, int(x1)))
                    y1 = max(0, min(h, int(y1)))
                    x2 = max(0, min(w, int(x2)))
                    y2 = max(0, min(h, int(y2)))
                    result['bounding_box'] = [x1, y1, x2, y2]
            
            result.setdefault('confidence', 0.5)
            result.setdefault('found', False)
            
            return result
        
        return None
    
    def verify_action(self, before_img: np.ndarray, after_img: np.ndarray,
                     expected_outcome: str, action_type: str) -> Dict:
        """Verify action succeeded by comparing before/after screenshots."""
        # Calculate pixel change
        diff = cv2.absdiff(before_img, after_img)
        change_percentage = (np.sum(diff > 10) / diff.size) * 100
        
        logger.debug(f"Pixel change: {change_percentage:.2f}%")
        
        # Quick fail for no visual change
        if action_type in ['click', 'input'] and change_percentage < 0.5:
            return {
                'success': False,
                'confidence': 0.3,
                'observations': f'No significant visual change ({change_percentage:.2f}% pixels changed)',
                'error_detected': True,
                'error_message': 'Action did not produce expected visual feedback',
                'pixel_change_pct': change_percentage
            }
        
        # Use VLM for detailed verification
        prompt = f"""Compare these UI states to verify an action.

            Expected outcome: "{expected_outcome}"
            Action type: {action_type}

            Analyze the SECOND (after) screenshot:
            1. Did the expected change occur?
            2. Are there error messages or unexpected states?
            3. Are there loading indicators (action still in progress)?
            4. For input actions: Is there new text in form fields?
            5. For click actions: Did navigation occur or did UI state change?

            Return ONLY valid JSON:
            {{
            "success": true/false,
            "confidence": 0.0-1.0,
            "observations": "what changed or remained",
            "error_detected": true/false,
            "error_message": "error text if found",
            "still_loading": true/false
            }}"""
        
        result = self.query_vlm(prompt, after_img)
        
        if not result:
            return {
                'success': False,
                'confidence': 0.0,
                'observations': 'VLM verification failed',
                'error_detected': False,
                'error_message': '',
                'pixel_change_pct': change_percentage
            }
        
        # Set defaults
        result.setdefault('success', False)
        result.setdefault('confidence', 0.5)
        result.setdefault('observations', '')
        result.setdefault('error_detected', False)
        result.setdefault('error_message', '')
        result.setdefault('still_loading', False)
        result['pixel_change_pct'] = change_percentage
        
        # Adjust confidence based on visual change
        if change_percentage < 1.0 and result['success']:
            result['confidence'] *= 0.5
            result['observations'] += f" (Low visual change: {change_percentage:.2f}%)"
        
        return result
    
    def _encode_image(self, image: np.ndarray) -> Optional[str]:
        """Encode image to base64 string."""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image)
            elif image.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
            else:  # BGR
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize to reduce payload
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return img_base64
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None
    
    def _parse_json_response(self, ai_response: str) -> dict:
        """Parse JSON from AI response with fallback strategies."""
        ai_response = ai_response.strip()
        
        # Remove markdown code blocks
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:-3].strip()
        elif ai_response.startswith("```"):
            ai_response = ai_response[3:-3].strip()
        
        try:
            return json.loads(ai_response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {ai_response[:500]}")
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            return {}