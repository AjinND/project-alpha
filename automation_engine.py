import logging
import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import JavascriptException
from typing import Dict, Optional, Tuple, List
from .visual_memory import VisualMemory
from .vlm_interface import VLMInterface
from .screenshot_manager import ScreenshotManager
from advanced_ocr_matcher import AdvancedOCRMatcher
from grounding_dion_locator import GroundingDINOLocator
from visual_imitation_system import VisualImitationLearner

logger = logging.getLogger(__name__)

class AutomationEngine:
    """Executes automation actions in browser."""
    
    def __init__(self, driver: webdriver.Chrome, vlm: VLMInterface, 
                 visual_memory: VisualMemory, json_data: Dict, 
                 screenshot_mgr: ScreenshotManager, 
                 imitation_learner: VisualImitationLearner = None):
        self.driver = driver
        self.vlm = vlm
        self.visual_memory = visual_memory
        self.json_data = json_data
        self.screenshot_mgr = screenshot_mgr
        self.imitation_learner = imitation_learner
        self.ocr_matcher = AdvancedOCRMatcher()
        self.dino_locator = GroundingDINOLocator()
        self.max_retries = 3
    
    def capture_screenshot(self) -> np.ndarray:
        """Capture current browser screenshot as OpenCV image."""
        try:
            png_data = self.driver.get_screenshot_as_png()
            return cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            raise
    
    def execute_action(self, action_plan: Dict, action_num: int) -> Dict:
        """Execute a single action with retries and verification."""
        intent = action_plan['intent']
        target_desc = action_plan.get('target_description', action_plan.get('audio', ''))
        
        logger.info(f"Executing: {intent} on '{target_desc}'")
        
        for attempt in range(self.max_retries):
            try:
                # Capture BEFORE screenshot
                screenshot_before = self.capture_screenshot()
                self.screenshot_mgr.save_screenshot(
                    screenshot_before, action_num, "before",
                    f"{intent}_{target_desc[:20]}"
                )
                
                action_succeeded = False
                
                # Handle different action types
                if intent == 'navigate':
                    url = self._extract_url(action_plan)
                    if url:
                        logger.info(f"Navigating to: {url}")
                        self.driver.get(url)
                        time.sleep(2)
                        action_succeeded = True
                    else:
                        logger.warning("No valid URL found, skipping navigation")
                        action_succeeded = True  # Already on page
                
                elif intent in ['click', 'input']:
                    # Locate element
                    position = self.locate_element_multi_strategy(
                        screenshot_before, target_desc, action_plan, action_num
                    )
                    
                    if not position:
                        logger.error(f"Element not found: {target_desc}")
                        if attempt < self.max_retries - 1:
                            time.sleep(2)
                            continue
                        return {
                            'status': 'failed',
                            'reason': 'Element not found',
                            'attempt': attempt + 1
                        }
                    
                    x, y = position
                    logger.info(f"Element located at ({x}, {y})")
                    
                    # Perform action
                    if intent == 'click':
                        action_succeeded = self.click_element_reliable(x, y, action_num)
                    else:  # input
                        data_key = action_plan.get('data_key')
                        if data_key and data_key in self.json_data:
                            value = str(self.json_data[data_key])
                            action_succeeded = self.input_text_reliable(x, y, value, action_num)
                            if action_succeeded:
                                logger.info(f"Entered '{value}' from key '{data_key}'")
                        else:
                            logger.warning(f"No data found for key '{data_key}'")
                            action_succeeded = False
                    
                    if not action_succeeded and attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                
                elif intent == 'wait':
                    wait_time = action_plan.get('duration', 2)
                    logger.info(f"Waiting {wait_time}s")
                    time.sleep(wait_time)
                    action_succeeded = True
                
                else:
                    logger.warning(f"Unknown intent: {intent}")
                    action_succeeded = False
                
                # Capture AFTER screenshot
                time.sleep(1.5)
                screenshot_after = self.capture_screenshot()
                self.screenshot_mgr.save_screenshot(
                    screenshot_after, action_num, "after", f"{intent}_completed"
                )
                
                # Verify action
                verification = self._verify_action(
                    screenshot_before, screenshot_after,
                    action_plan.get('expected_outcome', ''),
                    intent, position if intent in ['click', 'input'] else None
                )
                
                verification_passed = (
                    action_succeeded and
                    verification.get('success') and
                    verification.get('confidence', 0) > 0.65 and
                    not verification.get('error_detected', False)
                )
                
                # Check pixel change for interactive actions
                if intent in ['click', 'input']:
                    pixel_change = verification.get('pixel_change_pct', 0)
                    min_change = 1.0 if intent == 'click' else 0.3
                    
                    if pixel_change < min_change:
                        verification_passed = False
                        logger.warning(
                            f"Verification FAILED: Visual change too low "
                            f"({pixel_change:.2f}% < {min_change}%)"
                        )
                
                if verification_passed:
                    logger.info(f"✓ Action verified successfully (attempt {attempt + 1})")
                    return {
                        'status': 'success',
                        'attempt': attempt + 1,
                        'verification': verification,
                        'position': position if intent in ['click', 'input'] else None
                    }
                else:
                    logger.warning(
                        f"Verification failed (attempt {attempt + 1}): "
                        f"{verification.get('observations')}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                
            except Exception as e:
                logger.error(f"Execution error (attempt {attempt + 1}): {e}", exc_info=True)
                
                # Capture error screenshot
                try:
                    error_screenshot = self.capture_screenshot()
                    self.screenshot_mgr.save_screenshot(
                        error_screenshot, action_num, "error",
                        f"exception_{type(e).__name__}"
                    )
                except:
                    pass
                
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
        
        return {
            'status': 'failed',
            'reason': 'Max retries exceeded',
            'attempt': self.max_retries
        }
    
    def locate_element_multi_strategy(self, screenshot: np.ndarray,
                                     target_description: str, context: Dict,
                                     action_num: int) -> Optional[Tuple[int, int]]:
        """Multi-strategy element location with imitation learning priority."""
        self.screenshot_mgr.save_screenshot(
            screenshot, action_num, "before",
            f"Locating: {target_description[:30]}"
        )
        
        # STRATEGY 1: Visual Imitation Learning (HIGHEST PRIORITY)
        if self.imitation_learner and 'visual_memory' in context:
            try:
                logger.info("🎯 Strategy 1: Visual Imitation Learning")
                memory = context['visual_memory']
                
                position = self.imitation_learner.find_matching_element(
                    screenshot, memory, search_strategy='multi_scale'
                )
                
                if position:
                    x, y = position
                    logger.info(f"✓ Imitation Learning found element at ({x}, {y})")
                    self.screenshot_mgr.save_element_highlight(
                        screenshot, position, action_num,
                        f"ImitationLearning: {target_description[:20]}"
                    )
                    return position
            except Exception as e:
                logger.warning(f"Imitation learning search failed: {e}")
        
        # STRATEGY 2: VLM Location
        try:
            logger.debug("Strategy 2: VLM location...")
            vlm_result = self.vlm.locate_element(screenshot, target_description, context)
            if vlm_result and vlm_result.get('found') and vlm_result.get('confidence', 0) > 0.7:
                bbox = vlm_result['bounding_box']
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                self.screenshot_mgr.save_element_highlight(
                    screenshot, center, action_num, f"VLM: {target_description[:20]}"
                )
                logger.info(f"VLM located '{target_description}' at {center}")
                return center
        except Exception as e:
            logger.warning(f"VLM location failed: {e}")
        
        # STRATEGY 3: CLIP Similarity
        try:
            logger.debug("Strategy 3: CLIP similarity...")
            candidates = self.visual_memory.extract_candidates(screenshot)
            clip_match = self.visual_memory.find_similar_element(
                screenshot, target_description, candidates
            )
            if clip_match:
                bbox, similarity = clip_match
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                self.screenshot_mgr.save_element_highlight(
                    screenshot, center, action_num, f"CLIP: {target_description[:20]}"
                )
                logger.info(f"CLIP match at {center} (sim: {similarity:.3f})")
                return center
        except Exception as e:
            logger.warning(f"CLIP matching failed: {e}")
        
        # STRATEGY 4: Grounding DINO
        try:
            logger.debug("Strategy 4: Grounding DINO...")
            dino_prompt = target_description
            if 'element_type' in context:
                dino_prompt = f"{context['element_type']} {dino_prompt}"
            
            detections = self.dino_locator.locate_element(
                screenshot, dino_prompt, box_threshold=0.3, text_threshold=0.25
            )
            if detections:
                best = detections[0]
                center = best['center']
                self.screenshot_mgr.save_element_highlight(
                    screenshot, center, action_num, f"DINO: {best['label']}"
                )
                logger.info(f"DINO found '{best['label']}' at {center}")
                return center
        except Exception as e:
            logger.warning(f"Grounding DINO failed: {e}")
        
        # STRATEGY 5: OCR + Fuzzy Match
        try:
            logger.debug("Strategy 5: OCR + Fuzzy match...")
            ocr_results = self.ocr_matcher.extract_text_with_positions(screenshot)
            match = self.ocr_matcher.fuzzy_match(target_description, ocr_results, threshold=70)
            if match:
                center = match['center']
                self.screenshot_mgr.save_element_highlight(
                    screenshot, center, action_num, f"OCR: {match['text']}"
                )
                logger.info(f"OCR matched '{match['text']}' at {center}")
                return center
        except Exception as e:
            logger.warning(f"OCR matching failed: {e}")
        
        self.screenshot_mgr.save_screenshot(
            screenshot, action_num, "error", "All_strategies_failed"
        )
        logger.error("All location strategies failed")
        return None
    
    def click_element_reliable(self, x: int, y: int, action_num: int) -> bool:
        """Reliable clicking with multiple methods and validation."""
        # Validate coordinates
        try:
            viewport_width = self.driver.execute_script("return window.innerWidth")
            viewport_height = self.driver.execute_script("return window.innerHeight")
            
            if not (0 <= x <= viewport_width and 0 <= y <= viewport_height):
                logger.error(
                    f"Click coordinates ({x}, {y}) outside viewport "
                    f"({viewport_width}x{viewport_height})"
                )
                
                if y > viewport_height:
                    scroll_amount = y - (viewport_height // 2)
                    self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                    time.sleep(0.5)
                    y = viewport_height // 2
                    logger.info(f"Scrolled and adjusted to ({x}, {y})")
        except Exception as e:
            logger.warning(f"Could not validate viewport: {e}")
        
        methods = [
            ("ActionChains", lambda: self._click_with_actions(x, y)),
            ("JavaScript click", lambda: self._click_with_js(x, y)),
            ("dispatchEvent", lambda: self._click_with_dispatch(x, y))
        ]
        
        for method_name, method_func in methods:
            try:
                if method_func():
                    logger.info(f"✓ Click succeeded with {method_name}")
                    time.sleep(0.5)
                    return True
            except Exception as e:
                logger.warning(f"✗ {method_name} failed: {e}")
        
        logger.error("All click methods failed")
        return False
    
    def _click_with_actions(self, x: int, y: int) -> bool:
        """Click using Selenium ActionChains."""
        try:
            actions = ActionChains(self.driver)
            actions.move_by_offset(x, y).click().perform()
            actions.reset_actions()
            return True
        except:
            try:
                ActionChains(self.driver).reset_actions()
            except:
                pass
            return False
    
    def _click_with_js(self, x: int, y: int) -> bool:
        """Click using JavaScript elementFromPoint."""
        try:
            return self.driver.execute_script(f"""
                const el = document.elementFromPoint({x}, {y});
                if (el) {{ el.click(); return true; }}
                return false;
            """)
        except JavascriptException:
            return False
    
    def _click_with_dispatch(self, x: int, y: int) -> bool:
        """Click using JavaScript dispatchEvent."""
        try:
            return self.driver.execute_script(f"""
                const el = document.elementFromPoint({x}, {y});
                if (el) {{
                    ['mousedown', 'mouseup', 'click'].forEach(eventType => {{
                        el.dispatchEvent(new MouseEvent(eventType, {{
                            bubbles: true, cancelable: true, view: window,
                            clientX: {x}, clientY: {y}
                        }}));
                    }});
                    return true;
                }}
                return false;
            """)
        except:
            return False
    
    def input_text_reliable(self, x: int, y: int, text: str, action_num: int) -> bool:
        """Reliable text input with validation."""
        try:
            # Validate and click first
            if not self.click_element_reliable(x, y, action_num):
                logger.error("Failed to click input field")
                return False
            
            time.sleep(0.5)
            
            # Try multiple input methods
            methods = [
                ("JS with events", lambda: self._input_with_js_events(x, y, text)),
                ("Selenium send_keys", lambda: self._input_with_selenium(text)),
                ("JS value", lambda: self._input_with_js_value(x, y, text))
            ]
            
            for method_name, method_func in methods:
                try:
                    if method_func():
                        time.sleep(0.5)
                        
                        # Verify input
                        entered_text = self._get_input_value(x, y)
                        if entered_text and (
                            text.lower() in entered_text.lower() or
                            entered_text.lower() in text.lower()
                        ):
                            logger.info(f"✓ Input succeeded with {method_name}")
                            return True
                except Exception as e:
                    logger.warning(f"✗ {method_name} failed: {e}")
            
            logger.error("All input methods failed")
            return False
        except Exception as e:
            logger.error(f"Input failed: {e}", exc_info=True)
            return False
    
    def _input_with_selenium(self, text: str) -> bool:
        """Input using Selenium send_keys."""
        try:
            active = self.driver.switch_to.active_element
            active.clear()
            active.send_keys(text)
            return True
        except:
            return False
    
    def _input_with_js_value(self, x: int, y: int, text: str) -> bool:
        """Input using JavaScript value property."""
        try:
            escaped = text.replace("'", "\\'")
            return self.driver.execute_script(f"""
                const el = document.elementFromPoint({x}, {y});
                if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {{
                    el.value = '{escaped}';
                    return true;
                }}
                return false;
            """)
        except:
            return False
    
    def _input_with_js_events(self, x: int, y: int, text: str) -> bool:
        """Input using JavaScript with proper events."""
        try:
            escaped = text.replace("'", "\\'")
            return self.driver.execute_script(f"""
                const el = document.elementFromPoint({x}, {y});
                if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {{
                    el.value = '{escaped}';
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }}
                return false;
            """)
        except:
            return False
    
    def _get_input_value(self, x: int, y: int) -> str:
        """Get current value of input field."""
        try:
            value = self.driver.execute_script(f"""
                const el = document.elementFromPoint({x}, {y});
                if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {{
                    return el.value;
                }}
                return '';
            """)
            return value or ""
        except:
            return ""
    
    def _extract_url(self, action_plan: Dict) -> Optional[str]:
        """Extract URL from action plan."""
        url = action_plan.get('target_url')
        if url and url.startswith('http'):
            return url
        
        target_desc = action_plan.get('target_description', '').lower()
        
        if 'localhost' in target_desc:
            import re
            match = re.search(r'localhost[:\s]*(\d+)', target_desc)
            if match:
                return f'http://localhost:{match.group(1)}'
            return 'http://localhost:3000'
        
        return None
    
    def _verify_action(self, before_img: np.ndarray, after_img: np.ndarray,
                      expected_outcome: str, action_type: str,
                      position: Optional[Tuple[int, int]]) -> Dict:
        """Verify action succeeded by checking visual changes."""
        # Check pixel changes
        diff = cv2.absdiff(before_img, after_img)
        change_percentage = (np.sum(diff > 10) / diff.size) * 100
        
        logger.debug(f"Pixel change: {change_percentage:.2f}%")
        
        # Minimal change detection
        if action_type in ['click', 'input'] and change_percentage < 0.5:
            logger.warning(f"Minimal visual change ({change_percentage:.2f}%)")
            return {
                'success': False,
                'confidence': 0.3,
                'observations': f'No significant visual change',
                'error_detected': True,
                'error_message': 'Action did not produce expected visual feedback',
                'pixel_change_pct': change_percentage
            }
        
        # Use VLM for verification if available
        try:
            vlm_result = self.vlm.verify_action(
                before_img, after_img, expected_outcome, action_type
            )
            vlm_result['pixel_change_pct'] = change_percentage
            return vlm_result
        except:
            # Fallback to pixel-based verification
            return {
                'success': change_percentage > 0.5,
                'confidence': min(change_percentage / 10, 0.9),
                'observations': f'{change_percentage:.2f}% pixels changed',
                'error_detected': False,
                'error_message': '',
                'pixel_change_pct': change_percentage
            }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.ocr_matcher.cleanup()
        except:
            pass
        
        try:
            self.dino_locator.cleanup()
        except:
            pass
        
        try:
            self.visual_memory.cleanup()
        except:
            pass