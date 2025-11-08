import logging
import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import JavascriptException
from typing import Dict, Optional, Tuple, List
from visual_memory import VisualMemory
from vlm_interface import VLMInterface
from screenshot_manager import ScreenshotManager
from advanced_ocr_matcher import AdvancedOCRMatcher
from grounding_dion_locator import GroundingDINOLocator
from visual_imitation_system import VisualImitationLearner

logger = logging.getLogger(__name__)


class AutomationEngine:
    """Executes automation actions in browser."""

    def __init__(self, driver: webdriver.Chrome, vlm: VLMInterface,
                 visual_memory: VisualMemory, json_data: Dict,
                 screenshot_mgr: ScreenshotManager = None, imitation_learner: VisualImitationLearner = None):
        self.driver = driver
        self.vlm = vlm
        self.visual_memory = visual_memory
        self.json_data = json_data or {}
        self.screenshot_mgr = screenshot_mgr or ScreenshotManager()
        self.ocr_matcher = AdvancedOCRMatcher()
        self.dino_locator = GroundingDINOLocator()
        self.imitation_learner = imitation_learner or VisualImitationLearner()
        self.max_retries = 1

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
        skip_clear = False
        for attempt in range(self.max_retries):
            try:
                screenshot = self.capture_screenshot()
                self.screenshot_mgr.save_screenshot(screenshot, action_num, "before")

                position = self.locate_element_multi_strategy(screenshot, action_plan, action_num)
                if position is None:
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

                intent = action_plan.get('intent', 'click')
                action_succeeded = False
                
                if intent == 'navigate':
                    url = self._extract_url(action_plan)
                    if url:
                        self.driver.get(url)
                        action_succeeded = True
                elif intent == 'click':
                    action_succeeded = self.click_element_reliable(x, y, action_num)
                elif intent == 'input':
                    data_key = action_plan.get('data_key')
                    if data_key and data_key in self.json_data:
                        value = str(self.json_data[data_key])
                        action_succeeded = self.input_text_reliable(x, y, value, action_num, skip_clear=skip_clear)
                        if action_succeeded:
                            logger.info(f"Entered '{value}' from key '{data_key}'")
                        skip_clear = True
                    else:
                        logger.warning(f"No data found for key '{data_key}'")
                        action_succeeded = False

                if not action_succeeded and attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue

                after_screenshot = self.capture_screenshot()
                self.screenshot_mgr.save_screenshot(after_screenshot, action_num, "after")

                verification = action_plan.get('verify')
                if verification:
                    verified = self.verify_action_with_vlm(after_screenshot, verification)
                    if verified:
                        logger.info("Verification succeeded")
                        return {'status': 'success', 'attempt': attempt + 1}
                    else:
                        logger.warning("Verification failed")
                        if attempt < self.max_retries - 1:
                            time.sleep(2)
                            continue
                else:
                    return {'status': 'success', 'attempt': attempt + 1}

            except Exception as e:
                logger.error(f"Execution error (attempt {attempt + 1}): {e}", exc_info=True)
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
    
    def locate_element_multi_strategy(self, screenshot: np.ndarray, action_plan: Dict, action_num: int) -> Optional[Tuple[int, int]]:
        """Try multiple strategies to find the target element."""
        target_description = action_plan.get('target_description', '')
        context = action_plan.get('context', {})
        logger.debug(f"Locating element: '{target_description}' (context={context})")

        # Strategy 1: Visual imitation memory
        try:
            logger.debug("Strategy 1: Visual imitation memory...")
            visual_memory_obj = action_plan.get('visual_memory')
            if visual_memory_obj:
                loc = self.imitation_learner.find_matching_element(screenshot, visual_memory_obj)
                if loc:
                    cx, cy = loc
                    self.screenshot_mgr.save_element_highlight(screenshot, (cx, cy), action_num, "imitation")
                    logger.info(f"Imitation found at {(cx, cy)}")
                    return (cx, cy)
        except Exception as e:
            logger.warning(f"Visual imitation failed: {e}")

        # Strategy 2: CLIP visual memory
        try:
            logger.debug("Strategy 2: CLIP visual memory...")
            if self.visual_memory:
                candidates = self.visual_memory.extract_candidates(screenshot)
                res = self.visual_memory.find_similar_element(
                    screenshot, target_description, candidates=candidates
                )
                if res:
                    bbox, sim = res
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        self.screenshot_mgr.save_element_highlight(screenshot, center, action_num, f"CLIP:{sim:.2f}")
                        logger.info(f"CLIP found bbox {bbox} sim={sim:.3f}")
                        return center
        except Exception as e:
            logger.warning(f"CLIP matching failed: {e}")

        # Strategy 3: VLM locate
        try:
            logger.debug("Strategy 3: VLM locate...")
            vlm_result = self.vlm.locate_element(screenshot, target_description, context)
            if vlm_result and vlm_result.get('found'):
                bbox = vlm_result.get('bounding_box')
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    self.screenshot_mgr.save_element_highlight(
                        screenshot, center, action_num, 
                        f"VLM:{vlm_result.get('element_type','')}"
                    )
                    logger.info(f"VLM found at {center}")
                    return tuple(center)
        except Exception as e:
            logger.warning(f"VLM failed: {e}")

        # Strategy 4: Grounding DINO
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
                center = best.get('center')
                self.screenshot_mgr.save_element_highlight(
                    screenshot, center, action_num, 
                    f"DINO: {best.get('label')}"
                )
                logger.info(f"DINO found '{best.get('label')}' at {center}")
                return tuple(center)
        except Exception as e:
            logger.warning(f"Grounding DINO failed: {e}")

        # Strategy 5: OCR + Fuzzy Match
        try:
            logger.debug("Strategy 5: OCR + Fuzzy match...")
            ocr_results = self.ocr_matcher.extract_text_with_positions(screenshot)
            match = self.ocr_matcher.fuzzy_match(target_description, ocr_results, threshold=70)
            if match:
                center = match['center']
                self.screenshot_mgr.save_element_highlight(
                    screenshot, center, action_num, 
                    f"OCR: {match.get('text')}"
                )
                logger.info(f"OCR matched at {center}")
                return tuple(center)
        except Exception as e:
            logger.warning(f"OCR matching failed: {e}")

        self.screenshot_mgr.save_screenshot(screenshot, action_num, "not_found", f"target_{target_description}")
        logger.error("All location strategies failed")
        return None

    def click_element_reliable(self, x: int, y: int, action_num: int) -> bool:
        """Reliable clicking with multiple methods and validation."""
        try:
            dpr = float(self.driver.execute_script("return window.devicePixelRatio || 1"))
            viewport_w = int(self.driver.execute_script("return window.innerWidth"))
            viewport_h = int(self.driver.execute_script("return window.innerHeight"))
            client_x = int(round(x / dpr))
            client_y = int(round(y / dpr))
            client_x = max(0, min(client_x, viewport_w - 1))
            client_y = max(0, min(client_y, viewport_h - 1))
        except Exception as e:
            logger.warning(f"Could not compute devicePixelRatio: {e}")
            client_x, client_y = int(x), int(y)

        logger.debug(f"Translated coords ({x},{y}) -> client ({client_x},{client_y})")

        try:
            el = self.driver.execute_script(
                "return document.elementFromPoint(arguments[0], arguments[1]);",
                client_x, client_y
            )
            if not el:
                logger.debug("No element at point - scrolling to bring into view")
                self.driver.execute_script(
                    "window.scrollBy(0, arguments[0] - window.innerHeight/2);",
                    client_y
                )
                time.sleep(0.35)
        except Exception as e:
            logger.debug(f"elementFromPoint check failed: {e}")

        methods = [
            ("ActionChains", lambda: self._click_with_actions(client_x, client_y)),
            ("JavaScript click", lambda: self._click_with_js(client_x, client_y)),
            ("dispatchEvent", lambda: self._click_with_dispatch(client_x, client_y))
        ]

        for method_name, method_func in methods:
            try:
                ok = method_func()
                if ok:
                    logger.info(f"✓ Click succeeded with {method_name} at client ({client_x},{client_y})")
                    time.sleep(0.35)
                    return True
                else:
                    logger.debug(f"{method_name} returned False")
            except Exception as e:
                logger.warning(f"{method_name} raised exception: {e}")

        logger.error("All click methods failed")
        try:
            ss = self.capture_screenshot()
            self.screenshot_mgr.save_screenshot(ss, action_num, "click_failed",
                                                f"click_failed_at_{client_x}_{client_y}")
        except:
            pass
        return False

    def _click_with_actions(self, client_x: int, client_y: int) -> bool:
        """Click using Selenium ActionChains."""
        try:
            rect = self.driver.execute_script("""
                const el = document.elementFromPoint(arguments[0], arguments[1]);
                if (!el) return null;
                const r = el.getBoundingClientRect();
                return {left: r.left, top: r.top, width: r.width, height: r.height};
            """, client_x, client_y)
            
            if not rect:
                logger.debug("No element found for ActionChains click")
                return False

            element = self.driver.execute_script(
                "return document.elementFromPoint(arguments[0], arguments[1]);",
                client_x, client_y
            )
            
            if element is None:
                logger.debug("Could not obtain WebElement for ActionChains")
                return False

            offset_x = int(round(client_x - rect['left']))
            offset_y = int(round(client_y - rect['top']))

            actions = ActionChains(self.driver)
            actions.move_to_element_with_offset(element, offset_x, offset_y).click().perform()
            try:
                actions.reset_actions()
            except:
                pass
            return True
        except Exception as e:
            logger.debug(f"ActionChains click failed: {e}")
            try:
                ActionChains(self.driver).reset_actions()
            except:
                pass
            return False

    def _click_with_js(self, client_x: int, client_y: int) -> bool:
        """Click using JavaScript."""
        try:
            return bool(self.driver.execute_script("""
                const el = document.elementFromPoint(arguments[0], arguments[1]);
                if (el) { 
                    try { 
                        el.click(); 
                        return true; 
                    } catch(e) { 
                        return false;
                    } 
                }
                return false;
            """, client_x, client_y))
        except Exception as e:
            logger.debug(f"_click_with_js exception: {e}")
            return False

    def _click_with_dispatch(self, client_x: int, client_y: int) -> bool:
        """Click using dispatchEvent."""
        try:
            return bool(self.driver.execute_script("""
                const el = document.elementFromPoint(arguments[0], arguments[1]);
                if (el) {
                    ['mousedown', 'mouseup', 'click'].forEach(eventType => {
                        el.dispatchEvent(new MouseEvent(eventType, {
                            bubbles: true, cancelable: true, view: window,
                            clientX: arguments[0], clientY: arguments[1]
                        }));
                    });
                    return true;
                }
                return false;
            """, client_x, client_y))
        except Exception as e:
            logger.debug(f"_click_with_dispatch exception: {e}")
            return False

    def input_text_reliable(self, x: int, y: int, text: str, action_num: int, skip_clear: bool = False) -> bool:
        """Input text using multiple methods with fallback."""
        try:
            # FIXED: Removed is_input parameter
            if not self.click_element_reliable(x, y, action_num):
                return False

            methods = [
                ("Selenium", lambda: self._input_with_selenium(text, skip_clear)),
                ("JS Value", lambda: self._input_with_js_value(x, y, text)),
                ("JS Events", lambda: self._input_with_js_events(x, y, text))
            ]

            for method_name, method_func in methods:
                try:
                    if method_func():
                        time.sleep(0.5)
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

    def _input_with_selenium(self, text: str, skip_clear: bool) -> bool:
        """Input using Selenium send_keys."""
        try:
            active = self.driver.switch_to.active_element
            tag = active.tag_name.upper() if active.tag_name else None
            if tag in ('INPUT', 'TEXTAREA'):
                if not skip_clear:
                    try:
                        active.clear()
                    except:
                        pass
                active.send_keys(text)
                return True
            return False
        except Exception as e:
            logger.debug(f"_input_with_selenium exception: {e}")
            return False

    def _input_with_js_value(self, x: int, y: int, text: str) -> bool:
        """Input using JavaScript value property."""
        try:
            # Convert to client coordinates
            dpr = float(self.driver.execute_script("return window.devicePixelRatio || 1"))
            client_x = int(round(x / dpr))
            client_y = int(round(y / dpr))
            
            escaped = text.replace("'", "\\'")
            return bool(self.driver.execute_script("""
                const el = document.elementFromPoint(arguments[0], arguments[1]);
                if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
                    el.focus();
                    el.value = arguments[2];
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    return true;
                }
                return false;
            """, client_x, client_y, escaped))
        except Exception as e:
            logger.debug(f"_input_with_js_value exception: {e}")
            return False

    def _input_with_js_events(self, x: int, y: int, text: str) -> bool:
        """Input using JavaScript with proper events."""
        try:
            dpr = float(self.driver.execute_script("return window.devicePixelRatio || 1"))
            client_x = int(round(x / dpr))
            client_y = int(round(y / dpr))
            
            escaped = text.replace("'", "\\'")
            return bool(self.driver.execute_script("""
                const el = document.elementFromPoint(arguments[0], arguments[1]);
                if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
                    el.focus();
                    el.value = arguments[2];
                    ['input','change','keyup','keydown'].forEach(evt => {
                        el.dispatchEvent(new Event(evt, { bubbles: true }));
                    });
                    return true;
                }
                return false;
            """, client_x, client_y, escaped))
        except Exception as e:
            logger.debug(f"_input_with_js_events exception: {e}")
            return False

    def _get_input_value(self, x: int, y: int) -> str:
        """Get current value of input field."""
        try:
            dpr = float(self.driver.execute_script("return window.devicePixelRatio || 1"))
            client_x = int(round(x / dpr))
            client_y = int(round(y / dpr))
            
            value = self.driver.execute_script("""
                let el = document.elementFromPoint(arguments[0], arguments[1]);
                if (el.tagName === 'LABEL') {
                    const forId = el.getAttribute('for');
                    if (forId) {
                        el = document.getElementById(forId);
                    }
                }
                if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
                    return el.value;
                }
                return '';
            """, client_x, client_y)
            return value or ""
        except Exception as e:
            logger.debug(f"_get_input_value exception: {e}")
            return ""

    def _extract_url(self, action_plan: Dict) -> Optional[str]:
        """Extract URL from action plan."""
        return action_plan.get('target_url')

    def verify_action_with_vlm(self, screenshot: np.ndarray, verification: Dict) -> bool:
        """Use VLM to verify action outcome."""
        try:
            desc = verification.get('description')
            if not desc:
                return True
            res = self.vlm.verify(screenshot, desc)
            return bool(res)
        except Exception as e:
            logger.warning(f"Verification error: {e}")
            return False

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