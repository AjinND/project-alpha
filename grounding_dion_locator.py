"""
Grounding DINO integration for zero-shot element detection
Install: pip install groundingdino-py
"""
import cv2
import numpy as np
from PIL import Image
import torch
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops

class GroundingDINOLocator:
    """Zero-shot object detection with text prompts"""
    
    def __init__(self, config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 checkpoint_path="weights/groundingdino_swint_ogc.pth"):
        self.model = None  # Lazy
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Grounding DINO init deferred on {self.device}")
    
    def _init_model(self):
        if self.model is None:
            self.model = load_model(self.config_path, self.checkpoint_path)
            print("Grounding DINO lazily loaded")
    
    def locate_element(self, image: np.ndarray, text_prompt: str, 
                      box_threshold: float = 0.35, text_threshold: float = 0.25) -> list:
        """
        Locate UI elements using natural language descriptions
        
        Args:
            image: Screenshot as numpy array (BGR)
            text_prompt: Natural language description (e.g., "email input field", "submit button")
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            List of detections: [(bbox, confidence, label), ...]
        """
        self._init_model()
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Run inference
        boxes, logits, phrases = predict(
            model=self.model,
            image=pil_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert boxes from normalized to pixel coordinates
        h, w = image.shape[:2]
        boxes = boxes * torch.tensor([w, h, w, h])
        
        # Convert to xyxy format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        
        detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            confidence = logit.item()
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'confidence': confidence,
                'label': phrase
            })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: list, 
                           output_path: str = "detections.png"):
        """Draw bounding boxes on image"""
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            text = f"{label}: {conf:.2f}"
            cv2.putText(result, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, result)
        return result
    
    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Grounding DINO resources freed")

# Usage example
if __name__ == "__main__":
    locator = GroundingDINOLocator()
    
    # Load screenshot
    screenshot = cv2.imread("browser_screenshot.png")
    
    # Locate elements with natural language
    detections = locator.locate_element(
        screenshot, 
        "email input field. username textbox. login button. submit button"
    )
    
    print(f"Found {len(detections)} elements:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['label']} at {det['center']} (conf: {det['confidence']:.3f})")
    
    # Visualize
    locator.visualize_detections(screenshot, detections)