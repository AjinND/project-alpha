"""
Grounding DINO integration for zero-shot element detection
"""
import cv2
import numpy as np
from PIL import Image
import torch
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
import logging

logger = logging.getLogger(__name__)


class GroundingDINOLocator:
    """Zero-shot object detection with text prompts"""
    
    def __init__(self, config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 checkpoint_path="weights/groundingdino_swint_ogc.pth"):
        self.model = None
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Grounding DINO init deferred on {self.device}")
    
    def _init_model(self):
        if self.model is None:
            try:
                self.model = load_model(self.config_path, self.checkpoint_path)
                logger.info("Grounding DINO lazily loaded")
            except Exception as e:
                logger.error(f"Failed to load Grounding DINO model: {e}")
                raise
    
    def locate_element(self, image: np.ndarray, text_prompt: str, 
                      box_threshold: float = 0.35, text_threshold: float = 0.25) -> list:
        """
        Locate UI elements using natural language descriptions.
        
        Args:
            image: Screenshot as numpy array (BGR)
            text_prompt: Natural language description
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            List of detections with bbox, center, confidence, and label
        """
        try:
            self._init_model()
            
            # FIXED: Convert BGR to RGB and create PIL Image properly
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # FIXED: Preprocess image for Grounding DINO
            # Instead of load_image (which expects file path), do preprocessing manually
            from groundingdino.util.inference import preprocess_image
            
            # Transform image to tensor
            transform = self._get_transform()
            image_tensor, _ = transform(pil_image, None)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self.device
                )
            
            if boxes is None or len(boxes) == 0:
                logger.debug(f"No detections for prompt: '{text_prompt}'")
                return []
            
            # Convert boxes from normalized to pixel coordinates
            h, w = image.shape[:2]
            boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
            
            # Convert to xyxy format
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            
            detections = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                
                # Clamp to image bounds
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                
                confidence = logit.item()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'confidence': confidence,
                    'label': phrase
                })
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.debug(f"Found {len(detections)} detections for '{text_prompt}'")
            return detections
            
        except Exception as e:
            logger.error(f"Grounding DINO locate_element failed: {e}", exc_info=True)
            return []
    
    def _get_transform(self):
        """Get the transform pipeline for Grounding DINO."""
        import torchvision.transforms as T
        
        # Standard Grounding DINO transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        def transform(image, target):
            # Resize to standard size
            image = image.resize((800, 800))
            image = normalize(image)
            return image, target
        
        return transform
    
    def visualize_detections(self, image: np.ndarray, detections: list, 
                           output_path: str = "detections.png"):
        """Draw bounding boxes on image."""
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
        logger.info(f"Saved visualization to {output_path}")
        return result
    
    def cleanup(self):
        """Free resources."""
        if self.model is not None:
            del self.model
            self.model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Grounding DINO resources freed")