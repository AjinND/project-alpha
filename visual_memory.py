import logging
import cv2
import torch
import clip
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class VisualMemory:
    """Stores and matches UI elements using CLIP embeddings."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.element_database = {}
        logger.info(f"Visual memory initialized on {self.device} (CLIP loading deferred)")
    
    def _init_clip(self):
        if self.model is None:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP lazily loaded")
    
    def store_reference(self, image_crop: np.ndarray, label: str, metadata: dict):
        """Store reference UI element from training video."""
        self._init_clip()
        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        self.element_database[label] = {
            'embedding': embedding.cpu(),
            'metadata': metadata
        }
        logger.debug(f"Stored reference for '{label}'")
    
    def match_element(self, live_crop: np.ndarray, label: str) -> float:
        """Match live crop to stored reference."""
        if label not in self.element_database:
            return 0.0
        self._init_clip()
        pil_image = Image.fromarray(cv2.cvtColor(live_crop, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            live_embedding = self.model.encode_image(image_input)
            live_embedding /= live_embedding.norm(dim=-1, keepdim=True)
        ref_embedding = self.element_database[label]['embedding'].to(self.device)
        similarity = (ref_embedding @ live_embedding.T).item()
        return similarity
    
    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Visual memory resources cleaned up")