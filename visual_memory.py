import logging
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional

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
            try:
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root="./clip_cache")
                self.model.eval()
                logger.info("CLIP lazily loaded")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                raise

    def store_reference(self, image_crop: np.ndarray, label: str, metadata: dict):
        """Store reference UI element from training video."""
        try:
            self._init_clip()
            pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
                embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-10)
            
            self.element_database[label] = {
                'embedding': embedding.cpu(),
                'metadata': metadata
            }
            logger.debug(f"Stored reference for '{label}' (db size={len(self.element_database)})")
        except Exception as e:
            logger.error(f"Failed to store reference for '{label}': {e}")

    def match_element(self, live_crop: np.ndarray, label: str) -> float:
        """Match live crop to stored reference."""
        if label not in self.element_database:
            return 0.0
        
        try:
            self._init_clip()
            pil_image = Image.fromarray(cv2.cvtColor(live_crop, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                live_embedding = self.model.encode_image(image_input)
                live_embedding = live_embedding / (live_embedding.norm(dim=-1, keepdim=True) + 1e-10)
            
            ref_embedding = self.element_database[label]['embedding'].to(self.device)
            
            # FIXED: Ensure proper matrix multiplication
            similarity = float((ref_embedding @ live_embedding.T).item())
            similarity = max(0.0, (similarity + 1.0) / 2.0)
            
            return similarity
        except Exception as e:
            logger.error(f"Match element failed for '{label}': {e}")
            return 0.0

    def extract_candidates(self,
                           screenshot: np.ndarray,
                           crop_size: int = 128,
                           stride: int = 64,
                           max_candidates: int = 400) -> List[Dict]:
        """Extract candidate regions from screenshot."""
        try:
            self._init_clip()

            h, w = screenshot.shape[:2]
            crop_size = max(32, min(crop_size, min(h, w)))
            stride = max(16, min(stride, crop_size))

            bboxes = []
            crops = []

            for y in range(0, h - crop_size + 1, stride):
                for x in range(0, w - crop_size + 1, stride):
                    x1, y1 = x, y
                    x2, y2 = x + crop_size, y + crop_size
                    crop = screenshot[y1:y2, x1:x2]
                    bboxes.append((x1, y1, x2, y2))
                    crops.append(crop)
                    if len(crops) >= max_candidates:
                        break
                if len(crops) >= max_candidates:
                    break

            if not crops:
                logger.debug("No candidate crops produced")
                return []

            # Preprocess and batch encode
            pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
            inputs = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)

            batch_size = 64
            embeddings = []
            
            with torch.no_grad():
                for i in range(0, inputs.shape[0], batch_size):
                    batch = inputs[i:i + batch_size]
                    emb = self.model.encode_image(batch)
                    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-10)
                    embeddings.append(emb.cpu())
            
            embeddings = torch.cat(embeddings, dim=0)

            candidates = []
            for bbox, emb, crop in zip(bboxes, embeddings, crops):
                candidates.append({
                    'bbox': bbox,
                    'embedding': emb,
                    'crop': crop
                })

            logger.debug(f"Extracted {len(candidates)} visual candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Extract candidates failed: {e}", exc_info=True)
            return []

    def find_similar_element(self,
                             screenshot: np.ndarray,
                             target_description: str,
                             candidates: Optional[List[Dict]] = None,
                             top_k: int = 1,
                             min_similarity: float = 0.52) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """Find best matching element for target description."""
        try:
            self._init_clip()
            
            # Encode target text
            text_inputs = clip.tokenize([target_description]).to(self.device)
            with torch.no_grad():
                text_emb = self.model.encode_text(text_inputs)
                text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-10)
                text_emb = text_emb.cpu()

            best = None
            best_score = -1.0

            if candidates is None:
                candidates = self.extract_candidates(screenshot)

            if candidates:
                # FIXED: Proper matrix multiplication with correct dimensions
                cand_embs = torch.stack([c['embedding'] for c in candidates], dim=0)  # (N, D)
                text_e = text_emb.squeeze(0)  # (D,) - remove batch dimension
                
                # Compute similarity: (N, D) @ (D,) -> (N,)
                sims = (cand_embs @ text_e).numpy()
                sims = (sims + 1.0) / 2.0
                
                idx = int(np.argmax(sims))
                score = float(sims[idx])
                
                if score > best_score:
                    best_score = score
                    best = (candidates[idx]['bbox'], score)

            # Check stored references
            if self.element_database:
                ref_labels = list(self.element_database.keys())
                ref_embs = torch.stack([self.element_database[l]['embedding'] for l in ref_labels], dim=0)
                
                # FIXED: Proper dimensions for text-to-reference comparison
                text_e = text_emb.squeeze(0)  # (D,)
                ref_sims = (ref_embs.squeeze(1) @ text_e).numpy()  # (N, D) @ (D,) -> (N,)
                ref_sims = (ref_sims + 1.0) / 2.0
                
                ref_idx = int(np.argmax(ref_sims))
                ref_score = float(ref_sims[ref_idx])
                
                if ref_score > best_score:
                    logger.debug(f"Text matches reference '{ref_labels[ref_idx]}' (score={ref_score:.3f})")
                    
                    if candidates:
                        # Match reference to candidates
                        ref_emb = ref_embs[ref_idx].squeeze(0)  # (D,)
                        cand_embs = torch.stack([c['embedding'] for c in candidates], dim=0)
                        
                        sims_ref = (cand_embs @ ref_emb).numpy()
                        sims_ref = (sims_ref + 1.0) / 2.0
                        
                        idx2 = int(np.argmax(sims_ref))
                        score2 = float(sims_ref[idx2])
                        
                        if score2 > best_score:
                            best_score = score2
                            best = (candidates[idx2]['bbox'], score2)
                    else:
                        best = (None, ref_score)

            if not best or best[1] < min_similarity:
                logger.debug(f"No CLIP match above threshold for '{target_description}'")
                return None

            bbox, sim = best
            logger.debug(f"CLIP found match bbox={bbox} sim={sim:.3f}")
            return (bbox, sim)
            
        except Exception as e:
            logger.error(f"Find similar element failed: {e}", exc_info=True)
            return None

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            try:
                del self.model
            except:
                pass
            self.model = None
            self.preprocess = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Visual memory resources cleaned up")