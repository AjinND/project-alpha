from dataclasses import dataclass

@dataclass
class Config:
    vlm_endpoint: str = "http://localhost:11434/api/generate"
    vlm_model: str = "llama3.2-vision"
    whisper_model: str = "large-v3"
    clip_model: str = "ViT-B/32"
    confidence_threshold: float = 0.7
    max_retries: int = 3
    request_timeout: int = 300
    screenshot_dir: str = "automation_screenshots"