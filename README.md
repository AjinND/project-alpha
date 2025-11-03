# Project Alpha

Project Alpha is a research / engineering codebase for vision-driven UI automation. It combines:

- Visual imitation learning from demonstration videos (cursor tracking + CLIP embeddings)
- Vision-Language Model (VLM) parsing for UI understanding and element localization
- Multiple element-location strategies (CLIP-based memory, GroundingDINO, OCR + fuzzy match)
- A Selenium-based automation engine that executes and verifies actions in a browser

The goal is to convert a demonstration (video + transcribed audio) and a small JSON of input data
into a sequence of automated UI actions (clicks, inputs, navigation) with visual verification and
artifacts (screenshots, summary HTML).

## Key features

- Learn interaction points and visual context from a recorded demonstration video (Visual Imitation).
- Use a VLM (local HTTP endpoint) as a fallback to parse voice commands and to verify UI state changes.
- Multi-strategy location pipeline: imitation memory -> VLM -> CLIP similarity -> GroundingDINO -> OCR.
- Robust action execution with retries, multiple click/input methods, and post-action verification.
- Organized screenshot capture and HTML summary generation for each automation session.

## Repository layout (important files)

- `action_planner.py` — Builds action plans from transcribed audio segments and demonstration video.
- `automation_engine.py` — Executes action plans inside a Selenium browser and verifies results.
- `visual_imitation_system.py` — Learns interaction memories from demonstration videos using CLIP.
- `vlm_interface.py` — Small client to call a local Vision-Language Model HTTP API and parse JSON answers.
- `visual_memory.py` — Stores CLIP embeddings for reference elements and matches live screenshots.
- `screenshot_manager.py` — Saves annotated screenshots and creates an HTML summary of a run.
- `advanced_ocr_matcher.py` — OCR-based extraction + fuzzy matching (PaddleOCR + fuzzywuzzy/RapidFuzz).
- `grounding_dion_locator.py` & `setup_groundingdino.py` — Grounding DINO integration and setup helper.
- `intent_classifier.py` — Classifies/transforms raw transcribed audio segments into actionable intents.
- `input.json` — Example JSON used as input data when filling forms or entering text.
- `requirements.txt` — Full Python dependency list used by the project.

There are a few test/example scripts: `test_app.py`, `test_app_bk.py`, `test_automation_bk.py` — use them as quick references.

## Requirements & recommended environment

- Python 3.10+ (project developed and tested on Python 3.10/3.11)
- Recommended: GPU with CUDA for CLIP / PyTorch speedups. The code falls back to CPU but will be slower.
- A modern Chrome/Chromium browser + corresponding `chromedriver` for Selenium automation.
- A running Vision-Language Model HTTP endpoint (default expected: `http://localhost:11434/api/generate`).

Install dependencies (creates a virtualenv, then install):

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:
- The `requirements.txt` contains large ML packages (torch, clip, paddleocr, groundingdino-py). Installation
	may require CUDA-compatible binaries. If you don't need all features (for instance, GroundingDINO), you can
	install a subset of dependencies.

## Quick setup (Grounding DINO)

Grounding DINO requires a config file and model weights. A helper script is included to download them:

```powershell
python setup_groundingdino.py
```

This will create `GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` and download weights into `weights/`.
If you already have the weights, make sure `weights/groundingdino_swint_ogc.pth` exists.

## Configuration

- The VLM endpoint and model name are configured inside `vlm_interface.py` and used by other modules. By default the code
	expects a local HTTP service at `http://localhost:11434/api/generate` and `llama3.2-vision` / `llama3.2:3b`-style model names.
- Input for automation is provided as a JSON mapping (see `input.json`). The ActionPlanner attempts to map voice commands to
	JSON keys when filling forms.

## Basic usage (conceptual)

1. Produce a transcription of a demonstration video (Whisper/OpenAI Whisper is referenced in requirements). The repo
	 expects `segments` in the form [(timestamp_in_seconds, transcribed_text), ...].
2. Load `input.json` (data to fill into forms) and pass it with segments + the demonstration video to `ActionPlanner`.
3. `ActionPlanner` will try imitation learning (if a demonstration video is provided) and fall back to VLM parsing.
4. The result is a list of action plans. Create a Selenium driver and pass plans to `AutomationEngine` to execute.

Example (high-level Python sketch):

```python
from selenium import webdriver
from action_planner import ActionPlanner
from automation_engine import AutomationEngine
from screenshot_manager import ScreenshotManager
from visual_memory import VisualMemory
from vlm_interface import VLMInterface

# 1) Prepare inputs
json_data = {...}             # load input.json
segments = [(1.2, 'click email field'), ...]  # transcription
video_path = 'demo.mp4'

# 2) Create planner and build action plans
planner = ActionPlanner(json_data)
plans = planner.parse_video_actions(segments, video_path)

# 3) Prepare execution environment
driver = webdriver.Chrome()
screenshot_mgr = ScreenshotManager()
vlm = VLMInterface('http://localhost:11434/api/generate', 'llama3.2-vision')
visual_mem = VisualMemory()
engine = AutomationEngine(driver, vlm, visual_mem, json_data, screenshot_mgr)

# 4) Execute plans
for i, plan in enumerate(plans):
		result = engine.execute_action(plan, i)
		print(result)
```

The repository includes more advanced logic (imitation learner, multi-strategy locating, verification). The sketch
above shows the wiring you can replicate in your scripts.

## Running the included examples / tests

- `test_app.py` and the other `test_*` files show example usage patterns used when developing. They are not full unit
	tests with a test runner, but a quick way to try pieces of the system.

To run one interactively (example):

```powershell
python test_app.py
```

Note: many examples assume you have a running VLM service and optionally model weights present.

## Development notes & tips

- CLIP and GroundingDINO are loaded lazily in code to reduce startup time. If you intend to use those features, run on a
	machine with a compatible GPU + CUDA for reasonable performance.
- The multi-strategy locator prefers imitation memory (if available), then VLM, then CLIP similarity, then GroundingDINO, then OCR.
- Tuning thresholds (confidence levels, OCR fuzzy match thresholds, DINO box thresholds) is often necessary for different UIs.
- The `ScreenshotManager` saves organized artifacts under `automation_screenshots/session_*` and produces `summary.html`.

## Troubleshooting

- If CLIP/torch fails to import, ensure you installed `torch` compatible with your CUDA version, or install the CPU-only wheel.
- If Grounding DINO inference raises model or CUDA errors, check `weights/groundingdino_swint_ogc.pth` exists and the `config` path in
	`grounding_dion_locator.py` (note: filename includes 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py').
- If the VLM endpoint is unreachable, either run your local VLM server (Ollama/llama.cpp-based service or other) or change the
	endpoint URL used by `VLMInterface`.

## Security & privacy

This repository interacts with local browsers and may process demonstration videos that contain private data. Keep
input data and model endpoints secure. Avoid sending secrets to third-party endpoints unless explicitly intended.

## Next steps / suggestions

- Add small runnable examples that wire end-to-end flows with a headless browser and a mock VLM for onboarding users.
- Add unit tests for pure-Python components (OCR fuzzy matcher, VLM client parsing, screenshot manager HTML generation).
- Provide a lightweight Docker Compose that runs a small VLM stub for local development (faster onboarding).

## License

MIT-style (no license file included) — add a LICENSE if you want to make licensing explicit.

---

If you'd like, I can also:

- add a small example script that runs an end-to-end demo using a mocked VLM response so newcomers can run quickly;
- create a smaller `requirements-dev.txt` with only the packages needed for CPU-only testing.
