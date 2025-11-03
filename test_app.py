import argparse
import json
import logging
import os
import time
import cv2
import numpy as np
import whisper
from moviepy import VideoFileClip
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from io import StringIO
import sys

from config import Config
from screenshot_manager import ScreenshotManager
from action_planner import ActionPlanner
from automation_engine import AutomationEngine
from intent_classifier import integrate_intent_classifier

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("automation_enhanced.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="AI Vision-Based Automation")
    parser.add_argument("--video", required=True, help="Path to demonstration video")
    parser.add_argument("--json", required=True, help="Path to JSON data file")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--skip-filter", action="store_true", help="Skip intent filtering")
    parser.add_argument("--no-imitation", action="store_true", help="Disable imitation learning")
    parser.add_argument("--screenshot-dir", default="automation_screenshots", help="Screenshot directory")
    return parser.parse_args()

def transcribe_audio(video_path: str, config: Config) -> list:
    """Extract and transcribe audio from video."""
    logger.info("Extracting audio from video...")
    audio_path = "temp_audio.mp3"
    
    # Suppress moviepy output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    logger.info(f"Transcribing with Whisper {config.whisper_model}...")
    model = whisper.load_model(config.whisper_model)
    result = model.transcribe(audio_path, verbose=False)
    
    segments = [(seg['start'], seg['text']) for seg in result['segments']]
    
    try:
        os.remove(audio_path)
    except:
        pass
    
    return segments

def main():
    args = parse_args()
    config = Config()
    
    # Update config from args
    if args.screenshot_dir:
        config.screenshot_dir = args.screenshot_dir
    
    # Validate inputs
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON not found: {args.json}")
    
    with open(args.json, 'r') as f:
        json_data = json.load(f)
    
    logger.info("=" * 70)
    logger.info("STARTING AI VISION-BASED AUTOMATION WITH IMITATION LEARNING")
    logger.info("=" * 70)
    
    # Initialize screenshot manager
    screenshot_mgr = ScreenshotManager(config.screenshot_dir)
    
    # Step 1: Extract and transcribe audio
    logger.info("\n" + "="*60)
    logger.info("STEP 1: AUDIO EXTRACTION & TRANSCRIPTION")
    logger.info("="*60)
    
    raw_segments = transcribe_audio(args.video, config)
    logger.info(f"✓ Transcribed {len(raw_segments)} audio segments")
    
    # Step 1.5: Filter actionable commands
    intent_classifications = None
    if args.skip_filter:
        logger.warning("⚠ Skipping intent filtering (--skip-filter)")
        segments = raw_segments
    else:
        logger.info("\n" + "="*60)
        logger.info("STEP 1.5: INTENT CLASSIFICATION & FILTERING")
        logger.info("="*60)
        
        filtered_segments = integrate_intent_classifier(raw_segments, json_data)
        segments = [(seg['timestamp'], seg['cleaned_text']) for seg in filtered_segments]
        intent_classifications = filtered_segments
        
        logger.info(f"✓ Filtered: {len(raw_segments)} → {len(segments)} actionable commands")
        logger.info(f"✓ Removed {len(raw_segments) - len(segments)} non-actionable segments")
    
    # Step 2: Parse video into action plan
    logger.info("\n" + "="*60)
    logger.info("STEP 2: VIDEO PARSING & VISUAL IMITATION LEARNING")
    logger.info("="*60)
    
    planner = ActionPlanner(json_data, no_imitation=args.no_imitation)
    action_plans = planner.parse_video_actions(
        segments,
        args.video,
        intent_classifications=intent_classifications
    )
    
    logger.info(f"✓ Generated {len(action_plans)} detailed actions")
    
    if not action_plans:
        logger.error("✗ No action plans generated. Check video and audio quality.")
        return
    
    # Log visual memory statistics
    actions_with_memory = sum(1 for plan in action_plans if 'visual_memory' in plan)
    logger.info(f"📸 {actions_with_memory}/{len(action_plans)} actions have visual memories")
    
    # Step 3: Initialize browser
    logger.info("\n" + "="*60)
    logger.info("STEP 3: BROWSER INITIALIZATION")
    logger.info("="*60)
    
    opts = Options()
    if args.headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)
    
    initial_url = 'http://localhost:3000'
    logger.info(f"Initial navigation to: {initial_url}")
    driver.get(initial_url)
    time.sleep(2)
    
    # Capture initial state
    initial_screenshot = driver.get_screenshot_as_png()
    initial_img = cv2.imdecode(np.frombuffer(initial_screenshot, np.uint8), cv2.IMREAD_COLOR)
    screenshot_mgr.save_screenshot(initial_img, 0, "initial", "Browser_started")
    
    logger.info("✓ Browser ready")
    
    # Step 4: Execute automation
    logger.info("\n" + "="*60)
    logger.info("STEP 4: AUTOMATION EXECUTION WITH VISUAL IMITATION")
    logger.info("="*60)
    
    engine = AutomationEngine(
        driver,
        planner.vlm,
        planner.visual_memory,
        json_data,
        screenshot_mgr,
        imitation_learner=planner.imitation_learner
    )
    
    execution_log = []
    
    try:
        for i, action_plan in enumerate(action_plans):
            action_num = i + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ACTION {action_num}/{len(action_plans)}: {action_plan['intent'].upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"Audio: {action_plan.get('audio', 'N/A')}")
            logger.info(f"Target: {action_plan.get('target_description', 'N/A')}")
            logger.info(f"Data: {action_plan.get('data_key', 'N/A')}")
            
            if 'visual_memory' in action_plan:
                logger.info(f"📸 Visual memory available from demo")
                logger.info(f"   Demo interaction point: {action_plan.get('interaction_point_demo')}")
            else:
                logger.info(f"⚠️ No visual memory - will use fallback strategies")
            
            logger.info(f"{'='*60}")
            
            result = engine.execute_action(action_plan, action_num)
            
            execution_log.append({
                'action_number': action_num,
                'timestamp': action_plan.get('timestamp', 0),
                'audio': action_plan.get('audio', ''),
                'intent': action_plan['intent'],
                'target': action_plan.get('target_description', ''),
                'data_key': action_plan.get('data_key'),
                'status': result['status'],
                'attempt': result.get('attempt', 0),
                'has_visual_memory': 'visual_memory' in action_plan,
                'details': result
            })
            
            if result['status'] == 'failed':
                logger.error(f"✗ Action {action_num} FAILED: {result.get('reason')}")
                if action_plan.get('critical', False):
                    logger.error("⚠ Critical action failed. Aborting automation.")
                    break
            else:
                logger.info(f"✓ Action {action_num} SUCCEEDED (attempt {result.get('attempt', 1)})")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Automation interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Fatal error during execution: {e}", exc_info=True)
    finally:
        # Step 5: Cleanup
        logger.info("\n" + "="*60)
        logger.info("STEP 5: CLEANUP & REPORT GENERATION")
        logger.info("="*60)
        
        try:
            final_screenshot = driver.get_screenshot_as_png()
            final_img = cv2.imdecode(np.frombuffer(final_screenshot, np.uint8), cv2.IMREAD_COLOR)
            screenshot_mgr.save_screenshot(final_img, 999, "final", "Automation_completed")
        except:
            pass
        
        # Cleanup resources
        engine.cleanup()
        planner.visual_memory.cleanup()
        
        driver.quit()
        logger.info("✓ Browser closed")
        logger.info("✓ Resources cleaned up")
    
    # Step 6: Generate reports
    logger.info("\n" + "="*60)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*60)
    
    if not execution_log:
        logger.warning("⚠ No actions were executed")
        return
    
    successful = sum(1 for log in execution_log if log['status'] == 'success')
    failed = len(execution_log) - successful
    success_rate = (successful/len(execution_log)*100) if execution_log else 0
    with_memory = sum(1 for log in execution_log if log.get('has_visual_memory'))
    
    logger.info(f"📊 Total actions: {len(execution_log)}")
    logger.info(f"✓ Successful: {successful}")
    logger.info(f"✗ Failed: {failed}")
    logger.info(f"📈 Success rate: {success_rate:.1f}%")
    logger.info(f"📸 Actions with visual memory: {with_memory}/{len(execution_log)}")
    
    # Save JSON report
    report = {
        'summary': {
            'total_raw_segments': len(raw_segments) if not args.skip_filter else len(segments),
            'filtered_segments': len(segments),
            'total_actions': len(execution_log),
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'actions_with_visual_memory': with_memory
        },
        'actions': execution_log,
        'configuration': {
            'vlm_model': config.vlm_model,
            'whisper_model': config.whisper_model,
            'clip_model': config.clip_model,
            'intent_filtering_enabled': not args.skip_filter,
            'visual_imitation_enabled': not args.no_imitation,
            'screenshot_directory': screenshot_mgr.session_dir
        }
    }
    
    report_path = 'automation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ JSON report saved: {report_path}")
    
    # Generate HTML summary
    html_path = screenshot_mgr.create_summary_html(execution_log)
    logger.info(f"✓ HTML summary created: {html_path}")
    logger.info(f"✓ Screenshots saved in: {screenshot_mgr.session_dir}")
    
    logger.info("\n" + "="*70)
    logger.info("AUTOMATION COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    main()