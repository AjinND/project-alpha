import os
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class ScreenshotManager:
    """Manages screenshot capture and organization."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or "automation_screenshots"
        self.session_dir = None
        self.action_counter = 0
        self.initialize_session()
    
    def initialize_session(self):
        """Create new session directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        logger.info(f"Screenshot session initialized: {self.session_dir}")
    
    def save_screenshot(self, image: np.ndarray, action_num: int, 
                        stage: str, description: str = "") -> str:
        """Save screenshot with organized naming."""
        action_dir = os.path.join(self.session_dir, f"action_{action_num:02d}")
        os.makedirs(action_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%H%M%S")
        desc_part = f"_{description.replace(' ', '_')}" if description else ""
        filename = f"{stage}_{timestamp}{desc_part}.png"
        filepath = os.path.join(action_dir, filename)
        
        annotated = image.copy()
        cv2.putText(annotated, f"Action {action_num} - {stage.upper()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if description:
            cv2.putText(annotated, description[:50], 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(filepath, annotated)
        logger.debug(f"Screenshot saved: {filepath}")
        return filepath
    
    def save_element_highlight(self, image: np.ndarray, position: Tuple[int, int], 
                               action_num: int, description: str) -> str:
        """Save screenshot with element highlighted."""
        highlighted = image.copy()
        x, y = position
        cv2.drawMarker(highlighted, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(highlighted, (x, y), 50, (0, 255, 0), 3)
        cv2.putText(highlighted, description, (x + 60, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return self.save_screenshot(highlighted, action_num, "locate", description)
    
    def create_summary_html(self, execution_log: List[Dict]) -> str:
        """Generate HTML summary with screenshots."""
        html_path = os.path.join(self.session_dir, "summary.html")
        html_content = self._build_html_content(execution_log)
        with open(html_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Summary HTML created: {html_path}")
        return html_path
    
    def _build_html_content(self, execution_log: List[Dict]) -> str:
        """Build HTML content with screenshots."""
        html = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>Automation Summary</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background: #f5f5f5;
                    }}
                    .header {{
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .action {{
                        background: white;
                        margin: 20px 0;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .action.success {{
                        border-left: 5px solid #4CAF50;
                    }}
                    .action.failed {{
                        border-left: 5px solid #f44336;
                    }}
                    .screenshots {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                        margin-top: 10px;
                    }}
                    .screenshots img {{
                        max-width: 300px;
                        border: 2px solid #ddd;
                        border-radius: 4px;
                        cursor: pointer;
                        transition: border-color 0.3s;
                    }}
                    .screenshots img:hover {{
                        border-color: #2196F3;
                    }}
                    h1 {{
                        color: #333;
                        margin: 0;
                    }}
                    .meta {{
                        color: #666;
                        font-size: 0.9em;
                        line-height: 1.6;
                    }}
                    .status {{
                        display: inline-block;
                        padding: 4px 12px;
                        border-radius: 12px;
                        font-weight: bold;
                        font-size: 0.85em;
                    }}
                    .status.success {{
                        background: #4CAF50;
                        color: white;
                    }}
                    .status.failed {{
                        background: #f44336;
                        color: white;
                    }}
                    .summary-stats {{
                        display: flex;
                        gap: 20px;
                        margin-top: 10px;
                    }}
                    .stat {{
                        padding: 10px 20px;
                        background: #f0f0f0;
                        border-radius: 4px;
                    }}
                    .stat-value {{
                        font-size: 1.5em;
                        font-weight: bold;
                        color: #333;
                    }}
                    .stat-label {{
                        font-size: 0.85em;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 Automation Execution Summary</h1>
                    <p>Session: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <div class="summary-stats">
                        <div class="stat">
                            <div class="stat-value">{len(execution_log)}</div>
                            <div class="stat-label">Total Actions</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{sum(1 for log in execution_log if log.get('status') == 'success')}</div>
                            <div class="stat-label">Successful</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{sum(1 for log in execution_log if log.get('status') == 'failed')}</div>
                            <div class="stat-label">Failed</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{(sum(1 for log in execution_log if log.get('status') == 'success') / len(execution_log) * 100) if execution_log else 0:.1f}%</div>
                            <div class="stat-label">Success Rate</div>
                        </div>
                    </div>
                </div>
            """
        
        for log in execution_log:
            status_class = "success" if log.get('status') == 'success' else "failed"
            action_num = log.get('action_number', 0)
            action_dir = f"action_{action_num:02d}"
            
            # Find all screenshots for this action
            action_path = os.path.join(self.session_dir, action_dir)
            screenshots = []
            if os.path.exists(action_path):
                screenshots = sorted([
                    f for f in os.listdir(action_path)
                    if f.endswith('.png')
                ])
            
            intent = log.get('intent', 'unknown').upper()
            audio = log.get('audio', 'N/A')
            target = log.get('target', 'N/A')
            data_key = log.get('data_key', 'N/A')
            status = log.get('status', 'unknown').upper()
            attempt = log.get('attempt', 0)
            has_memory = log.get('has_visual_memory', False)
            
            html += f"""
                <div class="action {status_class}">
                    <h2>Action {action_num}: {intent}</h2>
                    <p class="meta">
                        <strong>Audio Command:</strong> {audio}<br>
                        <strong>Target:</strong> {target}<br>
                        <strong>Data Key:</strong> {data_key}<br>
                        <strong>Attempts:</strong> {attempt}<br>
                        <strong>Visual Memory:</strong> {'✓ Yes' if has_memory else '✗ No'}<br>
                        <span class="status {status_class}">{status}</span>
                    </p>
            """
            
            if screenshots:
                html += '        <div class="screenshots">\n'
                for screenshot in screenshots:
                    rel_path = os.path.join(action_dir, screenshot)
                    # Create a cleaner display name
                    display_name = screenshot.replace('_', ' ').replace('.png', '')
                    html += f'            <img src="{rel_path}" alt="{display_name}" title="{display_name}" onclick="window.open(this.src)">\n'
                html += '        </div>\n'
            
            html += '    </div>\n'
        
        html += """
            </body>
            </html>
        """
        return html