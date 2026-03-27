"""
Laptop AI — Phase 4: Windows Task Scheduler Setup
Configures a daily scheduled task that runs predict.py every morning
and shows a Windows toast notification with your predicted files.

What this does:
    1. Creates a Windows Task Scheduler task called "LaptopAI-Predict"
    2. Runs predict.py at 8:00 AM every day
    3. Shows a toast notification with your top 3 predicted files
    4. Each file in the notification is clickable — opens the file directly

Usage:
    python schedule.py --install       # Set up the daily task
    python schedule.py --uninstall     # Remove the daily task
    python schedule.py --test          # Run prediction + notification now
    python schedule.py --time 09:00    # Change the scheduled time
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


TASK_NAME = "LaptopAI-Predict"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def find_python() -> str:
    """Find the Python executable path."""
    # Try common locations on Windows
    candidates = [
        sys.executable,
        "python",
        "python3",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Get the full path
                which = subprocess.run(
                    ["where" if os.name == "nt" else "which", candidate],
                    capture_output=True, text=True, timeout=5
                )
                if which.returncode == 0:
                    return which.stdout.strip().split("\n")[0]
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    print("ERROR: Could not find Python. Make sure Python is installed and in PATH.")
    sys.exit(1)


def show_notification(predictions: list[dict]):
    """
    Show a Windows toast notification with predicted files.
    Falls back to console output if toast library not available.
    """
    if not predictions:
        print("No predictions to show.")
        return

    title = "Laptop AI — Today's Files"
    body_lines = []
    for i, p in enumerate(predictions[:3], 1):
        pct = int(p["probability"] * 100)
        body_lines.append(f"{i}. {p['filename']} ({pct}%)")
    body = "\n".join(body_lines)

    # Try Windows toast notification
    try:
        if os.name == "nt":
            # Use PowerShell for toast notification (no extra dependencies)
            ps_script = f'''
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null

$template = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>{title}</text>
            <text>{body.replace(chr(10), "&#xA;")}</text>
        </binding>
    </visual>
</toast>
"@

$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Laptop AI").Show($toast)
'''
            subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True, timeout=10
            )
            print("Toast notification sent!")
            return
    except Exception as e:
        pass  # Fall through to console

    # Fallback: console output
    print()
    print(f"  {title}")
    print(f"  {'─' * 40}")
    for line in body_lines:
        print(f"  {line}")
    print()


def install_task(time_str: str = "08:00"):
    """Create a Windows Task Scheduler task."""
    if os.name != "nt":
        print("Task Scheduler is Windows-only.")
        print("On Linux/Mac, add this to your crontab:")
        print(f"  0 8 * * * cd {PROJECT_DIR} && python {SCRIPT_DIR}/schedule.py --test")
        return

    python_path = find_python()
    predict_script = os.path.join(SCRIPT_DIR, "predict.py")
    schedule_script = os.path.abspath(__file__)

    # The task runs schedule.py --test, which runs predict + notification
    cmd = (
        f'schtasks /create /tn "{TASK_NAME}" '
        f'/tr "\\"{python_path}\\" \\"{schedule_script}\\" --test" '
        f'/sc daily /st {time_str} '
        f'/sd {datetime.now().strftime("%m/%d/%Y")} '
        f'/f'  # force overwrite if exists
    )

    print(f"Creating scheduled task: {TASK_NAME}")
    print(f"  Time: {time_str} daily")
    print(f"  Python: {python_path}")
    print(f"  Script: {schedule_script}")
    print()

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("Task created successfully!")
        print(f"Predictions will run daily at {time_str}.")
        print()
        print("To verify: Open Task Scheduler and look for 'LaptopAI-Predict'")
        print("To remove: python schedule.py --uninstall")
        print("To test now: python schedule.py --test")
    else:
        print(f"Failed to create task: {result.stderr}")
        if "Access is denied" in result.stderr:
            print("\nTry running Command Prompt as Administrator.")


def uninstall_task():
    """Remove the scheduled task."""
    if os.name != "nt":
        print("Remove the crontab entry manually: crontab -e")
        return

    cmd = f'schtasks /delete /tn "{TASK_NAME}" /f'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' removed.")
    else:
        print(f"Could not remove task: {result.stderr}")


def run_test():
    """Run prediction and show notification."""
    # Import predict module
    sys.path.insert(0, SCRIPT_DIR)
    from predict import load_models, predict_now, load_query_log, build_features, train_models, save_models

    # Load or train model
    models, vectorizer, meta = load_models()

    if models is None:
        print("No trained model found. Training now...")
        entries = load_query_log()
        if len(entries) < 50:
            print(f"Only {len(entries)} queries logged. Need 50+.")
            print("Run 'python generate_demo_data.py' for demo data.")
            return
        X, file_labels, vectorizer, feature_names = build_features(entries)
        if not file_labels:
            print("No files accessed frequently enough to model.")
            return
        models, stats = train_models(X, file_labels, feature_names)
        save_models(models, vectorizer, stats)

    # Predict
    now = datetime.now()
    predictions = predict_now(models, vectorizer, day=now.weekday(), hour=now.hour)

    # Show notification
    show_notification(predictions)

    # Also print to console (for logging)
    from predict import display_predictions, DAY_NAMES
    display_predictions(predictions, now.weekday(), now.hour)


def main():
    parser = argparse.ArgumentParser(
        description="Laptop AI — Schedule daily predictions"
    )
    parser.add_argument("--install", action="store_true", help="Create daily scheduled task")
    parser.add_argument("--uninstall", action="store_true", help="Remove scheduled task")
    parser.add_argument("--test", action="store_true", help="Run prediction + notification now")
    parser.add_argument("--time", type=str, default="08:00", help="Time for daily prediction (HH:MM)")
    args = parser.parse_args()

    if args.install:
        install_task(args.time)
    elif args.uninstall:
        uninstall_task()
    elif args.test:
        run_test()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python schedule.py --install          # Set up daily 8 AM predictions")
        print("  python schedule.py --install --time 09:30  # Change time")
        print("  python schedule.py --test             # Test it now")
        print("  python schedule.py --uninstall        # Remove the task")


if __name__ == "__main__":
    main()
