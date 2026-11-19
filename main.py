"""
Main entry point for Indian Traffic Sign Recognition and Road Safety Decision-Support Monitor.
Usage:
  python main.py             # Runs Road Safety Monitor Demonstrator CLI
  python main.py --gui       # Launches Desktop GUI Decision-Support Application
  python main.py --audit     # Runs Task 17 False-Positive & State Integrity Auditor
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Indian Traffic Sign Recognition & Safety Monitor")
    parser.add_argument("--gui", action="store_true", help="Launch the desktop GUI decision-support monitor")
    parser.add_argument("--audit", action="store_true", help="Run Task 17 False-Positive & State Integrity Auditor")
    parser.add_argument("--demo", action="store_true", help="Run scenario simulation demonstrator CLI (default)")

    args = parser.parse_args()

    if args.gui:
        from trafficsign.ui.app import RoadSafetyMonitorApp
        print("Launching Desktop Road Safety Decision-Support Monitor GUI...")
        app = RoadSafetyMonitorApp()
        app.run()
    elif args.audit:
        from trafficsign.app.audit_runner import Task17AuditRunner
        runner = Task17AuditRunner()
        runner.execute()
    else:
        from trafficsign.app.demonstrator import RoadSafetyMonitorDemonstrator
        demo = RoadSafetyMonitorDemonstrator()
        demo.execute()


if __name__ == "__main__":
    main()
