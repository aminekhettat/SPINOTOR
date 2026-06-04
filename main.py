"""
    SPINOTOR Application - Main Entry Point
==================================================

Launches the GUI application for BLDC motor simulation with V/f control.

Usage:
    python main.py

Requirements:
    See requirements.txt

:author: Amine Khettat — Blind Systems
:version: 0.12.0

.. versionadded:: 0.12.0
    Initial application release
"""

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# Add project root to path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# application-specific modules will be imported after path insertion inside main()

logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the BLDC motor control application.

    Initializes PySide6 application and launches the main GUI window.
    """
    logger.info("=" * 60)
    logger.info("BLIND SYSTEMS SPINOTOR v0.12.0")
    logger.info("=" * 60)

    try:
        # After path adjustment, import project modules
        from src.ui.main_window import BLDCMotorControlGUI
        from src.utils.config import LOG_FORMAT, LOG_LEVEL

        # Configure logging now that format and level are available
        logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)

        # Create Qt application
        app = QApplication(sys.argv)

        # Set application info
        app.setApplicationName("BLIND SYSTEMS SPINOTOR")
        app.setApplicationVersion("0.12.0")

        # Create and show main window
        window = BLDCMotorControlGUI()
        window.show()

        logger.info("Application window opened successfully")
        logger.info("Ready for simulation - Use F5 to start, F6 to stop, F7 to reset")
        logger.info("Press Ctrl+Q to quit or use Quit button")

        # Run application event loop
        sys.exit(app.exec())

    except Exception as e:
        logger.exception(f"Fatal error in application startup: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
