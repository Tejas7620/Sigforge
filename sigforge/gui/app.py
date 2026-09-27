"""
SigForge GUI Application launcher.

Creates the QApplication, applies the dark theme, and shows the main window.
"""
from __future__ import annotations

import sys
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def run_app(argv: Optional[List[str]] = None) -> int:
    """Create and run the SigForge GUI application.

    Args:
        argv: Command-line arguments (typically sys.argv).

    Returns:
        Application exit code.
    """
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont

    from sigforge.gui.main_window import MainWindow
    from sigforge.gui.styles import get_stylesheet

    if argv is None:
        argv = sys.argv

    # Enable High DPI scaling
    app = QApplication(argv)
    app.setApplicationName("SigForge")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("SigForge Team")

    # Set default font
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    # Apply global stylesheet
    app.setStyleSheet(get_stylesheet())

    # Create and show main window
    window = MainWindow()
    window.show()

    return app.exec()
