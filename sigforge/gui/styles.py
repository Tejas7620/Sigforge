"""
SigForge dark theme — professional QSS stylesheet.

Inspired by modern dark UI conventions: subtle borders, high contrast text,
accent colors for interactive elements, color-coded confidence badges.
"""

# Color palette
COLORS = {
    "bg_primary": "#0f1117",        # main background (near-black)
    "bg_secondary": "#1a1d27",      # panel/card background
    "bg_tertiary": "#252833",       # input fields, elevated surfaces
    "bg_hover": "#2d3142",          # hover state
    "bg_active": "#363b4f",         # active/pressed state

    "text_primary": "#e8eaf0",      # main text
    "text_secondary": "#9ba1b0",    # secondary text
    "text_muted": "#6b7280",        # muted text / labels

    "accent": "#6366f1",            # primary accent (indigo)
    "accent_hover": "#818cf8",      # accent hover
    "accent_active": "#4f46e5",     # accent pressed
    "accent_glow": "rgba(99, 102, 241, 0.15)",  # subtle glow

    "border": "#2d3142",            # subtle borders
    "border_focus": "#6366f1",      # focused element border

    "success": "#22c55e",           # green — validated
    "warning": "#eab308",           # yellow — strong inference
    "caution": "#f97316",           # orange — weak hypothesis
    "neutral": "#9ca3af",           # grey — unknown
    "error": "#ef4444",             # red — errors

    "scrollbar_bg": "#1a1d27",
    "scrollbar_handle": "#363b4f",
    "scrollbar_hover": "#4b5066",
}


def get_stylesheet() -> str:
    """Return the complete QSS dark theme stylesheet."""
    c = COLORS
    return f"""
    /* ===== Global ===== */
    QMainWindow {{
        background-color: {c['bg_primary']};
        color: {c['text_primary']};
    }}

    QWidget {{
        background-color: {c['bg_primary']};
        color: {c['text_primary']};
        font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
        font-size: 13px;
    }}

    /* ===== Menu Bar ===== */
    QMenuBar {{
        background-color: {c['bg_secondary']};
        color: {c['text_primary']};
        border-bottom: 1px solid {c['border']};
        padding: 2px;
    }}

    QMenuBar::item {{
        padding: 6px 12px;
        border-radius: 4px;
    }}

    QMenuBar::item:selected {{
        background-color: {c['bg_hover']};
    }}

    QMenu {{
        background-color: {c['bg_secondary']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 4px;
    }}

    QMenu::item {{
        padding: 6px 24px;
        border-radius: 4px;
    }}

    QMenu::item:selected {{
        background-color: {c['accent']};
        color: white;
    }}

    /* ===== Tool Bar ===== */
    QToolBar {{
        background-color: {c['bg_secondary']};
        border-bottom: 1px solid {c['border']};
        padding: 4px 8px;
        spacing: 6px;
    }}

    QToolBar::separator {{
        width: 1px;
        background-color: {c['border']};
        margin: 4px 8px;
    }}

    /* ===== Dock Widgets ===== */
    QDockWidget {{
        color: {c['text_primary']};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}

    QDockWidget::title {{
        background-color: {c['bg_secondary']};
        color: {c['text_secondary']};
        padding: 8px 12px;
        border-bottom: 1px solid {c['border']};
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    QDockWidget::close-button, QDockWidget::float-button {{
        background: transparent;
        border: none;
        padding: 2px;
    }}

    /* ===== Tab Widget ===== */
    QTabWidget::pane {{
        background-color: {c['bg_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: {c['bg_secondary']};
        color: {c['text_secondary']};
        padding: 8px 16px;
        border: 1px solid {c['border']};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 2px;
        font-weight: 500;
    }}

    QTabBar::tab:selected {{
        background-color: {c['bg_primary']};
        color: {c['accent']};
        border-bottom: 2px solid {c['accent']};
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {c['bg_hover']};
        color: {c['text_primary']};
    }}

    /* ===== Buttons ===== */
    QPushButton {{
        background-color: {c['bg_tertiary']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {c['bg_hover']};
        border-color: {c['accent']};
    }}

    QPushButton:pressed {{
        background-color: {c['bg_active']};
    }}

    QPushButton:disabled {{
        color: {c['text_muted']};
        background-color: {c['bg_secondary']};
    }}

    QPushButton#btn_primary {{
        background-color: {c['accent']};
        color: white;
        border: none;
        font-weight: 600;
    }}

    QPushButton#btn_primary:hover {{
        background-color: {c['accent_hover']};
    }}

    QPushButton#btn_primary:pressed {{
        background-color: {c['accent_active']};
    }}

    QPushButton#btn_danger {{
        background-color: #991b1b;
        color: white;
        border: none;
    }}

    QPushButton#btn_danger:hover {{
        background-color: {c['error']};
    }}

    /* ===== Input Fields ===== */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c['bg_tertiary']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 10px;
        selection-background-color: {c['accent']};
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {c['border_focus']};
    }}

    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {c['bg_secondary']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        selection-background-color: {c['accent']};
    }}

    /* ===== Labels ===== */
    QLabel {{
        color: {c['text_primary']};
        background: transparent;
    }}

    QLabel#label_section {{
        color: {c['text_secondary']};
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 4px 0px;
    }}

    QLabel#label_muted {{
        color: {c['text_muted']};
    }}

    /* ===== Progress Bar ===== */
    QProgressBar {{
        background-color: {c['bg_tertiary']};
        border: none;
        border-radius: 4px;
        text-align: center;
        color: {c['text_primary']};
        font-size: 11px;
        min-height: 16px;
    }}

    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['accent']}, stop:1 {c['accent_hover']});
        border-radius: 4px;
    }}

    /* ===== Scroll Bars ===== */
    QScrollBar:vertical {{
        background-color: {c['scrollbar_bg']};
        width: 10px;
        border-radius: 5px;
        margin: 0px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {c['scrollbar_handle']};
        border-radius: 5px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {c['scrollbar_hover']};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        background-color: {c['scrollbar_bg']};
        height: 10px;
        border-radius: 5px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {c['scrollbar_handle']};
        border-radius: 5px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {c['scrollbar_hover']};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* ===== Text Browser / Plain Text ===== */
    QTextEdit, QPlainTextEdit, QTextBrowser {{
        background-color: {c['bg_tertiary']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px;
        font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
        font-size: 12px;
    }}

    /* ===== Group Box ===== */
    QGroupBox {{
        background-color: {c['bg_secondary']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        margin-top: 16px;
        padding-top: 20px;
        font-weight: 600;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 4px 12px;
        color: {c['text_secondary']};
    }}

    /* ===== Splitter ===== */
    QSplitter::handle {{
        background-color: {c['border']};
    }}

    QSplitter::handle:horizontal {{
        width: 2px;
    }}

    QSplitter::handle:vertical {{
        height: 2px;
    }}

    /* ===== Tree / List / Table ===== */
    QTreeView, QListView, QTableView {{
        background-color: {c['bg_tertiary']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        alternate-background-color: {c['bg_secondary']};
    }}

    QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{
        background-color: {c['accent']};
        color: white;
    }}

    QHeaderView::section {{
        background-color: {c['bg_secondary']};
        color: {c['text_secondary']};
        border: 1px solid {c['border']};
        padding: 6px 10px;
        font-weight: 600;
        font-size: 11px;
    }}

    /* ===== Status Bar ===== */
    QStatusBar {{
        background-color: {c['bg_secondary']};
        color: {c['text_secondary']};
        border-top: 1px solid {c['border']};
        font-size: 12px;
    }}

    /* ===== Tooltips ===== */
    QToolTip {{
        background-color: {c['bg_secondary']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    """


def get_badge_style(badge_color: str) -> str:
    """Return inline QSS for a confidence badge label."""
    return f"""
        background-color: {badge_color};
        color: white;
        border-radius: 10px;
        padding: 2px 10px;
        font-weight: 700;
        font-size: 11px;
    """
