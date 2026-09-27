"""SigForge entry point."""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    """Launch the SigForge GUI application."""
    from sigforge.gui.app import run_app
    sys.exit(run_app(sys.argv))


if __name__ == "__main__":
    main()

