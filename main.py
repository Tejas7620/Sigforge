import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sigforge.gui.app import run_app


def main():
    sys.exit(run_app(sys.argv))


if __name__ == "__main__":
    main()
