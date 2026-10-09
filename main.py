"""Terminal entry point for the AI attendance management system."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ai_assistant import AIConfigurationError, AIRequestError, AttendanceAssistant
from attendance_manager import AttendanceDataError, AttendanceManager


DATA_FILE = Path(__file__).with_name("attendance.json")


def load_env_file(env_file: Path) -> None:
    """Load simple KEY=VALUE entries without overriding shell environment values."""
    if not env_file.exists():
        return

    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OSError(f"cannot read .env file: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[len("export ") :].strip()
        if "=" not in entry:
            raise ValueError(f"invalid .env entry on line {line_number}: expected KEY=VALUE")

        key, value = (part.strip() for part in entry.split("=", maxsplit=1))
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid variable name in .env on line {line_number}")

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", maxsplit=1)[0].rstrip()
        os.environ.setdefault(key, value)


def main() -> int:
    project_dir = Path(__file__).parent
    env_file = project_dir / ".env"
    if not env_file.exists():
        env_file = project_dir / ".venv" / ".env"

    try:
        load_env_file(env_file)
    except (OSError, ValueError) as error:
        print(f"Could not load .env: {error}", file=sys.stderr)
        return 1

    manager = AttendanceManager(DATA_FILE)
    try:
        manager.load()
    except AttendanceDataError as error:
        print(f"Could not load attendance data: {error}", file=sys.stderr)
        return 1

    api_key = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        print(
            "HF_TOKEN is not set. Set your Hugging Face access token in your terminal "
            "before starting "
            "the assistant (see README.md).",
            file=sys.stderr,
        )
        return 1

    assistant = AttendanceAssistant(
        api_key=api_key,
        model=os.getenv("HF_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")),
        base_url=os.getenv(
            "HF_BASE_URL",
            os.getenv(
                "OPENAI_BASE_URL",
                "https://router.huggingface.co/v1/chat/completions",
            ),
        ),
    )

    print("AI Attendance Management System")
    print("Ask about attendance or request a change. Type 'exit' to quit.")
    print(f"Attendance dates: {manager.date_range_label}")

    while True:
        try:
            request = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if request.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if not request:
            continue

        try:
            result = assistant.handle_request(request, manager)
            print(f"Assistant: {result.message}")
            if result.updated_count:
                print(f"Updated {result.updated_count} attendance record(s).")
        except (AIConfigurationError, AIRequestError, AttendanceDataError) as error:
            print(f"Error: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
