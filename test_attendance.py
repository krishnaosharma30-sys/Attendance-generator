"""Unit tests for attendance calculations, persistence, and AI validation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_assistant import AIRequestError, AttendanceAssistant
from attendance_manager import (
    ATTENDANCE_DATES,
    AttendanceDataError,
    AttendanceManager,
)
from main import load_env_file


class EnvFileTests(unittest.TestCase):
    def test_loads_values_and_preserves_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "HF_TOKEN='from-file'\nHF_MODEL=test-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HF_TOKEN": "from-shell"}, clear=True):
                load_env_file(env_file)
                self.assertEqual(os.environ["HF_TOKEN"], "from-shell")
                self.assertEqual(os.environ["HF_MODEL"], "test-model")


class AttendanceManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_file = Path(self.temp_dir.name) / "attendance.json"
        self.data_file.write_text(
            json.dumps(
                {
                    "students": [
                        {
                            "name": "Test Student",
                            "roll_no": "1",
                            "total_days": 0,
                            "present_days": 0,
                            "attendance_percentage": 0,
                            "attendance": {
                                day: "Absent" for day in ATTENDANCE_DATES
                            },
                        },
                        {
                            "name": "Second Student",
                            "roll_no": "2",
                            "total_days": 0,
                            "present_days": 0,
                            "attendance_percentage": 0,
                            "attendance": {
                                day: "Present" for day in ATTENDANCE_DATES
                            },
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.manager = AttendanceManager(self.data_file)
        self.manager.load()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_recalculates_student_and_class_averages(self) -> None:
        summary = self.manager.summary()
        self.assertEqual(summary["students"][0]["present_days"], 0)
        self.assertEqual(summary["students"][0]["attendance_percentage"], 0.0)
        self.assertEqual(summary["students"][1]["attendance_percentage"], 100.0)
        self.assertEqual(summary["class_average_percentage"], 50.0)

    def test_update_persists_attendance_and_metrics(self) -> None:
        count = self.manager.apply_updates(
            [{"roll_no": "1", "date": ATTENDANCE_DATES[0], "status": "Present"}]
        )
        stored = json.loads(self.data_file.read_text(encoding="utf-8"))
        student = stored["students"][0]
        self.assertEqual(count, 1)
        self.assertEqual(student["present_days"], 1)
        self.assertEqual(student["attendance_percentage"], 14.29)
        self.assertEqual(student["attendance"][ATTENDANCE_DATES[0]], "Present")

    def test_invalid_batch_does_not_partially_change_data(self) -> None:
        with self.assertRaisesRegex(AttendanceDataError, "unknown roll number"):
            self.manager.apply_updates(
                [
                    {"roll_no": "1", "date": ATTENDANCE_DATES[0], "status": "Present"},
                    {"roll_no": "unknown", "date": ATTENDANCE_DATES[1], "status": "Present"},
                ]
            )
        self.assertEqual(self.manager.students()[0]["present_days"], 0)


class AssistantValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assistant = AttendanceAssistant(
            api_key="test-key",
            model="test-model",
            base_url="https://example.com/v1/chat/completions",
        )

    def test_rejects_malformed_update(self) -> None:
        response = {
            "action": "update_attendance",
            "message": "Updated.",
            "updates": [
                {"roll_no": "1", "date": "2026-10-15", "status": "Present"}
            ],
        }
        with self.assertRaises(AIRequestError):
            self.assistant._validate_response(response)

    def test_rejects_non_string_action_without_crashing(self) -> None:
        response = {
            "action": ["update_attendance"],
            "message": "No update.",
            "updates": [],
        }
        with self.assertRaises(AIRequestError):
            self.assistant._validate_response(response)

    def test_valid_update_is_saved_only_after_response_validation(self) -> None:
        class FakeManager:
            data = {"students": []}
            summary = lambda _self: {}
            apply_updates = lambda _self, updates: len(updates)

        api_json = json.dumps(
            {
                "action": "update_attendance",
                "message": "Attendance recorded.",
                "updates": [
                    {
                        "roll_no": "1",
                        "date": ATTENDANCE_DATES[0],
                        "status": "Present",
                    }
                ],
            }
        )
        with patch.object(self.assistant, "_call_model", return_value=api_json):
            result = self.assistant.handle_request("Mark roll 1 present", FakeManager())
        self.assertEqual(result.updated_count, 1)


if __name__ == "__main__":
    unittest.main()
