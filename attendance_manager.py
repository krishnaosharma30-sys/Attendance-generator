"""JSON storage and Python-side attendance calculations."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any


ATTENDANCE_DATES = tuple(
    date.fromisoformat(f"2026-10-{day:02d}").isoformat() for day in range(8, 15)
)
VALID_STATUSES = {"Present", "Absent"}


class AttendanceDataError(Exception):
    """Raised when attendance data is missing, invalid, or cannot be saved."""


class AttendanceManager:
    def __init__(self, data_file: Path) -> None:
        self.data_file = data_file
        self.data: dict[str, Any] = {}

    @property
    def date_range_label(self) -> str:
        return f"{ATTENDANCE_DATES[0]} to {ATTENDANCE_DATES[-1]}"

    def load(self) -> None:
        try:
            with self.data_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as error:
            raise AttendanceDataError(
                f"data file not found: {self.data_file}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise AttendanceDataError(f"cannot read JSON data: {error}") from error

        self._validate_data(data)
        self.data = data
        self.recalculate()

    def _validate_data(self, data: Any) -> None:
        if not isinstance(data, dict) or not isinstance(data.get("students"), list):
            raise AttendanceDataError("top-level JSON must contain a students list")

        seen_roll_numbers: set[str] = set()
        for index, student in enumerate(data["students"]):
            if not isinstance(student, dict):
                raise AttendanceDataError(f"student at index {index} must be an object")
            name = student.get("name")
            roll_no = student.get("roll_no")
            attendance = student.get("attendance")
            if not isinstance(name, str) or not name.strip():
                raise AttendanceDataError(f"student at index {index} has an invalid name")
            if not isinstance(roll_no, str) or not roll_no.strip():
                raise AttendanceDataError(
                    f"student at index {index} has an invalid roll_no"
                )
            if roll_no in seen_roll_numbers:
                raise AttendanceDataError(f"duplicate roll_no: {roll_no}")
            seen_roll_numbers.add(roll_no)
            if not isinstance(attendance, dict):
                raise AttendanceDataError(f"attendance for roll_no {roll_no} must be an object")
            if set(attendance) != set(ATTENDANCE_DATES):
                raise AttendanceDataError(
                    f"attendance for roll_no {roll_no} must contain exactly the "
                    "seven supported dates"
                )
            for day, status in attendance.items():
                if not isinstance(status, str) or status not in VALID_STATUSES:
                    raise AttendanceDataError(
                        f"invalid status for roll_no {roll_no} on {day}: {status!r}"
                    )

    def students(self) -> list[dict[str, Any]]:
        return self.data["students"]

    def summary(self) -> dict[str, Any]:
        """Return fresh summary metrics after recalculating each student's values."""
        self.recalculate()
        percentages = [
            student["attendance_percentage"] for student in self.students()
        ]
        class_average = round(sum(percentages) / len(percentages), 2) if percentages else 0.0
        return {
            "student_count": len(self.students()),
            "class_average_percentage": class_average,
            "students": [
                {
                    "name": student["name"],
                    "roll_no": student["roll_no"],
                    "total_days": student["total_days"],
                    "present_days": student["present_days"],
                    "attendance_percentage": student["attendance_percentage"],
                }
                for student in self.students()
            ],
        }

    def recalculate(self) -> None:
        for student in self.students():
            student["total_days"] = len(ATTENDANCE_DATES)
            student["present_days"] = sum(
                status == "Present" for status in student["attendance"].values()
            )
            student["attendance_percentage"] = round(
                student["present_days"] / student["total_days"] * 100, 2
            )

    def apply_updates(self, updates: list[dict[str, str]]) -> int:
        """Validate all updates before changing and persisting any attendance."""
        student_by_roll = {
            student["roll_no"]: student for student in self.students()
        }
        checked: list[tuple[dict[str, Any], str, str]] = []
        seen: set[tuple[str, str]] = set()

        for update in updates:
            roll_no = update["roll_no"]
            day = update["date"]
            status = update["status"]
            if roll_no not in student_by_roll:
                raise AttendanceDataError(f"unknown roll number: {roll_no}")
            if day not in ATTENDANCE_DATES:
                raise AttendanceDataError(f"unsupported attendance date: {day}")
            if status not in VALID_STATUSES:
                raise AttendanceDataError(f"unsupported attendance status: {status}")
            key = (roll_no, day)
            if key in seen:
                raise AttendanceDataError(
                    f"duplicate update for roll_no {roll_no} on {day}"
                )
            seen.add(key)
            checked.append((student_by_roll[roll_no], day, status))

        original_statuses = [
            (student, day, student["attendance"][day])
            for student, day, _ in checked
        ]
        for student, day, status in checked:
            student["attendance"][day] = status
        self.recalculate()

        try:
            self.save()
        except AttendanceDataError:
            for student, day, previous_status in original_statuses:
                student["attendance"][day] = previous_status
            self.recalculate()
            raise
        return len(checked)

    def save(self) -> None:
        """Write atomically so a failed save cannot leave a partial JSON file."""
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.data_file.parent,
                delete=False,
            ) as file:
                temp_name = file.name
                json.dump(self.data, file, indent=2)
                file.write("\n")
            os.replace(temp_name, self.data_file)
        except OSError as error:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
            raise AttendanceDataError(f"cannot save JSON data: {error}") from error
