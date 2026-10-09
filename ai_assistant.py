"""OpenAI-compatible API client and strict response validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from attendance_manager import ATTENDANCE_DATES, AttendanceDataError, AttendanceManager


class AIConfigurationError(Exception):
    """Raised for invalid AI client configuration."""


class AIRequestError(Exception):
    """Raised when the AI request or response is invalid."""


@dataclass(frozen=True)
class AssistantResult:
    message: str
    updated_count: int = 0


class AttendanceAssistant:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: int = 30,
    ) -> None:
        if not api_key.strip():
            raise AIConfigurationError("API key cannot be empty")
        if not model.strip():
            raise AIConfigurationError("model name cannot be empty")
        if not base_url.startswith(("https://", "http://")):
            raise AIConfigurationError("API base URL must use HTTP or HTTPS")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def handle_request(
        self, request_text: str, manager: AttendanceManager
    ) -> AssistantResult:
        payload = self._build_payload(request_text, manager)
        response_content = self._call_model(payload)
        response = self._parse_response(response_content)
        action, message, updates = self._validate_response(response)

        if action == "update_attendance":
            try:
                updated_count = manager.apply_updates(updates)
            except AttendanceDataError:
                raise
            return AssistantResult(message=message, updated_count=updated_count)
        return AssistantResult(message=message)

    def _build_payload(
        self, request_text: str, manager: AttendanceManager
    ) -> dict[str, Any]:
        system_message = (
            "You are an assistant for a classroom attendance system. Interpret the "
            "user's request using the supplied JSON data. Return only one JSON object "
            "with exactly these keys: action, message, updates. action must be "
            "'answer', 'update_attendance', or 'unsupported'. message must be a "
            "helpful string. updates must be an array of objects containing exactly "
            "roll_no, date, and status. Use updates only for explicit attendance "
            "changes; dates must be one of the supported ISO dates and status must "
            "be 'Present' or 'Absent'. Never invent student identifiers or dates. "
            "For questions about percentages or class average, use the supplied "
            "Python-calculated summary. For unsupported or ambiguous changes, ask "
            "the user to clarify and return action 'unsupported' with no updates. "
            "Never claim an update was saved; Python saves validated updates."
        )
        user_content = {
            "request": request_text,
            "supported_dates": list(ATTENDANCE_DATES),
            "attendance_data": manager.data,
            "calculated_summary": manager.summary(),
        }
        return {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": json.dumps(user_content, ensure_ascii=False),
                },
            ],
        }

    def _call_model(self, payload: dict[str, Any]) -> str:
        request = Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                api_response = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise AIRequestError(
                f"AI service returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise AIRequestError(f"could not reach AI service: {error}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIRequestError("AI service returned invalid JSON") from error

        try:
            content = api_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise AIRequestError(
                "AI service response is missing choices[0].message.content"
            ) from error
        if not isinstance(content, str):
            raise AIRequestError("AI response content must be a JSON string")
        return content

    def _parse_response(self, content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise AIRequestError("AI response content is not valid JSON") from error

    def _validate_response(
        self, response: Any
    ) -> tuple[str, str, list[dict[str, str]]]:
        if not isinstance(response, dict):
            raise AIRequestError("AI response must be a JSON object")
        if set(response) != {"action", "message", "updates"}:
            raise AIRequestError(
                "AI response must contain exactly action, message, and updates"
            )

        action = response["action"]
        message = response["message"]
        updates = response["updates"]
        if not isinstance(action, str) or action not in {
            "answer",
            "update_attendance",
            "unsupported",
        }:
            raise AIRequestError(f"unsupported AI action: {action!r}")
        if not isinstance(message, str) or not message.strip():
            raise AIRequestError("AI response message must be a non-empty string")
        if not isinstance(updates, list):
            raise AIRequestError("AI response updates must be an array")

        validated_updates: list[dict[str, str]] = []
        for index, update in enumerate(updates):
            if not isinstance(update, dict) or set(update) != {
                "roll_no",
                "date",
                "status",
            }:
                raise AIRequestError(
                    f"update at index {index} must contain exactly roll_no, date, "
                    "and status"
                )
            roll_no, day, status = (
                update["roll_no"],
                update["date"],
                update["status"],
            )
            if not isinstance(roll_no, str) or not roll_no.strip():
                raise AIRequestError(f"update at index {index} has invalid roll_no")
            if not isinstance(day, str) or day not in ATTENDANCE_DATES:
                raise AIRequestError(
                    f"update at index {index} has unsupported date: {day!r}"
                )
            if not isinstance(status, str) or status not in {"Present", "Absent"}:
                raise AIRequestError(
                    f"update at index {index} has invalid status: {status!r}"
                )
            validated_updates.append(
                {"roll_no": roll_no, "date": day, "status": status}
            )

        if action == "update_attendance" and not validated_updates:
            raise AIRequestError("update_attendance action must include at least one update")
        if action != "update_attendance" and validated_updates:
            raise AIRequestError(f"{action} action cannot contain attendance updates")
        return action, message.strip(), validated_updates
