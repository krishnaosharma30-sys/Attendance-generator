# AI Attendance Management System

A small terminal-based Python application that keeps classroom attendance in
`attendance.json`. It sends each natural-language request and the attendance
data to Hugging Face's OpenAI-compatible chat-completions API. Python validates
all model responses, calculates the attendance percentages, and applies
validated attendance changes to the JSON file.

## Requirements

- Python 3.10 or newer
- A Hugging Face access token with permission to use an inference provider

No third-party Python packages are required.

## Run on Windows PowerShell

Create a `.env` file in the project directory (next to `main.py`) with your
Hugging Face token:

```text
HF_TOKEN=hf_your-token
```

The token is available from your Hugging Face account settings. Do not paste it
into the source code or commit the `.env` file. Then start the terminal app:

```powershell
python main.py
```

The app reads `.env` automatically. Values already set in the terminal
environment take precedence over values in `.env`. For compatibility, if the
project-folder `.env` file is absent, the app also checks `.venv/.env`.

The default model is `Qwen/Qwen2.5-7B-Instruct`. Choose a model enabled for
Hugging Face Inference Providers on your account and override the default if
needed:

```powershell
$env:HF_MODEL = "provider/model-name"
```

Optional overrides are `HF_BASE_URL` (defaults to
`https://router.huggingface.co/v1/chat/completions`), `HF_MODEL`, and
`HF_TOKEN`. The app also recognizes `HUGGINGFACEHUB_API_TOKEN` and the
`OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` environment variables
for compatibility with other OpenAI-compatible services.

Enter a question or attendance change at the `You:` prompt. Type `exit` or
`quit` to leave. For example:

- `Show the attendance percentage for Rahul Sharma`
- `What is the class average?`
- `Mark roll number 101 absent on 2026-10-12`

The supported dates are `2026-10-08` through `2026-10-14`, inclusive. Statuses
are stored as `Present` or `Absent`. Student percentages and the class average
are recalculated in Python; the class average is the arithmetic mean of the
students' individual percentages.

## Data and tests

`attendance.json` starts with five sample students. Updates are validated
before any changes are made and written atomically so the JSON file remains
consistent. To run the standard-library unit tests:

```powershell
python -m unittest -v
```
