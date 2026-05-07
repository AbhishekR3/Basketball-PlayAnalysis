#!/usr/bin/env python3
"""
Basketball Play Analysis — Implementation Agent

An AI agent powered by Claude that reads your codebase and implements ideas
you describe in plain English.

Usage:
    python implementation_agent.py "Add acceleration features to Feature_Engineering.py"
    python implementation_agent.py          # interactive prompt
    ANTHROPIC_API_KEY=sk-... python implementation_agent.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.resolve()

SYSTEM_PROMPT = """You are an expert ML/computer-vision engineer implementing improvements to the
Basketball-PlayAnalysis project. You have full access to the codebase via tools.

## Pipeline overview
1. Passing_Simulation.py / RandomMovement_Simulation.py  — Pygame-based video generation
2. Object_Tracking.py      — YOLOv10s + DeepSORT multi-object tracking (custom deep_sort/)
3. Feature_Engineering.py  — Temporal/spatial feature extraction, PCA optimisation
4. Neural_Network.py       — Bidirectional LSTM + self-attention, pruning, augmentation
5. Data_Loading.py         — AWS PostGIS database ingestion (SQLAlchemy + GeoAlchemy2)
6. utils.py                — Shared CSV I/O and logging helpers

## Tech stack
Python 3.11, PyTorch 2.3.0, TensorFlow 2.16.1, Ultralytics YOLO 8.2.56,
OpenCV 4.9 headless, DeepSORT (deep_sort/), Pandas 2.1.4, NumPy 1.23.5,
SciPy 1.9.3, Scikit-learn 1.1.3, Pygame 2.5.2, Boto3 1.35.4,
SQLAlchemy 2.0.25, GeoAlchemy2 0.15.2, Pytest 7.4.0

## How to implement an idea
1. Read every file relevant to the idea before writing anything.
2. Understand the existing patterns, variable names, and style in those files.
3. Make targeted, minimal edits that integrate naturally with existing code.
4. After writing, verify syntax with check_python_syntax.
5. Summarise every file you changed and why.

Never truncate existing code — always keep what was already there unless you're
explicitly removing something the user asked you to remove.
"""

MAX_TOKENS = 16000

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _resolve(path: str) -> Path:
    """Resolve path relative to project root; refuse to escape it."""
    p = (PROJECT_ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if not str(p).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path outside project root: {path}")
    return p


def tool_read_file(path: str) -> str:
    try:
        return _resolve(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: file not found — {path}"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def tool_write_file(path: str, content: str) -> str:
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content):,} chars to {path}"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


def tool_edit_file(path: str, old_string: str, new_string: str) -> str:
    try:
        p = _resolve(path)
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return "ERROR: old_string not found in file — check for exact whitespace/indentation"
        if count > 1:
            return f"ERROR: old_string appears {count} times — add more surrounding context to make it unique"
        p.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {path}"
    except FileNotFoundError:
        return f"ERROR: file not found — {path}"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR editing {path}: {e}"


def tool_list_directory(path: str = ".") -> str:
    try:
        p = _resolve(path)
        lines = []
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue
            tag = "dir " if item.is_dir() else "file"
            size = "" if item.is_dir() else f"  {item.stat().st_size:>10,} B"
            lines.append(f"[{tag}]  {item.name}{size}")
        return "\n".join(lines) or "(empty)"
    except FileNotFoundError:
        return f"ERROR: directory not found — {path}"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR listing {path}: {e}"


def tool_search_codebase(pattern: str, path: str = ".", file_glob: str = "*.py") -> str:
    try:
        p = _resolve(path)
        result = subprocess.run(
            ["grep", "-rn", "--include", file_glob, pattern, str(p)],
            capture_output=True, text=True, timeout=20,
        )
        out = result.stdout.strip()
        if not out:
            return f"No matches for '{pattern}'"
        # Keep relative paths in output
        out = out.replace(str(PROJECT_ROOT) + "/", "")
        return out[:6000] + ("\n… (truncated)" if len(out) > 6000 else "")
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR searching: {e}"


def tool_check_python_syntax(path: str) -> str:
    try:
        p = _resolve(path)
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(p)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return f"Syntax OK — {path}"
        return f"Syntax error in {path}:\n{result.stderr.strip()}"
    except FileNotFoundError:
        return f"ERROR: file not found — {path}"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR checking syntax: {e}"


# ---------------------------------------------------------------------------
# Tool schemas (JSON Schema)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": (
            "Read a file from the project. Returns full file contents. "
            "Always read a file before editing it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to project root, e.g. 'Feature_Engineering.py' or 'deep_sort/tracker.py'",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write (create or fully overwrite) a file. "
            "Use this to create new files or completely replace small files. "
            "For targeted edits to existing files, prefer edit_file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to project root"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace one exact occurrence of old_string with new_string inside a file. "
            "old_string must be unique in the file — include enough surrounding context "
            "(function signature, indentation, neighbouring lines) to make it unique. "
            "Read the file first so you copy the exact whitespace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to project root"},
                "old_string": {"type": "string", "description": "Exact text to replace (must be unique in the file)"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories at a path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to project root. Defaults to project root.",
                    "default": ".",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_codebase",
        "description": "Grep the codebase for a pattern. Returns file:line:match. Useful for finding where things are defined or called.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or string to search for"},
                "path": {
                    "type": "string",
                    "description": "Directory to search in (relative to project root). Defaults to entire project.",
                    "default": ".",
                },
                "file_glob": {
                    "type": "string",
                    "description": "File glob pattern, e.g. '*.py'. Defaults to '*.py'.",
                    "default": "*.py",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "check_python_syntax",
        "description": "Check Python file for syntax errors without executing it. Run this after writing or editing any .py file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Python file path relative to project root"},
            },
            "required": ["path"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

DISPATCH = {
    "read_file":          lambda a: tool_read_file(a["path"]),
    "write_file":         lambda a: tool_write_file(a["path"], a["content"]),
    "edit_file":          lambda a: tool_edit_file(a["path"], a["old_string"], a["new_string"]),
    "list_directory":     lambda a: tool_list_directory(a.get("path", ".")),
    "search_codebase":    lambda a: tool_search_codebase(a["pattern"], a.get("path", "."), a.get("file_glob", "*.py")),
    "check_python_syntax": lambda a: tool_check_python_syntax(a["path"]),
}


def execute_tool(name: str, args: dict) -> str:
    fn = DISPATCH.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        return fn(args)
    except Exception as e:
        return f"ERROR executing {name}: {e}"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _short(s: str, n: int = 120) -> str:
    s = s.strip()
    return s[:n] + " …" if len(s) > n else s


def _print_tool_call(name: str, args: dict) -> None:
    arg_str = ", ".join(
        f"{k}={repr(v)[:60]}" for k, v in args.items() if k != "content"
    )
    if "content" in args:
        arg_str += f", content=<{len(args['content'])} chars>"
    print(f"\n  ▶ {name}({arg_str})", flush=True)


def _print_tool_result(result: str) -> None:
    lines = result.strip().splitlines()
    preview = "\n    ".join(lines[:6])
    suffix = f"\n    … ({len(lines) - 6} more lines)" if len(lines) > 6 else ""
    print(f"    {preview}{suffix}", flush=True)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(idea: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Please implement the following idea in the Basketball-PlayAnalysis project:\n\n"
                f"{idea.strip()}\n\n"
                f"Start by reading the relevant files, then make the changes."
            ),
        }
    ]

    print(f"\n{'━'*65}")
    print(f"  Basketball Play Analysis — Implementation Agent")
    print(f"{'━'*65}")
    print(f"  Idea: {_short(idea, 200)}")
    print(f"{'━'*65}\n")

    iteration = 0
    max_iterations = 30  # safety ceiling

    while iteration < max_iterations:
        iteration += 1

        # Stream the model response
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOL_SCHEMAS,
            messages=messages,
        ) as stream:
            # Stream text tokens to terminal
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)

            response = stream.get_final_message()

        # Done — no more tool calls
        if response.stop_reason == "end_turn":
            print(f"\n\n{'━'*65}")
            print("  ✓ Implementation complete")
            print(f"{'━'*65}\n")
            break

        # Handle tool calls
        if response.stop_reason != "tool_use":
            print(f"\n[stop_reason={response.stop_reason}]")
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and collect results
        tool_results = []
        for block in tool_use_blocks:
            _print_tool_call(block.name, block.input)
            result = execute_tool(block.name, block.input)
            _print_tool_result(result)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    else:
        print(f"\n[Warning: reached max iterations ({max_iterations})]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        idea = " ".join(sys.argv[1:])
    else:
        print("Basketball Play Analysis — Implementation Agent")
        print("Describe your idea (press Enter twice or Ctrl-D when done):\n")
        lines = []
        try:
            while True:
                line = input()
                if not line and lines and not lines[-1]:
                    break
                lines.append(line)
        except EOFError:
            pass
        idea = "\n".join(lines).strip()
        if not idea:
            print("No idea provided. Exiting.")
            return

    run_agent(idea)


if __name__ == "__main__":
    main()
