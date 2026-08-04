"""File access tools: read_file and list_directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .registry import ToolRegistry

if TYPE_CHECKING:
    from ..config import Config


def register_file_tools(
    registry: ToolRegistry,
    config: Config,
) -> None:
    """Register read_file and list_directory tools."""

    def _resolve_path(path: str) -> Path:
        """Resolve a path relative to repo root, preventing directory traversal."""
        root = config.repo_root
        target = (root / path).resolve()
        # Ensure the resolved path is within the repo root
        if not str(target).startswith(str(root)):
            raise ValueError(f"Path '{path}' is outside the repository root")
        return target

    async def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read the contents of a file in the repository.

        Args:
            path: File path relative to the repository root.
            start_line: First line to read (1-indexed, default 1).
            end_line: Last line to read (0 = read to end).
        """
        try:
            target = _resolve_path(path)
        except ValueError as e:
            return str(e)

        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file():
            return f"Not a file: {path}"

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading file: {e}"

        lines = text.split("\n")
        start = max(1, start_line)
        end = len(lines) if end_line <= 0 else min(end_line, len(lines))

        selected = lines[start - 1:end]
        # Add line numbers
        numbered = []
        for i, line in enumerate(selected, start):
            numbered.append(f"{i:4d} | {line}")

        result = "\n".join(numbered)
        if len(result) > 8000:
            result = result[:8000] + "\n... [truncated]"
        return f"```\n{result}\n```"

    registry.register(
        name="read_file",
        description=(
            "Read the contents of a file in the repository. "
            "Supports reading specific line ranges. "
            "Use this after search_code or get_symbols to read specific code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repository root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read (1-indexed, default 1).",
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (0 = read to end).",
                    "default": 0,
                },
            },
            "required": ["path"],
        },
        handler=read_file,
    )

    async def list_directory(path: str = ".") -> str:
        """List files and directories in the given path.

        Args:
            path: Directory path relative to the repository root (default '.').
        """
        try:
            target = _resolve_path(path)
        except ValueError as e:
            return str(e)

        if not target.exists():
            return f"Directory not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"

        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        exclude = set(config.index.exclude_dirs)

        for entry in entries:
            if entry.name in exclude:
                continue
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
            else:
                size = entry.stat().st_size
                if size > 1024:
                    size_str = f"{size // 1024}K"
                else:
                    size_str = f"{size}B"
                lines.append(f"  {entry.name} ({size_str})")

        return f"Contents of {path}:\n" + "\n".join(lines)

    registry.register(
        name="list_directory",
        description=(
            "List files and subdirectories in a directory. "
            "Useful for exploring the repository structure."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to repository root (default '.').",
                    "default": ".",
                },
            },
        },
        handler=list_directory,
    )
