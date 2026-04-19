"""Interview source adapter — loads structured YAML interview transcripts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mimir.adapters.base import Chunk

# Expected YAML schema:
# date: "YYYY-MM-DD"
# topic: "..."
# participants: ["name1", "name2"]
# acl: ["internal"]
# reference: "interview://..."   (optional)
# transcript:
#   - speaker: "..."
#     text: "..."


class InterviewAdapter:
    """Load interview transcripts from YAML files and return Chunk objects."""

    def load(self, file_path: str | Path) -> list[Chunk]:
        """Parse *file_path* and return one Chunk per interview.

        Raises FileNotFoundError if the file does not exist.
        Raises ValueError if the YAML is missing required fields.
        """
        path = Path(file_path)
        data: dict[str, Any] = yaml.safe_load(path.read_text())
        self._validate(data, path)
        content = self._format_transcript(data)
        date_str = str(data.get("date", ""))
        topic = str(data.get("topic", path.stem))
        reference = str(data.get("reference", f"interview://{date_str}/{topic}"))
        return [
            Chunk(
                id=f"interview_{date_str}_{topic}".replace(" ", "_"),
                source_type="interview",
                content=content,
                acl=list(data.get("acl", ["internal"])),
                retrieved_at=datetime.now(UTC),
                reference=reference,
                metadata={
                    "date": date_str,
                    "topic": topic,
                    "participants": data.get("participants", []),
                },
            )
        ]

    @staticmethod
    def _validate(data: dict[str, Any], path: Path) -> None:
        if "transcript" not in data:
            raise ValueError(f"Interview YAML {path} missing 'transcript' key")
        for i, turn in enumerate(data["transcript"]):
            if "speaker" not in turn or "text" not in turn:
                raise ValueError(f"Transcript turn {i} missing 'speaker' or 'text' in {path}")

    @staticmethod
    def _format_transcript(data: dict[str, Any]) -> str:
        topic = data.get("topic", "Interview")
        date = data.get("date", "")
        header = f"# {topic}"
        if date:
            header += f" ({date})"
        lines = [header, ""]
        for turn in data.get("transcript", []):
            lines.append(f"{turn['speaker']}: {turn['text']}")
        return "\n".join(lines)
