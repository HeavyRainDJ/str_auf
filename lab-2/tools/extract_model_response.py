from __future__ import annotations

import argparse
import re
from pathlib import Path


FILE_MARKER = re.compile(r"^FILE:\s*(.+?)\s*$", re.MULTILINE)


def extract(source: Path, output_dir: Path) -> list[Path]:
    text = source.read_text(encoding="utf-8-sig")
    markers = list(FILE_MARKER.finditer(text))
    if not markers:
        raise ValueError("No FILE markers found")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index, marker in enumerate(markers):
        filename = marker.group(1).strip()
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe filename: {filename}")

        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        segment = text[start:end].strip()
        lines = segment.splitlines()

        if not lines or not lines[0].startswith("```"):
            raise ValueError(f"Missing opening fence for {filename}")
        lines = lines[1:]

        while lines and not lines[-1].strip():
            lines.pop()
        if not lines or lines[-1].strip() != "```":
            raise ValueError(f"Missing closing fence for {filename}")
        lines.pop()

        target = output_dir / filename
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(target)

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    for path in extract(args.source, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
