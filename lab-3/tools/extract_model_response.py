from __future__ import annotations

import argparse
import re
from pathlib import Path


FENCED = re.compile(
    r"```(?:python|py|text|txt|requirements)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def clean_code_block(block: str) -> str:
    lines = block.strip().splitlines()
    if lines and lines[0].strip().lower() in {"python", "py", "text", "txt", "requirements"}:
        lines.pop(0)
    if lines and lines[0].strip().lower() in {"копировать", "copy"}:
        lines.pop(0)
    return "\n".join(lines).strip() + "\n"


def extract(source: Path, output_dir: Path) -> list[Path]:
    text = source.read_text(encoding="utf-8-sig")
    blocks = [clean_code_block(match) for match in FENCED.findall(text)]

    if not blocks:
        solution_match = re.search(
            r"(?:^|\n)(?:1\.\s*)?Полный\s+solution\.py\s*\n"
            r"(?:python\s*\n)?(?:Копировать\s*\n)?(.*?)"
            r"(?=\n(?:2\.\s*)?Полный\s+requirements\.txt)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        requirements_match = re.search(
            r"(?:^|\n)(?:2\.\s*)?Полный\s+requirements\.txt\s*\n"
            r"(?:text\s*\n|python\s*\n)?(?:Копировать\s*\n)?(.*?)"
            r"(?=\n(?:3\.\s*)?Краткое)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not solution_match or not requirements_match:
            raise ValueError("Could not locate solution.py and requirements.txt")
        blocks = [clean_code_block(solution_match.group(1)), clean_code_block(requirements_match.group(1))]

    python_blocks = [block for block in blocks if "import " in block and "def " in block]
    requirements_blocks = [
        block
        for block in blocks
        if block not in python_blocks and any(name in block.lower() for name in ("numpy", "scipy", "torch"))
    ]
    if not python_blocks or not requirements_blocks:
        raise ValueError("Response does not contain both recognizable files")

    output_dir.mkdir(parents=True, exist_ok=True)
    solution = output_dir / "solution.py"
    requirements = output_dir / "requirements.txt"
    solution.write_text(max(python_blocks, key=len), encoding="utf-8")
    requirements.write_text(requirements_blocks[0], encoding="utf-8")
    return [solution, requirements]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in extract(args.source, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
