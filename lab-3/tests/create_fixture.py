from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from odf.opendocument import OpenDocumentText
from odf.table import Table, TableCell, TableRow
from odf.text import H, P


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Fixture directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    cats = root / "Кошки"
    cats.mkdir()
    (cats / "кот_01.txt").write_text(
        "Кот ест рыбу. Кошка спит. Кот и кошка домашние животные.",
        encoding="utf-8",
    )
    (cats / "кот_02.html").write_text(
        "<html><head><style>Скрытый стиль</style><script>bad()</script></head>"
        "<body><h1>Домашние кошки</h1><p>Кошка любит рыбу, кот спит.</p></body></html>",
        encoding="utf-8",
    )

    sport = root / "Спорт"
    sport.mkdir()
    docx = Document()
    docx.add_heading("Футбольный матч", level=1)
    docx.add_paragraph("Команда забила гол и выиграла матч.")
    table = docx.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Игрок"
    table.cell(0, 1).text = "Гол"
    docx.save(sport / "спорт_01.docx")

    odt = OpenDocumentText()
    odt.text.addElement(H(outlinelevel=1, text="Спортивная тренировка"))
    odt.text.addElement(P(text="Спортсмен тренируется и готовится к матчу."))
    table_odt = Table()
    row = TableRow()
    for value in ("Команда", "Победа"):
        cell = TableCell()
        cell.addElement(P(text=value))
        row.addElement(cell)
    table_odt.addElement(row)
    odt.text.addElement(table_odt)
    odt.save(str(sport / "спорт_02.odt"))

    encoding = root / "Кодировка"
    encoding.mkdir()
    (encoding / "cp1251.txt").write_bytes(
        "Проверка файла в кодировке CP1251: ёж, чай и текст.".encode("cp1251")
    )

    empty = root / "Пустые"
    empty.mkdir()
    (empty / "empty.txt").write_text("", encoding="utf-8")

    print(root)


if __name__ == "__main__":
    main()
