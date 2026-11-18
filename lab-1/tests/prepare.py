from pathlib import Path
import json, hashlib
from docx import Document
from odf.opendocument import OpenDocumentText
from odf.text import P, Span
from odf.table import Table, TableRow, TableCell

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures'
expected = {}
def txt(case, name, text):
    d = FIX / case; d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding='utf-8')
    expected.setdefault(case, {})[name] = text

txt('basic', 'a.txt', 'Кот ест рыбу. Кот спит.\nЁж и кот.')
txt('basic', 'b.txt', 'Кот ест рыбу.')
txt('basic', 'empty.txt', '')
txt('all_empty', 'a.txt', '')
txt('all_empty', 'b.txt', ' \n\t')
(FIX / 'empty_dir').mkdir(parents=True, exist_ok=True)
expected['empty_dir'] = {}
txt('repeated', 'a.txt', 'аааа а Я ёж ЁЖ кот123кот')
txt('formats', 'a.txt', 'Кот ест рыбу.\nЁж спит.')
d = FIX / 'formats'
(d/'b.html').write_text('<html><head><style>стиль</style><script>секрет</script></head><body><p>Кот ест рыбу.</p><table><tr><td>Ёж спит.</td></tr></table></body></html>', encoding='utf-8')
doc = Document(); doc.add_paragraph('Кот ест рыбу.'); doc.add_table(rows=1, cols=1).cell(0, 0).text = 'Ёж спит.'; doc.save(d/'c.docx')
odt = OpenDocumentText(); p = P(); p.addText('Кот '); p.addElement(Span(text='ест')); p.addText(' рыбу.'); odt.text.addElement(p)
t = Table(); r = TableRow(); c = TableCell(); c.addElement(P(text='Ёж спит.')); r.addElement(c); t.addElement(r); odt.text.addElement(t); odt.save(str(d/'d.odt'))
for name in ['b.html', 'c.docx', 'd.odt']:
    expected['formats'][name] = 'Кот ест рыбу.\nЁж спит.'
bad = FIX/'corrupt'; bad.mkdir(exist_ok=True); (bad/'broken.docx').write_bytes(b'not a zip')
(ROOT/'tests'/'expected_texts.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')
source = ROOT.parent/'тексты для тестов'
manifest = [{'path': p.relative_to(source).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source.rglob('*')) if p.is_file()]
(ROOT/'tests'/'corpus_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('Prepared', len(expected), 'control corpora;', len(manifest), 'original files')
