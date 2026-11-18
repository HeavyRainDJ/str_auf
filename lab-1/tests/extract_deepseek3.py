from pathlib import Path
import re

src = Path(r'C:/Users/user/Desktop/deepseek-3.txt').read_text(encoding='utf-8')
matches = re.findall(r'```python\r?\n(.*?)\r?\n```', src, re.S)
if not matches:
    raise RuntimeError('Python code block not found')
out = Path(r'C:/Users/user/Desktop/Строганов/Лаб-1/Выполнение/models/deepseek/03')
out.mkdir(parents=True, exist_ok=True)
(out / 'solution.py').write_text(matches[-1], encoding='utf-8')
(out / 'response.md').write_text(src, encoding='utf-8')
