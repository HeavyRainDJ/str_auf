from pathlib import Path
import re
src = Path(r'C:/Users/user/Desktop/answer-2.txt').read_text(encoding='utf-8')
code = re.search(r'```python\r?\n(.*?)\r?\n```', src, re.S).group(1)
out = Path(r'C:/Users/user/Desktop/Строганов/Лаб-1/Выполнение/models/korolev/02')
out.mkdir(parents=True, exist_ok=True)
(out/'solution.py').write_text(code, encoding='utf-8')
(out/'response.md').write_text(src, encoding='utf-8')
