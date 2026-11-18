import re
import sys
import os
from PyPDF2 import PdfReader

# -------------------------------------------------------------------------
# 1. Разработка регулярных выражений (Task 1)
# -------------------------------------------------------------------------

def get_regex_pattern():
    """
    Возвращает скомпилированное регулярное выражение.
    
    Логика:
    1. Ищем ключевое слово (рисунок, таблица, от, до...).
    2. После него ищем ОДИН ИЛИ БОЛЕЕ символов, которые являются "разрываемыми" пробелами.
       Сюда входят: пробел, табуляция (\t), новая строка (\n), возврат каретки (\r),
       вертикальная табуляция (\v), перевод страницы (\f).
    3. Ищем число.
    
    Важно: Мы НЕ используем просто \s, так как в некоторых реализациях \s может 
    включать и неразрывный пробел (\u00A0), а нам нужно найти именно его отсутствие.
    Поэтому мы явно перечисляем "плохие" символы.
    """
    # Список слов
    keywords = r'(рисунок|рис\.?|таблица|табл\.?|от|до|менее|свыше|более)'
    
    # [ \t\n\r\f\v]+ - один или более символов стандартной пустоты (включая перенос строки!)
    bad_spaces = r'[ \t\n\r\f\v]+'
    
    digits = r'(\d+)'
    
    pattern_str = fr'(?i)\b{keywords}{bad_spaces}{digits}'
    return re.compile(pattern_str)

# -------------------------------------------------------------------------
# 2. Реализация функции поиска (Task 2)
# -------------------------------------------------------------------------

def find_missing_nbsp_in_pdf(file_path):
    """
    Ищет в PDF-файле "опасные" разрывы.
    
    Возвращает:
        (status, list_of_matches)
    """
    matches_list = []
    regex = get_regex_pattern()
    
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"Ошибка при открытии файла: {e}")
        return False, []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        
        if not text:
            continue

        # Ищем по всему тексту страницы сразу, чтобы поймать разрыв "Рисунок \n 1"
        for match in regex.finditer(text):
            found_str = match.group(0)
            
            # Определяем номер строки.
            # match.start() - индекс начала совпадения в общем тексте страницы.
            # Считаем количество переносов строк (\n) до этого места.
            # +1, так как нумерация строк для человека начинается с 1.
            line_number = text.count('\n', 0, match.start()) + 1
            
            # Для красоты вывода заменяем переносы строк внутри найденного фрагмента на пробел,
            # чтобы в консоли таблица не ломалась.
            display_str = found_str.replace('\n', ' [LF] ').replace('\r', '')
            
            coordinates = (page_num + 1, line_number)
            matches_list.append((display_str, coordinates))

    has_errors = len(matches_list) > 0
    return has_errors, matches_list

# -------------------------------------------------------------------------
# 3. Реализация программного обеспечения (Task 3)
# -------------------------------------------------------------------------

def main():
    print("=== Проверка PDF на отсутствие неразрывных пробелов ===")
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Введите путь к pdf-файлу: ").strip()
        file_path = file_path.replace('"', '').replace("'", "")

    if not os.path.exists(file_path):
        print("Файл не найден.")
        return

    print(f"\nАнализ файла: {file_path} ...")
    
    is_found, results = find_missing_nbsp_in_pdf(file_path)

    if is_found:
        print(f"\n[!] Найдено потенциальных ошибок: {len(results)}")
        print("Найдены сочетания с обычными пробелами или уже разорванные переносом строки:")
        print(f"{'Стр.':<5} | {'Строка':<7} | {'Фрагмент (LF = перенос)'}")
        print("-" * 50)
        
        for text_fragment, coords in results:
            page, line = coords
            # Обрезаем слишком длинные фрагменты для красивого вывода
            display_text = (text_fragment[:45] + '..') if len(text_fragment) > 45 else text_fragment
            print(f"{page:<5} | {line:<7} | {display_text}")
    else:
        print("\n[OK] Ошибок не найдено. Везде используются неразрывные пробелы.")

if __name__ == "__main__":
    main()