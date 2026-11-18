import PyPDF2
import re
from typing import Tuple, List

def check_non_breaking_space(filename: str) -> Tuple[bool, List[Tuple[str, int, int]]]:
    """
    Проверяет наличие неразрывного пробела в специальных сочетаниях слов и чисел.
    Ошибка - если неразрывного пробела НЕТ в сочетаниях:
    "рисунок 1", "таблица 2", "от 1 до 5", "менее 5", "свыше 25"
    Также проверяет сокращения "рис." и "таб."
    
    Args:
        filename (str): Путь к PDF-файлу
        
    Returns:
        Tuple[bool, List[Tuple[str, int, int]]]: 
        - bool: True если неразрывные пробелы присутствуют (все корректно), False если отсутствуют
        - List[Tuple[str, int, int]]: Список найденных ошибок в формате (найденная строка, номер страницы, номер строки)
    """
    
    # Список для хранения найденных ошибок (отсутствие неразрывного пробела)
    errors = []
    
    try:
        # Открываем PDF файл
        with open(filename, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Проходим по каждой странице
            for page_num, page in enumerate(pdf_reader.pages, 1):
                # Извлекаем текст со страницы
                text = page.extract_text()
                
                # Поиск всех специальных сочетаний с помощью регулярных выражений
                # Сочетания: "рисунок 1", "таблица 2", "от 1 до 5", "менее 5", "свыше 25"
                # Сокращения: "рис. 1", "таб. 2"
                
                # Паттерн для поиска сочетаний с пробелами (не неразрывными)
                patterns = [
                    r'(?i)(?:рисунок|рис\.?)\s+\d+',
                    r'(?i)(?:таблица|таб\.?)\s+\d+',
                    r'(?i)от\s+\d+',
                    r'(?i)до\s+\d+',
                    r'(?i)менее\s+\d+',
                    r'(?i)свыше\s+\d+'
                ]
                
                # Проверяем текст на наличие всех типов сочетаний
                for pattern in patterns:
                    # Ищем все совпадения в тексте
                    matches = re.finditer(pattern, text)
                    
                    for match in matches:
                        found_text = match.group()
                        
                        # Проверяем, есть ли неразрывный пробел в найденном тексте
                        if '\xa0' not in found_text:
                            # Нужно определить номер строки - ищем в исходном тексте
                            lines = text.split('\n')
                            line_number = 1
                            pos = 0
                            
                            # Находим строку, в которой находится совпадение
                            for i, line in enumerate(lines, 1):
                                if match.start() >= pos and match.start() < pos + len(line) + 1:
                                    line_number = i
                                    break
                                pos += len(line) + 1
                            
                            errors.append((found_text, page_num, line_number))
                
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        return False, []
    
    # Возвращаем результат
    # Если ошибок нет (все сочетания имеют неразрывные пробелы) - True
    # Если есть ошибки (отсутствуют неразрывные пробелы) - False
    return len(errors) == 0, errors

def main():
    """
    Основная функция программы
    """
    # Запрашиваем путь к PDF файлу у пользователя
    pdf_path = input("Введите путь к PDF-файлу: ").strip()
    
    if not pdf_path:
        print("Путь к файлу не указан")
        return
    
    # Выполняем проверку
    is_correct, errors = check_non_breaking_space(pdf_path)
    
    # Выводим результаты
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТ ПРОВЕРКИ")
    print("="*50)
    
    if is_correct:
        print("Все специальные сочетания содержат неразрывные пробелы")
        print("Правильно: рисунок\xa01, таблица\xa02")
        print("Правильно: от\xa01, до\xa05")
        print("Правильно: менее\xa05, свыше\xa025")
        print("Правильно: рис.\xa01, таб.\xa02")
    else:
        print("Найдены ошибки - отсутствуют неразрывные пробелы:")
        print(f"  Найдено {len(errors)} проблемных сочетаний:")
        
        for i, (found_text, page_num, line_num) in enumerate(errors, 1):
            print(f"  {i}. Строка: '{found_text}'")
            print(f"     Страница: {page_num}, строка: {line_num}")
    
    print("="*50)

if __name__ == "__main__":
    main()
