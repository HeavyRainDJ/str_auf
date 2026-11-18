import re
import PyPDF2

def find_non_breaking_spaces(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        # Регулярное выражение для поиска сочетаний «рисунок 1», «таблица 2», «рис. 1», «таб. 2», «от 1 до 5», «менее 5», «свыше 25»
        pattern = re.compile(r'\b(рисунок\s+\d+|таблица\s+\d+|рис\.\s+\d+|таб\.\s+\d+|от\s+\d+|до\s+\d+|менее\s+\d+|свыше\s+\d+)\b', re.IGNORECASE)
        result = {
            'наличие_ошибки_с_пробелом': False,
            'найденные_строки': []
        }
        
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                for match in pattern.finditer(text):
                    found_string = match.group(0)
                    # Проверяем отсутствие неразрывного пробела (Unicode: \u00A0)
                    if '\u00A0' not in found_string:
                        result['наличие_ошибки_с_пробелом'] = True
                    
                    # Определяем номер строки, в которой найдено вхождение
                    line_num = text[:match.start()].count('\n') + 1
                    result['найденные_строки'].append((found_string, (page_num + 1, line_num)))
    
    return (result['наличие_ошибки_с_пробелом'], result['найденные_строки'])

def main():
    pdf_path = input("Введите путь к PDF-файлу: ")
    result = find_non_breaking_spaces(pdf_path)
    print("Наличие ошибки с неразрывным пробелом:", result[0])
    if result[1]:
        print("Найденные строки с ошибками:")
        for item in result[1]:
            print(f"Строка: {item[0]}, координаты: страница {item[1][0]}, строка {item[1][1]}")
    else:
        print("Ошибок с неразрывными пробелами не найдено.")

if __name__ == "__main__":
    main()
