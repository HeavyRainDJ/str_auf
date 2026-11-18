import json
import os
import glob

def parse_yagpt_json(input_filename, output_file):
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем структуру. Обычно история лежит в "messages"
        messages = data.get("messages", [])
        
        for msg in messages:
            content = msg.get("content", {})
            
            # --- Обработка запроса пользователя ---
            if "request" in content:
                request_data = content["request"]
                text = request_data.get("request_text", "")
                
                output_file.write("User:\n")
                output_file.write(text.strip())
                output_file.write("\n\n")
            

            # --- Обработка ответа модели ---
            if "response" in content:
                response_data = content["response"]
                output_file.write("Model:\n")
                # Основной текст ответа лежит в plain_response_text
                for card in response_data.get("cards"):
                    output_file.write(f"{'- '*20}\n\n")
                    text = card.get("text_card", {}).get("text", "")
                    output_file.write(text.strip())
                    output_file.write("\n\n")
                    
                
                # Разделитель между парами сообщений (если их несколько в одном файле)
                output_file.write(f"{'-'*40}\n\n")
                
    except Exception as e:
        print(f"Ошибка при обработке файла {input_filename}: {e}")

def main():
    # Имя итогового файла
    
    
    # Находим все json файлы в текущей папке
    json_files = glob.glob("*.json")
    
    if not json_files:
        print("JSON файлы не найдены в текущей директории.")
        return

    print(f"Найдено файлов: {len(json_files)}. Обработка...")
    for json_file in json_files:
        output_filename = f"{json_file[:-5]}_res.txt"
        with open(output_filename, 'w', encoding='utf-8') as outfile:
        
            parse_yagpt_json(json_file, outfile)
            print(f"Обработан: {json_file}")

    print(f"\nГотово! Результат сохранен в файл: {output_filename}")

if __name__ == "__main__":
    main()