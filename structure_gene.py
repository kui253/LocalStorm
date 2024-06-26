import re
import time
import json
import pickle
import random
import sys
import requests
from tqdm import tqdm
from src.file_man import FileManager

url = "http://127.0.0.1:10011/qa_generate"


OUTLINE = "You need to organize the content relationship based on the given content and generate a content outline. \
    1. If the document contains a content outline, you can refer to it. \
    2. Note that you only need to organize the text content and should avoid using the original text content unless absolutely necessary. \
    3. The output format of the content outline should follow the markdown multi-level title format, such as the first-level title starting with # and the second-level title starting with ##. \
    4. Output one title at a time, regardless of the title level, and each time output \n to separate lines."

data_dir, output_dir = sys.argv[1], sys.argv[2]

fm = FileManager(data_dir)
section_content_list = fm.get_all_files()
for n, section in tqdm(enumerate(section_content_list)):
    out_file_name = section["name"]
    contents = section["content"].page_content
    question_entity = f"content you need to refer：{contents}"

    data = {
        "system_prompt": OUTLINE,
        "context_list": [question_entity],
    }
    time.sleep(0.5)
    response_entity = requests.post(url, json=data)
    if response_entity.status_code == 200:
        json_data_entity = response_entity.json()
        answer_entity = json_data_entity["response"]
        with open(f"{output_dir}/{out_file_name}", "w", encoding="utf-8") as f:
            f.write(answer_entity[0])
