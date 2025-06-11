import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import json
import time
import pandas as pd
import re
from src.data_loader import load_data
from textwrap import dedent
import unicodedata
import glob

# This function handles a single JSON block and extracts stance data from it
def extract_stance_data(data):
    extracted_data = []
    
    # Handle direct list of entries
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                sentence = entry.get("sentence")
                topic = entry.get("topic")
                stance = entry.get("stance")
                
                if sentence and topic and stance:
                    extracted_data.append({
                        "sentence": sentence.strip(),
                        "topic": topic.strip(),
                        "stance": stance.strip()
                    })
    
    # Handle dict with sentences key            
    elif isinstance(data, dict) and "sentences" in data:
        for entry in data["sentences"]:
            if isinstance(entry, dict):
                sentence = entry.get("sentence")
                topic = entry.get("topic")
                stance = entry.get("stance")
                
                if sentence and topic and stance:
                    extracted_data.append({
                        "sentence": sentence.strip(),
                        "topic": topic.strip(),
                        "stance": stance.strip()
                    })
    
    return extracted_data

def clean_and_parse_json(json_string):
    """Clean a JSON string containing Hebrew and special characters and parse it"""
    # Replace problematic quotes and directional markers
    cleaned = json_string.replace('\u201c', '"').replace('\u201d', '"')  # Smart quotes
    cleaned = cleaned.replace('\u05f4', '"')  # Hebrew gershayim
    cleaned = cleaned.replace('\u200f', '')   # Right-to-left mark
    cleaned = cleaned.replace('\u200e', '')   # Left-to-right mark
    
    # Normalize Unicode and standardize quotes
    cleaned = unicodedata.normalize('NFKD', cleaned)
    cleaned = re.sub(r'[""]', '"', cleaned)
    
    # Parse the cleaned JSON
    return json.loads(cleaned)

# print(os.getenv("OPENAI_API_KEY"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

def get_completion(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5, 
    )
    return response.choices[0].message.content

def get_completion_from_messages(messages, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message.content


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def extract_string_list(array_text):
    # Extract individual string entries using regex
    string_matches = re.findall(r'"(.*?)"', array_text, re.DOTALL)
    return [s.strip() for s in string_matches]

def create_requests(texts, topics):
    requests = []

    # Create requests for each text and topic pair
    for i in range(len(texts)):
        input_sent = texts[i]
        input_topic = topics[i]

        prompt = dedent(f"""
            ### Task:
            Determine the stance of each Hebrew sentence toward a given topic.

            ### Example:
            - Input: "הממשלה עושה כל שביכולתה כדי להילחם בטרור."
            - Topic: "פוליטיקה - ימין"
            - Output: {{
                "sentence": "הממשלה עושה כל שביכולתה כדי להילחם בטרור.",
                "topic": "פוליטיקה - ימין",
                "stance": "תומך"
            }}

            ### Input:
            Sentence: {input_sent}
            Topic: {input_topic}

            ### Output:
            {{
                "sentence": "{input_sent}",
                "topic": "{input_topic}",
                "stance": "בעד" | "נגד" | "נייטרלי"
            }}
        """)

        # print(prompt)
        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an advanced NLP model specializing in stance detection."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000
            }
        }

        requests.append(request)

    return requests



if __name__ == "__main__":
    # load data
    texts = []
    topics = []
    # with open('./scripts/output/topics.jsonl', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         record = json.loads(line)
    #         text = record.get("sentence")
    #         topic = record.get("topic")
    #         if text == "" or topic == "":
    #             continue
    #         texts.append(text)
    #         topics.append(topic)

    # # Create a DataFrame
    # df = pd.DataFrame({
    #     "sentence": texts,
    #     "topic": topics
    # })
    path_to_csv = './Data/topic_stance_dataset.csv'
    df = pd.read_csv(path_to_csv)

    # input_sent = texts[0]
    # input_topic = topics[0]
    for i in range(100):
        input_sent = df['sentence'][i]
        input_topic = df['topic'][i]

        prompt = dedent(f"""
            ### Task:
            The task is stance detection in Hebrew. I will give you a sentence in Hebrew and a topic in Hebrew, and you will detect whether the stance is בעד, נגד, or נייטרלי.
            Notice that **stance** is NOT **sentiment**, and the stance is supposed to be calculated toward the topic and not generally.

            ### Example:
            - Input: עורכי הצהובונים חוגגים תאונת דרכים קטלנית
            - Topic: תאונות דרכים
            The stance here is: נייטרלי
            Since the topic is תאונות דרכים, but if the topic were עורכי הצהובונים, then the stance would be נגד.

            - Output: {{
                "sentence": עורכי הצהובונים חוגגים תאונת דרכים קטלנית,
                "topic": תאונות דרכים,
                "stance": נייטרלי
                "Explanation": מאחר והנושא הוא "תאונות דרכים", אבל אם הנושא היה "עורכי הצהובונים" העמדה הייתה "נגד".
            }}

            ### Input:
            Sentence: {input_sent}
            Topic: {input_topic}

            ### Output:
            {{
                "sentence": "{input_sent}",
                "topic": "{input_topic}",
                "stance": "בעד" | "נגד" | "נייטרלי"
                "Explanation": "Provide a brief explanation of the stance."
            }}
        """)

        response = get_completion(prompt)
        # write response to text file:
        with open('./Data/response.txt', 'a', encoding='utf-8') as f:
            f.write(f"Response: {response}\n\n")
        # print(response)

    # for i in range(0, len(texts), 1000):
    #     # Create requests
    #     texts_batch = texts[i:i + 1000]
    #     topics_batch = topics[i:i + 1000]
    #     requests = create_requests(texts_batch, topics_batch)

    #     json_name = f'./Data/requests/topic_stance_detection_{i}.jsonl'
    #     # write to jsonl file
    #     with open(json_name, 'w', encoding='utf-8') as file:
    #         for request in requests:
    #             json_str = json.dumps(request, ensure_ascii=False)
    #             file.write(json_str + '\n')

    #     requests = read_jsonl(json_name)
    #     # print(requests[0])

    # ---------------------------------------------------------- Batch creation ----------------------------------------------------------
    
    # i = 18000
    # json_name = f'./Data/requests/topic_stance_detection_{i}.jsonl'
    # # Upload your batch input file
    # batch_input_file = client.files.create(
    #     file=open(json_name, "rb"),
    #     purpose="batch"
    # )

    # print(f"Batch input file: {batch_input_file}")

    # # Create the batch
    # batch_input_file_id = batch_input_file.id
    # batch = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={"description": "nightly eval job"}
    # )

    # with open('./Data/requests/batches_ids.txt', 'w', encoding='utf-8') as file:
    #     file.write(batch.id)

    # ---------------------------------------------------------- Batch status check ----------------------------------------------------------

    # with open('./Data/requests/batches_ids.txt', 'r', encoding='utf-8') as file:
    #     batches_id = file.read().splitlines()  

    # for id in batches_id:
    #     print(f"Batch ID: {id}")
    #     # Wait for batch to complete
    #     batch_info = client.batches.retrieve(id)
    #     print(batch_info)

    # # batches = client.batches.list()
    # # for batch in batches:
    # #     print(batch.id, batch.status)

    # # Check if batch completed successfully
    # if batch_info.status == "completed":
    #     output_file_id = batch_info.output_file_id
    #     output_file_response = client.files.content(output_file_id)

    #     json_objects = [json.loads(obj) for obj in output_file_response.text.strip().split("\n")]

    #     parsed_data = []

    #     for i, obj in enumerate(json_objects):
    #         try:
    #             content = obj['response']['body']['choices'][0]['message']['content']

    #             # Attempt to load the content as JSON (some LLMs return it as a string)
    #             parsed_obj = json.loads(content)

    #             # Optionally validate the structure
    #             assert all(k in parsed_obj for k in ["sentence", "topic", "stance"])
    #             parsed_data.append(parsed_obj)

    #         except Exception as e:
    #             print(f"✗ Unexpected error parsing object {i}: {e}")
    #             print("Partial content:", content[:300])
    #             continue

    #     print(f"\n✅ Done. Successfully parsed {len(parsed_data)} sentences from {len(json_objects)} objects.")


    #     os.makedirs('./Data/output', exist_ok=True)
    #     # Base file path
    #     base_path = './Data/output/topic_stance_dataset_0.json'

    #     # Find an available filename
    #     output_path = base_path
    #     counter = 1
    #     while os.path.exists(output_path):
    #         output_path = base_path.replace('0.json', f'{counter}.json')
    #         counter += 1

    #     # Save the file
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(parsed_data, f, ensure_ascii=False, indent=4)

    #     print(f"✅ Results saved to {output_path}")

    # # Check for errors 
    # error_file_id = "file-LLsMjDt7j3aGSHuVpo2uEu"
    # error_response = client.files.content(error_file_id)
    # print(error_response.text)  # This will show why the requests failed


    # ---------------------------------------------------------- Batch cancel / list ----------------------------------------------------------

    # Cancel a batch
    # cancel = client.batches.cancel('batch_68261a1eca8c81909adfdf4eff2f3fe1')

    # print(cancel)

    # # Get a list of all batches
    # list_of_batches = client.batches.list(limit=10)
    # print(list_of_batches)

    


