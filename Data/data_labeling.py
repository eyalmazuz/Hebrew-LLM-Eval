import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import json
import time
import pandas as pd
import re
from src.data_loader import load_data

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
        temperature=0, 
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

if __name__ == "__main__":
    # data = load_data("biunlp/HeSum")
    # requests = []

    # for i in range(0, len(data), 500):
    #     prompt = f"""
    #         Given a list of texts in Hebrew, identify the sentences that express a stance (i.e., where an opinion, argument, or controversy is expressed).
    #         Classify each such sentence as either תומך (supportive), מתנגד (opposing), or ניטרלי (neutral) — and **ONLY** these categories.

    #         # Definitions:
    #         - A sentence is considered תומך (supportive) if it expresses a positive opinion or agreement with a topic.
    #         - A sentence is considered מתנגד (opposing) if it expresses a negative opinion or disagreement with a topic.
    #         - A sentence is considered ניטרלי (neutral) if it does not express a clear opinion or stance on any topic.

    #         ### Instructions:
    #         1. Identify all sentences in the text that express a stance.
    #         2. For each such sentence, infer the **topic** it refers to (translate the topic to Hebrew).
    #         3. Classify the stance as תומך, מתנגד, or ניטרלי based on the definitions above.
    #         4. Return a JSON object where each entry includes:
    #         - the original sentence,
    #         - the inferred topic (in Hebrew),
    #         - the stance label (in Hebrew).

    #         ### Input Texts:
    #         {data['summary'][i:i+500]}

    #         ### Output Format (JSON):
    #         {{
    #             "sentences": [
    #                 {{
    #                     "sentence": "המדיניות החדשה של הממשלה מעולה בעיניי",
    #                     "topic": "ממשלה",
    #                     "stance": "תומך"
    #                 }},
    #                 ...
    #             ]
    #         }}
    #     """
        

    #     # print(prompt)
    #     request = {
    #         "custom_id": f"request-{i}",
    #         "method": "POST",
    #         "url": "/v1/chat/completions",
    #         "body": {
    #             "model": "gpt-4o-mini",
    #             "messages": [
    #                 {"role": "system", "content": "You are an advanced NLP model specializing in stance detection."},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             "max_tokens": 1000
    #         }
    #     }

    #     requests.append(request)

    # # write to jsonl file
    # with open('./Data/requests/data_labeling_requests.jsonl', 'w', encoding='utf-8') as file:
    #     for request in requests:
    #         json_str = json.dumps(request, ensure_ascii=False)
    #         file.write(json_str + '\n')

    # requests = read_jsonl('./Data/requests/data_labeling_requests.jsonl')
    # print(requests[0])

    # ---------------------------------------------------------- Batch creation ----------------------------------------------------------
    # # Upload your batch input file
    # batch_input_file = client.files.create(
    #     file=open("./Data/requests/data_labeling_requests.jsonl", "rb"),
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

    # batch_id = batch.id
    # print(f"Batch created with ID: {batch_id}")

    # ---------------------------------------------------------- Batch status check ----------------------------------------------------------

    # Wait for batch to complete
    batch_info = client.batches.retrieve('batch_6814c4247a58819086abc9e656028ddc')
    print(batch_info)

    # Check if batch completed successfully
    if batch_info.status == "completed":
        output_file_id = batch_info.output_file_id
        output_file_response = client.files.content(output_file_id)

        json_objects = [json.loads(obj) for obj in output_file_response.text.strip().split("\n")]
        parsed_data = []

        sentence_pattern = re.compile(r'{\s*"sentence"\s*:\s*"([^"]+)",\s*"topic"\s*:\s*"([^"]+)",\s*"stance"\s*:\s*"([^"]+)"\s*}')

        for i, obj in enumerate(json_objects):
            try:
                content = obj['response']['body']['choices'][0]['message']['content']

                # Extract all JSON blocks inside triple-backticks
                json_blocks = re.findall(r"```json\s*({[\s\S]+?})\s*```", content)

                if not json_blocks:
                    print(f"⚠️ No JSON block in object {i}, skipping.")
                    continue

                found_any = False
                for block in json_blocks:
                    matches = sentence_pattern.findall(block)
                    for match in matches:
                        sentence, topic, stance = match
                        parsed_data.append({
                            "sentence": sentence.strip(),
                            "topic": topic.strip(),
                            "stance": stance.strip()
                        })
                        found_any = True

                if not found_any:
                    print(f"✗ No valid sentence entries parsed in object {i}")

            except Exception as e:
                print(f"✗ Unexpected error parsing object {i}: {e}")
                print("Partial content:", content[:300])
                continue

        print(f"\n✅ Done. Successfully parsed {len(parsed_data)} sentences from {len(json_objects)} objects.")

        os.makedirs('./Data/output', exist_ok=True)
        with open('./Data/output/labeled_data.json', 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)

    # # Check for errors 
    # error_file_id = "file-LLsMjDt7j3aGSHuVpo2uEu"
    # error_response = client.files.content(error_file_id)
    # print(error_response.text)  # This will show why the requests failed


    # ---------------------------------------------------------- Batch cancel / list ----------------------------------------------------------

    # # Cancel a batch
    # cancel = client.batches.cancel('batch_6814be0f90348190b6565d7fcd1f1450')
    # print(cancel)

    # Get a list of all batches
    # list_of_batches = client.batches.list(limit=10)
    # print(list_of_batches)

    


