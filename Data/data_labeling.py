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

# Main function to process the batch results
def process_batch_results(output_file_response):
    # Load the batch response data
    json_objects = [json.loads(obj) for obj in output_file_response.text.strip().split("\n")]
    
    # Create output directories
    os.makedirs('./Data/output', exist_ok=True)
    os.makedirs('./Data/output/debug', exist_ok=True)
    
    # Save the raw response for debugging
    with open("./Data/output/raw_response.jsonl", "w", encoding="utf-8") as f:
        for obj in json_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    # Parse the data
    parsed_data = []
    
    for i, obj in enumerate(json_objects):
        try:
            # Extract the content directly from the response structure
            if ('response' in obj and 
                'body' in obj['response'] and 
                'choices' in obj['response']['body'] and 
                len(obj['response']['body']['choices']) > 0 and
                'message' in obj['response']['body']['choices'][0] and
                'content' in obj['response']['body']['choices'][0]['message']):
                
                content = obj['response']['body']['choices'][0]['message']['content']
                
                # Save raw content for debugging
                if i % 100 == 0 or i == 165:  # Save specific ones plus some samples
                    with open(f"./Data/output/debug/raw_content_{i}.txt", "w", encoding="utf-8") as f:
                        f.write(content)
                
                # First try to find JSON in code blocks (```json ... ```)
                json_blocks = re.findall(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
                
                # If no code blocks found, try to extract a JSON array directly
                if not json_blocks:
                    array_match = re.search(r"\[\s*\{[\s\S]+?\}\s*\]", content)
                    if array_match:
                        json_blocks = [array_match.group(0)]
                
                # Process any found JSON blocks
                found_valid_entries = False
                
                for block in json_blocks:
                    try:
                        # Clean the block and parse
                        data = clean_and_parse_json(block)
                        
                        # Extract stance data from the parsed JSON
                        new_entries = extract_stance_data(data)
                        
                        if new_entries:
                            parsed_data.extend(new_entries)
                            found_valid_entries = True
                    
                    except Exception as e:
                        # Try once more with even more cleaning if the first attempt failed
                        try:
                            # Super aggressive cleaning - strip all non-ASCII quotes and normalize
                            super_clean = re.sub(r'[^\x00-\x7F]+[""]', '"', block)
                            data = json.loads(super_clean)
                            
                            # Extract stance data from the parsed JSON
                            new_entries = extract_stance_data(data)
                            
                            if new_entries:
                                parsed_data.extend(new_entries)
                                found_valid_entries = True
                        
                        except Exception as e2:
                            if i == 165:  # Special debugging for the problematic case
                                print(f"Failed to parse block in object {i}: {e2}")
                            continue
                
                # If no JSON blocks or parsing failed, try the entire content directly
                if not found_valid_entries:
                    try:
                        # Try parsing the content directly as JSON
                        data = clean_and_parse_json(content)
                        
                        # Extract stance data from the parsed JSON
                        new_entries = extract_stance_data(data)
                        
                        if new_entries:
                            parsed_data.extend(new_entries)
                            found_valid_entries = True
                    
                    except Exception:
                        # If we get here, we've tried everything normally - will handle in fallback
                        pass
                
                # If no valid entries found, log the issue
                if not found_valid_entries and (i == 165 or i % 50 == 0):  # Limit logging to avoid console spam
                    print(f"✗ No valid entries parsed in object {i}")
            
            else:
                if i % 50 == 0:  # Limit logging
                    print(f"⚠️ Invalid response structure in object {i}")
        
        except Exception as e:
            if i % 50 == 0:  # Limit logging
                print(f"✗ Unexpected error processing object {i}: {e}")
            continue
    
    # Special handling for object 165 if needed
    if any(i == 165 for i, _ in enumerate(json_objects)):
        try:
            # Direct hardcoded parsing for the known problematic entry
            with open(f"./Data/output/debug/raw_content_165.txt", "r", encoding="utf-8") as f:
                content = f.read()
            
            # Remove markdown code block markers
            clean_content = re.sub(r'```(?:json)?\s*|\s*```', '', content)
            
            # Handle directional markers and quotes
            clean_content = clean_content.replace('\u201c', '"').replace('\u201d', '"')
            clean_content = clean_content.replace('\u05f4', '"')
            clean_content = clean_content.replace('\u200f', '').replace('\u200e', '')
            
            data = json.loads(clean_content)
            
            # Extract stance data
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and "sentence" in entry and "topic" in entry and "stance" in entry:
                        parsed_data.append({
                            "sentence": entry["sentence"].strip(),
                            "topic": entry["topic"].strip(),
                            "stance": entry["stance"].strip()
                        })
                print("Successfully parsed object 165 with special handling!")
        
        except Exception as e:
            print(f"Special handling for object 165 failed: {e}")
    
    # Manual fallback strategy for any remaining debug files
    if not any(165 in data.values() for data in parsed_data):
        debug_files = glob.glob('./Data/output/debug/raw_content_*.txt')
        
        for debug_file in debug_files:
            try:
                obj_id = int(re.search(r'raw_content_(\d+)\.txt', debug_file).group(1))
                
                # Skip if we already have data from this object
                if any(obj_id in str(data.values()) for data in parsed_data):
                    continue
                
                with open(debug_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Remove markdown or code blocks completely
                clean_content = re.sub(r'```(?:json)?\s*|\s*```', '', content)
                
                try:
                    # Try parsing directly first
                    data = clean_and_parse_json(clean_content)
                    new_entries = extract_stance_data(data)
                    
                    if new_entries:
                        parsed_data.extend(new_entries)
                        print(f"Parsed object {obj_id} in fallback processing")
                
                except Exception:
                    # If that fails, try to extract anything that looks like a JSON array
                    array_match = re.search(r'\[\s*\{[\s\S]+?\}\s*\]', clean_content)
                    
                    if array_match:
                        try:
                            json_array = array_match.group(0)
                            data = clean_and_parse_json(json_array)
                            new_entries = extract_stance_data(data)
                            
                            if new_entries:
                                parsed_data.extend(new_entries)
                                print(f"Parsed object {obj_id} in secondary fallback")
                        
                        except Exception:
                            pass
            
            except Exception:
                # If we can't even parse the filename, skip it
                continue
    
    print(f"\n✅ Done. Successfully parsed {len(parsed_data)} sentences from {len(json_objects)} objects.")
    
    # Save the parsed data
    with open('./Data/output/labeled_data_art.json', 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=4)
    
    return parsed_data


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

if __name__ == "__main__":
    list_of_topics = {
            "הדתה","תחבורה ציבורית בשבת", "גיור", 
            "ימין פוליטי", "שמאל פוליטי", "מרכז פוליטי", 
            "שיקום עזה", "הסכמי שלום", "פינוי יישובים",
            "החזקת נשק", "שירות צבאי", "מילואים",
            "יוקר המחיה","אינפלציה", "מחירי דלק",
            "חופש ביטוי", "זכויות אדם",
            "חופש העיתונות", "תקשורת", 
            "כושר ותזונה", "קורונה", "חיסונים",
            "תרבות צעירים", "לימודי ליבה", "חינוך מיני", "שביתות במערכת החינוך",
            "חניה", "שיפור תשתיות", "תחבורה ציבורית",
            "איכות הסביבה", "הגנת הסביבה",
            "הגירה יהודית", "הגירה בלתי חוקית",
            "השקעה בפריפריה", "פיתוח הנגב והגליל", "הכרה בישובים",
            "הטרדה מינית", "אלימות",
            "חדשנות וטכנולוגיה", "יזמות", "הייטק",
            "ספורט תחרותי", "אולימפיאדה"
    }

    # data = load_data("biunlp/HeSum")
    
    # disp = data['article'][0]
    # # print(disp)

    # full_text_prompt = dedent(f"""
    #         ### Task:
    #         Analyze each Hebrew sentence and determine whether it expresses a stance (support, opposition, or neutrality) toward any of the topics listed below.

    #         ### Stance Types:
    #         - תומך (supportive): expresses clear support or praise.
    #         - מתנגד (opposing): expresses criticism or disapproval.
    #         - נייטרלי (neutral): not clearly supportive or opposing, or topic only implied.

    #         ### Topics:
    #         Choose **only one** of the following topics for each sentence (even if not a perfect match, choose the closest one):
    #         {list_of_topics}

    #         ### Instructions:
    #         1. Split each input into individual sentences.
    #         2. For each sentence:
    #         - Determine if it expresses any stance (explicit or implied).
    #         - If so, assign the closest matching topic and appropriate stance.
    #         - If it doesn't express a stance, classify as נייטרלי.
    #         3. Output a JSON list of relevant sentences with the following format:

    #         ### Output Format:
    #         [
    #         {{
    #             "sentence": "המשפט בעברית",
    #             "topic": "הנושא המתאים מהרשימה",
    #             "stance": "תומך" | "מתנגד" | "נייטרלי"
    #         }},
    #         ...
    #         ]

    #         ### Example:
    #         [
    #         {{
    #             "sentence": "הממשלה עושה כל שביכולתה כדי להילחם בטרור.",
    #             "topic": "פוליטיקה - ימין",
    #             "stance": "תומך"
    #         }},
    #         {{
    #             "sentence": "אין מקום לסרבנות במילואים.",
    #             "topic": "מילואים",
    #             "stance": "תומך"
    #         }}
    #         ]

    #         ### Input:
    #         {disp}
    #         """)
    
    # # prompt = f"""
    # #         ### Task:
    # #         Given a list of Hebrew texts, identify **sentences that express a stance** — that is, an opinion, argument, or controversy — **toward one of the defined topics below**.
            
    # #         **If a sentence expresses a stance but not clearly toward any listed topic, choose the topic that is semantically closest and mark the stance as נייטרלי (neutral).**

    # #         ### Stance Labels:
    # #         - תומך (supportive): expresses support, praise, or positive opinion toward a topic.
    # #         - מתנגד (opposing): expresses disagreement, criticism, or negative opinion toward a topic.
    # #         - נייטרלי (neutral): no clear stance toward the topic, or topic is inferred by proximity.

    # #          ### Topics:
    # #         Use only one of the following predefined topics:
    # #         {list_of_topics}

    # #         ### Instructions:
    # #         1. Split the input into sentences (use '.', '|', or similar delimiters as needed).
    # #         2. Identify any sentence that expresses a stance (explicit or implied) toward one of the listed topics.
    # #         3. For each such sentence:
    # #             - Select the **most relevant topic** from the list.
    # #             - Classify the **stance**: תומך, מתנגד, or נייטרלי.
    # #             - The stance may be **explicit** (e.g., "אני בעד") or **implied** (e.g., praise, criticism, sarcasm).
    # #             - If a sentence contains multiple clauses with differing opinions, choose the **clause that relates to one of the listed topics** and use that for classification.
    # #             - Do not annotate the same sentence more than once.

    # #         ### Output Format (JSON):
    # #         {{
    # #             "sentences": [
    # #                 {{
    # #                     "sentence": "המשפט בעברית",
    # #                     "topic": "הנושא הרלוונטי מהרשימה",
    # #                     "stance": "תומך" | "מתנגד" | "נייטרלי"
    # #                 }},
    # #                 ...
    # #             ]
    # #         }}

    # #         ### Example Output:
    # #         {{
    # #             "sentences": [
    # #                 {{
    # #                     "sentence": "המדיניות החדשה של הממשלה מעולה בעיניי",
    # #                     "topic": "ימין פוליטי",
    # #                     "stance": "תומך"
    # #                 }},
    # #                 {{
    # #                     "sentence": "העיתונות עשתה עבודה חשובה כשחשפה את ההתנהלות במינוי הרמטכ\"ל",
    # #                     "topic": "תקשורת",
    # #                     "stance": "תומך"
    # #                 }}
    # #             ]
    # #         }}

    # #         ### Input Texts:
    # #         {disp}
    # #     """

    # response = get_completion(full_text_prompt)
    # print(response)


    # data = load_data("biunlp/HeSum")
    # # suma = data["summary"]
    # art = data["article"][:500]
    # requests = []

    # for i in range(len(art)):
    #     # chunk = suma[i:i+50]
    #     # if not any(s.strip() for s in chunk):
    #     #     continue

    #     prompt = dedent(f"""
    #         ### Task:
    #         Analyze each Hebrew sentence and determine whether it expresses a stance (support, opposition, or neutrality) toward any of the topics listed below.

    #         ### Stance Types:
    #         - תומך (supportive): expresses clear support or praise.
    #         - מתנגד (opposing): expresses criticism or disapproval.
    #         - נייטרלי (neutral): not clearly supportive or opposing, or topic only implied.

    #         ### Topics:
    #         Choose **only one** of the following topics for each sentence (even if not a perfect match, choose the closest one):
    #         {list_of_topics}

    #         ### Instructions:
    #         1. Split each input into individual sentences.
    #         2. For each sentence:
    #         - Determine if it expresses any stance (explicit or implied).
    #         - If so, assign the closest matching topic and appropriate stance.
    #         - If it doesn't express a stance, classify as נייטרלי.
    #         3. Output a JSON list of relevant sentences with the following format:

    #         ### Output Format:
    #         [
    #         {{
    #             "sentence": "המשפט בעברית",
    #             "topic": "הנושא המתאים מהרשימה",
    #             "stance": "תומך" | "מתנגד" | "נייטרלי"
    #         }},
    #         ...
    #         ]

    #         ### Example:
    #         [
    #         {{
    #             "sentence": "הממשלה עושה כל שביכולתה כדי להילחם בטרור.",
    #             "topic": "פוליטיקה - ימין",
    #             "stance": "תומך"
    #         }},
    #         {{
    #             "sentence": "אין מקום לסרבנות במילואים.",
    #             "topic": "מילואים",
    #             "stance": "תומך"
    #         }}
    #         ]

    #         ### Input:
    #         {art[i]}
    #         """)
        

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
    # with open('./Data/requests/art_data_labeling_requests.jsonl', 'w', encoding='utf-8') as file:
    #     for request in requests:
    #         json_str = json.dumps(request, ensure_ascii=False)
    #         file.write(json_str + '\n')

    # requests = read_jsonl('./Data/requests/art_data_labeling_requests.jsonl')
    # print(requests[0])

    # ---------------------------------------------------------- Batch creation ----------------------------------------------------------
    # # Upload your batch input file
    # batch_input_file = client.files.create(
    #     file=open("./Data/requests/art_data_labeling_requests.jsonl", "rb"),
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
    batch_info = client.batches.retrieve('batch_68207ef3d3b0819092f4fa96a606c017')
    # print(batch_info)

    # Check if batch completed successfully
    if batch_info.status == "completed":
        output_file_id = batch_info.output_file_id
        output_file_response = client.files.content(output_file_id)

        data = process_batch_results(output_file_response)

        # json_objects = [json.loads(obj) for obj in output_file_response.text.strip().split("\n")]
        # os.makedirs('./Data/output', exist_ok=True)
        # with open("./Data/output/labeled_data_art.jsonl", "w", encoding="utf-8") as f:
        #     for obj in json_objects:
        #         f.write(json.dumps(obj, ensure_ascii=False) + "\n")


        # parsed_data = []

        # for i, obj in enumerate(json_objects):
        #     try:
        #         content = obj['response']['body']['choices'][0]['message']['content']

        #         # Extract all JSON blocks inside triple-backticks
        #         json_blocks = re.findall(r"```json\s*([\s\S]+?)\s*```", content)

        #         if not json_blocks:
        #             print(f"⚠️ No JSON block in object {i}, skipping.")
        #             continue

        #         found_any = False
        #         for block in json_blocks:
        #             try:
        #                 data = json.loads(block)

        #                 # Expecting: {"sentences": [ {...}, {...}, ... ]}
        #                 if isinstance(data, dict) and "sentences" in data:
        #                     for entry in data["sentences"]:
        #                         sentence = entry.get("sentence")
        #                         topic = entry.get("topic")
        #                         stance = entry.get("stance")
        #                         if sentence and topic and stance:
        #                             parsed_data.append({
        #                                 "sentence": sentence.strip(),
        #                                 "topic": topic.strip(),
        #                                 "stance": stance.strip()
        #                             })
        #                             found_any = True
        #                 else:
        #                     print(f"✗ Unexpected JSON structure in object {i}, skipping block.")

        #             except json.JSONDecodeError as e:
        #                 print(f"✗ JSON decode error in object {i}: {e}")
        #                 print("Block content:", block[:300])
        #                 continue

        #         if not found_any:
        #             print(f"✗ No valid sentence entries parsed in object {i}")

        #     except Exception as e:
        #         print(f"✗ Unexpected error parsing object {i}: {e}")
        #         print("Partial content:", content[:300])
        #         continue

        # print(f"\n✅ Done. Successfully parsed {len(parsed_data)} sentences from {len(json_objects)} objects.")

        # os.makedirs('./Data/output', exist_ok=True)
        # with open('./Data/output/labeled_data_art.json', 'w', encoding='utf-8') as f:
        #     json.dump(parsed_data, f, ensure_ascii=False, indent=4)

    # # Check for errors 
    # error_file_id = "file-LLsMjDt7j3aGSHuVpo2uEu"
    # error_response = client.files.content(error_file_id)
    # print(error_response.text)  # This will show why the requests failed


    # ---------------------------------------------------------- Batch cancel / list ----------------------------------------------------------

    # # Cancel a batch
    # cancel = client.batches.cancel('batch_6814be0f90348190b6565d7fcd1f1450')
    # print(cancel)

    # # Get a list of all batches
    # list_of_batches = client.batches.list(limit=10)
    # print(list_of_batches)

    


