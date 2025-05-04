import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import json
import time
import pandas as pd
import re

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
    # prompt = "Write a one-sentence bedtime story about a unicorn."
    # response = get_completion(prompt)
    # print(response)

    # ---------------------------------------------------------- JSONL creation ----------------------------------------------------------
    # # data = read_jsonl('./Data/summarization-7-heb.jsonl')
    # # data = ["חניה בתל אביב", "שיקום עזה", "ספרות", "הדתה", "רדיפה דתית", "תרבות צעירים", "ביטחון", "כושר ותזונה", "דימוי גוף", "קורונה", "מירי רגב", "בנימין נתניהו", 
    # #         "מניפולציות תקשורתיות", "תעמולה", "חופש העיתונות", "תקשורת", "מלחמה בטרור", "פוליטיקה", "חמאס", "הדרת נשים", "התנחלויות", "הפרות סדר של פעילי ימין", 
    # #         "הפרות סדר של פעילי שמאל", "חופש ביטוי", "שחיתות", "בחירות", "הפגנות", "הסכסוך הישראלי-פלסטיני", "חרם על ישראל", "הסברה", "הסברה ישראלית", "הטרדה מינית",
    # #         "פמיניזם", "הסכמי שלום", "גיוס חרדים", "יוקר המחיה", "משבר הדיור", "רפורמה משפטית", "סוגיית השבויים", "פינוי יישובים", "מחירי מזון", "עצמאות אנרגטית", "זכויות להט״ב", 
    # #         "מעמד הר הבית", "עובדים זרים", "מבקשי מקלט", "פער חברתי", "הפרדה מגדרית", "תחבורה ציבורית בשבת", "התערבות בג״ץ", "עצמאות השב״כ", "הפרטת שירותים ציבוריים", 
    # #         "שכר מינימום", "שירות לאומי", "לימודי ליבה", "ייצוג נשים", "איכות הסביבה", "מתווה הגז", "פדיון שבויים", "שביתות במערכת החינוך", "זכויות פלסטינים",
    # #         "אלימות במגזר הערבי", "עונש מוות למחבלים", "גבולות ישראל", "קצבאות נכים", "לימודי יהדות", "שוויון בנטל",
    # #         "אינפלציה", "מדיניות חוץ", "הפללת צריכת קנאביס", "הפרדת דת ומדינה", "מחירי דלק", "שמירת שבת במרחב הציבורי", "פיקוח על מחירים", "חינוך ממלכתי-דתי", 
    # #         "כשרות במסעדות", "השכלה גבוהה", "מעמד העברית", "גיור", "נישואים אזרחיים", "חופש דת", "ריבונות בשטחי יהודה ושומרון", "צמצום פערים", "אבטלה", "מעמד ירושלים", 
    # #         "חובת חיסונים", "צמצום בירוקרטיה", "פרטיות ברשת", "צנזורה צבאית", "שביתת רופאים", "תעשיות הייטק", "חינוך חינם מגיל אפס", "מכסות יבוא", "ממשלת אחדות", 
    # #         "מס עשירים", "זכות השיבה", "זכויות בעלי חיים", "חקלאות ישראלית", "זיהום אוויר", "מחסור במים", "אלימות משטרתית", "נשק בלתי חוקי", "ביטחון אישי", 
    # #         "חוק הלאום", "חינוך לדמוקרטיה", "שכר בכירים במגזר הציבורי", "מונופולים", "שוק ההון", "מחאה אזרחית", "ייצוג עדתי", "תוכניות ריאליטי", 
    # #         "תיירות פנים", "זכויות חשודים", "עבודה מהבית", "שימוש בכוח בעימותים", "העסקה קבלנית", "זכויות עובדי קבלן", "אלימות נגד נשים", "סייבר וביטחון לאומי", 
    # #         "חופש אקדמי", "חינוך מיני", "השקעה בפריפריה", "סובסידיות לחקלאים", "ניהול משבר המים", "ביטוח בריאות ממלכתי", "ועדות קבלה ביישובים", "הגירה יהודית", 
    # #         "הבטחת הכנסה", "התחממות גלובלית", "רפואה פרטית", "עמלות בנקאיות", "שווקים פיננסיים", "צמצום תאונות דרכים", "אשראי צרכני", "צהרונים מסובסדים", 
    # #         "פיקוח על מסגרות חינוך", "שעות עבודה", "מדיניות דיור", "כוח אדם בחינוך", "כבישי אגרה", "יוזמות שלום אזוריות", "מערכת המשפט", "שיפור תשתיות", 
    # #         "פיתוח הנגב והגליל", "קיצוץ בתקציב הביטחון", "נגישות לנכים", "הכרה ביישובים בדואים", "תקציב המדינה", "משבר האקלים", "הגנת הסביבה", "זכויות אדם", 
    # #         "אנרגיה מתחדשת", "חינוך טכנולוגי", "חופש עיסוק", "ביטוח אבטלה", "בתי ספר דמוקרטיים", "רפורמה בחינוך", "שירותי רווחה", "תרבות ישראלית", "הדיור הציבורי", 
    # #         "פנסיה תקציבית", "רפורמה במערכת הבריאות", "רב תרבותיות", "מיסוי מקרקעין", "הסעת המונים", "מדיניות הגירה", "דיור בר השגה", "שוויון בחינוך", "פערי שכר מגדריים", 
    # #         "אפליה עדתית", "מלחמת המפרץ", "הסדרים עם לבנון", "אבטחת מידע", "קליטת עלייה", "זכויות ילדים", "שיטות הצבעה", "אלימות בבתי ספר", "פיקוח על בנקים", 
    # #         "ביטוח לאומי", "תיווך דירות", "מדינה דו-לאומית", "עיצומים על איראן", "משבר הפליטים", "עיר ועיירה", "הרשות הפלסטינית", "זכויות נפגעי עבירה", "הכרה בישובים", 
    # #         "מדיניות אכיפה", "שחיקת מעמד המורים", "רשתות חברתיות", "זכויות דיגיטליות", "תרבות הסטארט-אפ", "מדיניות מיסוי", "ענף הבנייה", "עצמאות התקשורת",
    # #         "זכויות קשישים", "מונופול תקשורת", "הישגיות בספורט"]
    # data = ["הישגיות בספורט", "ספורט תחרותי", "ספורט קבוצתי", "ספורט אישי", "אולימפיאדה", "ספורט מקצועי", "ספורט חובבני", "אימון ספורט", "תזונה בספורט", "בריאות בספורט",
    #         "חדשנות וטכנולוגיה", "אבטחת מידע", "תרבות הסטארט-אפ", "יזמות", "הייטק", "תעשייה מסורתית", "תעשיות עתירות ידע", "תעשיות כבדות", "תעשיות קלות", "תעשיות טכנולוגיות",
    #         "אלימות וחוק", "פיקוח על בנקים", "אלימות נגד נשים", "זכויות חשודים", "אלימות משטרתית", "אלימות במגזר הערבי", "הטרדה מינית", "אלימות בבתי ספר", "אלימות במשפחה", "אלימות פוליטית",
    #         "פיתוח אזורי", "הכרה בישובים", "עיר ועיירה",  "הכרה ביישובים בדואים", "פיתוח הנגב והגליל", "השקעה בפריפריה", "תכנון עירוני", "תיירות פנים", "תיירות חוץ",
    #         "מדיניות חוץ והגירה", "משבר הפליטים", "קליטת עלייה", "מדיניות הגירה", "הגירה יהודית", "מדיניות חוץ", "חרם על ישראל", "הגירה בלתי חוקית", "חוק השבות", 
    #         "תחבורה ותשתיות", "חניה בתל אביב", "חניות", "תשתיות תחבורה", "תחבורה ציבורית", "תחבורה פרטית", "תחבורה חכמה", "כבישים", "רכבות", "אוטובוסים", "צמצום תאונות דרכים", "כבישי אגרה"]
    # requests = []
    # contents = []
    # # content =""
    # # for i in range(15):
    # #     content += f"{i%5}. " + data[i]['text_raw']
    # #     content += "\n"
    # #     if (i+1) % 5 == 0:
    # #         contents.append(content)
    # #         content = ""

    # for i in range(len(data)):
    #     prompt = f"""Given the following topic: __{data[i]}__, generate **30** sentences as follows:
    #         1. **10** sentences that support the topic.
    #         2. **10** sentences that oppose the topic.
    #         3. **10** sentences that are neutral towards the topic.
    #         Each sentence should be **one** of the following:
    #         - **Supportive**: Expresses a positive opinion or agreement with the topic.
    #         - **Opposing**: Expresses a negative opinion or disagreement with the topic.
    #         - **Neutral**: Does not express a clear opinion or stance towards the topic.
    #         The sentences should be in **HEBREW** and should be **realistic** and **natural**.
            
    #         ### **Instructions:**
    #         <INST>
    #            1. Make sure the sentences do not follow the same pattern or structure.
    #            2. Ensure that the sentences are **realistic** and **natural**.
    #            3. Avoid using **repetitive** phrases or structures.
    #         </INST>
            
    #         ### **Output Format:**
    #             ```json
    #             {{
    #                 "topic": "{data[i]}",
    #                 "supportive_sentences": [
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     ...
    #                 ],
    #                 "opposing_sentences": [
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     ...
    #                 ],
    #                 "neutral_sentences": [
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     ...
    #                 ],
    #                 "all_sentences": [
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     "<SENTENCE>",
    #                     ...
    #                 ]
    #             }}
    #             ```
    #         """
        

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

    # # for i, content in enumerate(contents):
    # #     prompt = f"""
    # #     Given five texts, identify the topics within them that contain a stance (i.e., where an opinion, argument, or controversy is expressed).
    # #     A topic contains a stance if there are differing perspectives, policy implications, or strong opinions present.
    # #     If similar or identical topics appear in multiple texts, **merge them into one unified topic**.
    # #     The topic shold be ONE to THREE words tops, and in Hebrew.
    # #     For example, if the texts discuss "Politics" you will write "פוליטיקה".

    # #     ### **Instructions:**
    # #     <INST>
    # #         1. Extract topics that contain a stance from the given texts.
    # #         2. Merge similar topics into a single entry to avoid redundancy.
    # #         3. Avoid general subjects that do not involve an opinion or controversy.
    # #         4. Summarize each topic concisely in a way that highlights the stance-related aspect.
    # #         5. Return a **JSON object** with a **list of stance-related topics**.
    # #     </INST>`

    # #     ### **Input Texts:**
    # #     {content}
        
    # #     ### **Output Format (JSON):**
    # #     ```json
    # #     {{
    # #         "topics": [
    # #             "<TOPIC>",
    # #             "<TOPIC>",
    # #             "<TOPIC>",
    # #             ...
    # #         ]
    # #     }}
    # #     """
    # #     # print(prompt)
    # #     request = {
    # #         "custom_id": f"request-{i}",
    # #         "method": "POST",
    # #         "url": "/v1/chat/completions",
    # #         "body": {
    # #             "model": "gpt-4o-mini",
    # #             "messages": [
    # #                 {"role": "system", "content": "You are an advanced NLP model specializing in stance detection. Your mission is to extract topics from texts to be later used as targets for stance detection."},
    # #                 {"role": "user", "content": prompt}
    # #             ],
    # #             "max_tokens": 1000
    # #         }
    # #     }

    # #     requests.append(request)

    # # write to jsonl file
    # with open('./Data/extra_data_creation.jsonl', 'w', encoding='utf-8') as file:
    #     for request in requests:
    #         json_str = json.dumps(request, ensure_ascii=False)
    #         file.write(json_str + '\n')

    # requests = read_jsonl('./Data/extra_data_creation.jsonl')
    # print(requests)

    # ---------------------------------------------------------- Batch creation ----------------------------------------------------------
    # # Upload your batch input file
    # batch_input_file = client.files.create(
    #     file=open("./Data/extra_data_creation.jsonl", "rb"),
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
    batch_info = client.batches.retrieve('batch_6808b70412808190ae98444979f4d9bf')
    print(batch_info)

    # Check if batch completed successfully
    if batch_info.status == "completed":
        output_file_id = batch_info.output_file_id
        output_file_response = client.files.content(output_file_id)
        # print(f"Batch output:\n{output_file_response.text}")

        json_objects = [json.loads(obj) for obj in output_file_response.text.strip().split("\n")]
        data = []

        for i, obj in enumerate(json_objects):
            try:
                content = obj['response']['body']['choices'][0]['message']['content']

                # Extract the JSON block
                match = re.search(r"```json\s*(\{.*)", content, re.DOTALL)
                if not match:
                    print(f"⚠️ Could not find JSON block in object {i}, skipping.")
                    continue

                raw_json = match.group(1).strip()
                raw_json = raw_json.removesuffix("```").strip()

                topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', raw_json)
                support_match = re.search(r'"supportive_sentences"\s*:\s*(\[[^\]]*?\])', raw_json, re.DOTALL)
                oppose_match = re.search(r'"opposing_sentences"\s*:\s*(\[[^\]]*?\])', raw_json, re.DOTALL)
                neutral_match = re.search(r'"neutral_sentences"\s*:\s*(\[[^\]]*?\])', raw_json, re.DOTALL)

                if not (topic_match and support_match and oppose_match and neutral_match):
                    print(f"⚠️ Missing fields in object {i}, skipping.")
                    continue

                topic = topic_match.group(1)
                supportive_sentences = extract_string_list(support_match.group(1))
                opposing_sentences = extract_string_list(oppose_match.group(1))
                neutral_sentences = extract_string_list(neutral_match.group(1))

                # print(f"Topic: {topic}")
                # print("Supportive Sentences:", supportive_sentences)
                # print("Opposing Sentences:", opposing_sentences)
                # print("Neutral Sentences:", neutral_sentences)
                # print("\n")

                data.append({
                    "topic": topic,
                    "supportive_sentences": supportive_sentences,
                    "opposing_sentences": opposing_sentences,
                    "neutral_sentences": neutral_sentences
                })

                # print(f"✓ Parsed topic {i}: {topic}")

            except Exception as e:
                print(f"✗ Failed to parse object {i}: {e}")
                print("Partial content:", content[:300])
                continue

        print(f"\nDone. Successfully parsed {len(data)} out of {len(json_objects)}")

        # Save the parsed data to a JSON file
        with open('./Data/batch_output.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # # Check for errors 
    # error_file_id = "file-LLsMjDt7j3aGSHuVpo2uEu"
    # error_response = client.files.content(error_file_id)
    # print(error_response.text)  # This will show why the requests failed


    # ---------------------------------------------------------- Batch cancel / list ----------------------------------------------------------

    # Cancel a batch
    # client.batches.cancel('batch_67ee395e02c0819091c5125d17dc47c1')

    # Get a list of all batches
    # client.batches.list(limit=10)

    


