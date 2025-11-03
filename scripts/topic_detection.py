from src.data_loader import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
import json
from src.utils import split_into_sentences

def get_topic_for_model(context, hebrew_sentence, model, tokenizer):
    """Get topic for a Hebrew sentence using the model."""
    dynamic_examples = [
        {
            "משפט": "עולה מניתוח של 7,500 מחקרים שפורסמו בנושא בין השנים 1973 עד 2011 כי ב־40 השנים האחרונות יש מגמת ירידה מתמשכת בפוריות הגבר בעולם המערבי ונרשמה ירידה של יותר מ־50 אחוזים בריכוז ובספירת הזרע",
            "ניתוח": "מדובר על פוריות הגבר בעולם המערבי, וירידה בריכוז ובספירת הזרע.",
            "נושא": "פוריות הגבר"
        },
        {
            "משפט": "עם עומסי החום של הקיץ והגידול בשימוש במזגנים, עולות גם התקלות, ומדריך זה סוקר את התקלות הנפוצות, עלויות התיקון והמלצות לבחירת מזגן",
            "ניתוח": "מדובר על מזגנים, תקלות נפוצות ועלויות תיקון.",
            "נושא": "תחזוקת מזגנים "
        },
        {
            "משפט": "משרד הבריאות פרסם את נתוני התחלואה בקורונה: אחוז החיוביים ירד; בישראל יש 8,310 חולים פעילים; סך המחלימים עומד על 325,862; מספר הנפטרים מפרוץ המגפה עומד על 2,735",
            "ניתוח": "מדובר על נתוני תחלואה בקורונה בישראל",
            "נושא": "קורונה"
        },
        {
            "משפט": "כחלק מהפתרונות היצירתיים הללו חוזרים אלינו לאחרונה משחקי הילדות של דור ההורים שלא ידע אינטרנט, טאבלט, אייפון ואקס בוקס, והסתפק במשחקי רחוב עם שאר הילדים",
            "ניתוח": "מדובר על משחקי הילדות של דור ההורים, משחקים ללא טכנולוגיה מודרנית.",
            "נושא": "משחקים"
        },
        {
            "משפט": "ישראל הצהירה כי לא תאפשר שיקום עזה ללא פתרון לסוגיית השבויים והנעדרים, אך נותר לראות אם תצליח לעמוד בהבטחתה",
            "ניתוח": "מדובר על הצהרת ישראל בנוגע לשיקום עזה ולסוגיית השבויים והנעדרים.",
            "נושא": "שיקום רצועת עזה"
        },
        {
            "משפט": "מאז שהמחיר ביניהם הושווה, רכבי הפנאי מזנבים במכירות המשפחתיים, כשיבואני המשפחתיים נאלצים להוריד מחירים או להעלות ברמת האבזור",
            "ניתוח": "מדובר על רכבי פנאי לעומת רכבים משפחתיים, והשפעת המחיר על המכירות.",
            "נושא": "רכבי פנאי"
        },
        {
            "משפט": "הפטרייה, קורדיספס שמה, משתלטת על מוחן של נמלים וכופה עליהן לטפס לגובה רב כדי לפזר את נבגיה",
            "ניתוח": "מדובר על פטרייה בשם קורדיספס שמשפיעה על נמלים.",
            "נושא": "קורדיספס"
        }
    ]

    base_prompt = """הוראות:
        בהינתן טקסט, ומשפט ממנו,
        עליך לקרוא את המשפט, לנתח אותו בקצרה, ולאחר מכן להחזיר את הנושא המרכזי שבו המשפט עוסק - השתמש בטקסט כקונטקסט.

        הגדרות:
        הנושא הוא התחום המרכזי של המשפט (למשל: פוליטיקה, רפואה, חינוך, ספורט, כלכלה, ביטחון, טכנולוגיה ועוד).
        אל תיתן יותר מנושא אחד.
        הנושא צריך להיות מילה אחת או ביטוי קצר (עד 3 מילים).
        אין צורך בניסוחים כמו "הנושא הוא" - כתוב רק את הנושא.
        אם לא ניתן לזהות נושא - כתוב: לא ברור.
        כאשר קיימת ישות פועלת (למשל: "ישראל הודיעה כי..."), זהה את התחום שבו עוסקת ההצהרה, ולא את שם הגוף הפועל.
        יש מקרים בהם תצטרך להיות כללי יותר או ספציפי יותר, בהתאם למשמעות המשפט.
        אם המשפט עוסק בכמה נושאים, בחר את הנושא המרכזי ביותר.
        אם הנושא שמצאת הוא ברבים - הפוך אותו ליחיד.

        שלבי עבודה:
        1. נתח את משמעות המשפט.
        2. זהה על איזה תחום עוסק המשפט.
        3. החזר את הנושא.
        """

    # בונים את חלק הדוגמאות לפרומפט
    examples_prompt = "\n".join([
        f'משפט: {ex["משפט"]}\nניתוח: {ex["ניתוח"]}\nנושא: {ex["נושא"]}'
        for ex in dynamic_examples
    ])

    # הפרומפט המלא עם הדוגמה החדשה לבדיקה
    final_prompt = base_prompt + "\n\nקונטקסט:" + context + "\n\nדוגמאות:\n" + examples_prompt + f"\n\nמשפט: {hebrew_sentence}\nניתוח:\nנושא:"


    # Encode the prompt
    inputs = tokenizer(final_prompt.strip(), return_tensors='pt', padding=True).to(model.device)
    
    # Get the length of the prompt in tokens
    prompt_length = inputs.input_ids.shape[1]
    
    # Generate only the new tokens (the topic)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            max_new_tokens=10,
            # temperature=0.1
        )
    
    # Extract only the new tokens (excluding the prompt)
    topic_tokens = outputs[0][prompt_length:]
    
    # Decode only the topic tokens
    topic = tokenizer.decode(topic_tokens, skip_special_tokens=True).strip()
    topic = topic.split("\n")[0]
    # topic = topic.split("\n")[0].replace("נושא:", "").strip().rstrip(".").rstrip(":")
    
    return {
        "sentence": hebrew_sentence,
        "topic": topic
    }

def get_topic(hebrew_sentence, model, tokenizer):
    # split into sentences
    sentences = split_into_sentences(hebrew_sentence)

    res = []

    for sentence in sentences:
        prompt = f"""
            המשימה שלך היא לזהות את נושא המשפט, הנושא לא תמיד יהיה כתוב במשפט.
            תחזיר **רק** את הנושא בלי מילים נוספות או הסברים.

            לדוגמה:
            משפט: אני לא מתכוון להצביע בבחירות הקרובות
            נושא: בחירות

            משפט: הילד שלי חלה אחרי החיסון, אני לא מתכוונת לחסן שוב
            נושא: חיסונים

            משפט: עיתון "ידיעות אחרונות" נלחם על הזכות שלו לפרסם את המידע הזה
            נושא: חופש העיתונות

            משפט: בינימין נתניהו פוגע במדינת ישראל
            נושא: בנימין נתניהו
            
            משפט: {sentence}
            נושא:
            """

        # Encode the prompt
        inputs = tokenizer(prompt.strip(), return_tensors='pt', padding=True).to(model.device)
        
        # Get the length of the prompt in tokens
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate only the new tokens (the topic)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=False,
                max_new_tokens=10
            )
        
        # Extract only the new tokens (excluding the prompt)
        topic_tokens = outputs[0][prompt_length:]
        
        # Decode only the topic tokens
        topic = tokenizer.decode(topic_tokens, skip_special_tokens=True).strip()
        topic = topic.split("\n")[0] 
        res.append({
            "sentence": sentence,
            "topic": topic
        })
    
    return res


# if __name__ == "__main__":
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if your hardware doesn't support bfloat16
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",  # or "fp4"
#     )    
#     model = AutoModelForCausalLM.from_pretrained('dicta-il/dictalm2.0', device_map='cuda', quantization_config=quant_config)
#     tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictalm2.0')
    
#     # Fix tokenizer configuration
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     data = load_data("biunlp/HeSum")
    
#     # Create a new file or clear the existing file before appending
#     with open("scripts/output/topics.jsonl", "w", encoding="utf-8") as f:
#         pass
    
#     # Process each sentence in the dataset
#     total_sentences = len(data['summary'])
#     print(f"Starting to process {total_sentences} sentences...")
    
#     for idx, sent in enumerate(data['summary']):
#         try:
#             # Get the topic for the current sentence
#             res = get_topic(sent, model, tokenizer)
            
#             # Print progress every 10 sentences
#             if idx % 10 == 0:
#                 print(f"--------------------------- Processed {idx}/{total_sentences} sentences ---------------------------")
            
#             for r in res:
#                 # Save into a JSONL file
#                 with open("scripts/output/topics.jsonl", "a", encoding="utf-8") as f:
#                     f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
#         except Exception as e:
#             # Log errors without stopping the entire process
#             print(f"Error processing sentence {idx}: {e}")
#             with open("scripts/output/errors.log", "a", encoding="utf-8") as f:
#                 f.write(f"Error at index {idx}: {str(e)}\nSentence: {sent}\n\n")