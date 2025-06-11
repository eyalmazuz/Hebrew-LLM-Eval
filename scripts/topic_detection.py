from src.data_loader import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
import json
from src.utils import split_into_sentences

def get_topic_for_model(hebrew_sentence, model, tokenizer):
    """Get topic for a Hebrew sentence using the model."""
    dynamic_examples = [
        # דוגמאות מאוזנות - לא כללי מידי, לא ספציפי מידי
        {
            "sentence": "ישראל לא תאפשר שיקום עזה ללא פתרון לשבויים",
            "topic": "שיקום עזה"  # לא "ישראל" (כללי מידי), לא "מדיניות שיקום עזה" (ספציפי מידי)
        },
        {
            "sentence": "פרמטר נוסף שמשפיע על המחיר הוא סוג החניה",
            "topic": "מחירי חניה"  # לא "חניה" (כללי מידי), לא "השפעת סוג חניה על מחיר" (ספציפי מידי)
        },
        {
            "sentence": "האורחים צילמו את האירוע והעלו תמונות לרשתות החברתיות",
            "topic": "רשתות חברתיות"  # לא "צילום לרשתות חברתיות" (ספציפי מידי)
        },
        {
            "sentence": "קבוצות של צעירים ממזרח ירושלים חמושים בקרשים הגיעו למקום",
            "topic": "צעירים מזרח ירושלים"  # לא "צעירים" (כללי מידי), לא "צעירים חמושים בקרשים" (ספציפי מידי)
        },
        {
            "sentence": "חמאס והג'יהאד קפצו על האירוע בנבי מוסא",
            "topic": "נבי מוסא"  # המקום הוא הנושא המרכזי
        },
        {
            "sentence": "משרד ההקדשים לא הציב שמירה באתר",
            "topic": "שמירה באתר"  # לא "שמירה" (כללי מידי), לא "משרד ההקדשים" (זה הסובייקט)
        },
        {
            "sentence": "האתר הפך למוקד הסתה נגד ישראל ויהודים",
            "topic": "הסתה"  # לא "הסתה נגד ישראל ויהודים" (ספציפי מידי)
        },
        {
            "sentence": "בינימין נתניהו פוגע במדינת ישראל",
            "topic": "בנימין נתניהו"  # כשמדובר באישיות - זה הנושא
        },
        {
            "sentence": "עיתון ידיעות אחרונות נלחם על הזכות לפרסם מידע",
            "topic": "חופש עיתונות"  # לא "ידיעות אחרונות" (ספציפי מידי), לא "עיתונות" (כללי מידי)
        },
        {
            "sentence": "הילד חלה אחרי החיסון",
            "topic": "חיסונים"  # נושא רפואי כללי
        }
    ]

    base_prompt = """
        המשימה שלך היא לזהות את **הנושא המרכזי** של המשפט ברמת פירוט מתאימה.

        כללי איזון חשובים:
        1. **לא כללי מידי** - אל תבחר במושגים רחבים כשיש נושא ספציפי יותר
        2. **לא ספציפי מידי** - אל תבחר בפרטים קטנים, בחר בנושא העיקרי
        3. **רמת הפירוט הנכונה** - כמו כותרת חדשות טובה

        דוגמאות לרמת פירוט נכונה:
        - "מחירי חניה" (לא "חניה" ולא "השפעת סוג חניה על מחיר")
        - "רשתות חברתיות" (לא "צילום לרשתות חברתיות")
        - "צעירים מזרח ירושלים" (לא "צעירים" ולא "צעירים חמושים")
        - "שיקום עזה" (לא "ישראל" ולא "מדיניות שיקום עזה")

        החזר רק את הנושא - ללא הסברים.
        """

    # בונים את חלק הדוגמאות לפרומפט
    examples_prompt = "\n".join([
        f'משפט: {ex["sentence"]}\nנושא: {ex["topic"]}'
        for ex in dynamic_examples
    ])

    # הפרומפט המלא עם הדוגמה החדשה לבדיקה
    final_prompt = base_prompt + "\n\nדוגמאות:\n" + examples_prompt + f"\n\nמשפט: {hebrew_sentence}\nנושא:"


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
            temperature=0.1
        )
    
    # Extract only the new tokens (excluding the prompt)
    topic_tokens = outputs[0][prompt_length:]
    
    # Decode only the topic tokens
    topic = tokenizer.decode(topic_tokens, skip_special_tokens=True).strip()
    topic = topic.split("\n")[0] 
    
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