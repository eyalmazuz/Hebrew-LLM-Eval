import re
import torch
import pandas as pd
from tqdm import tqdm

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
import pandas as pd
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import os

# ----------------------------------------------------------------------------
# OpenAI client for labeling
# ----------------------------------------------------------------------------
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

def split_into_sentences(text):
    """Split text into sentences."""
    if not isinstance(text, str):
        return []
    separators = r"[■|•.\n]"
    sentences = [sent.strip() for sent in re.split(separators, text) if sent.strip()]
    return sentences


# def get_topic_for_model(hebrew_sentence, model, tokenizer):
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
        בהינתן משפט,
        עליך לקרוא את אותו, לנתח אותו בקצרה, ולאחר מכן להחזיר את הנושא המרכזי שבו המשפט עוסק.

        הגדרות:
        הנושא הוא התחום המרכזי של המשפט (למשל: פוליטיקה, רפואה, חינוך, ספורט, כלכלה, ביטחון, טכנולוגיה ועוד).
        אל תיתן יותר מנושא אחד.
        הנושא צריך להיות מילה אחת או ביטוי קצר (עד 3 מילים).
        אין צורך בניסוחים כמו "הנושא הוא" - כתוב רק את הנושא.
        אם לא ניתן לזהות נושא - כתוב: לא ברור.
        כאשר קיימת ישות פועלת (למשל: "ישראל הודיעה כי..."), זהה את התחום שבו עוסקת ההצהרה, ולא בהכרח את שם הגוף הפועל.
        יש מקרים בהם תצטרך להיות כללי יותר או ספציפי יותר, בהתאם למשמעות המשפט.
        אם המשפט עוסק בכמה נושאים, בחר את הנושא המרכזי ביותר.
        אם הנושא שמצאת הוא ברבים - הפוך אותו ליחיד.

        שלבי עבודה:
        1. נתח את משמעות המשפט.
        2. זהה על איזה תחום עוסק המשפט.
        3. החזר את הנושא.
        """
    
    examples_prompt = "\n".join([
        f'משפט: {ex["משפט"]}\nניתוח: {ex["ניתוח"]}\nנושא: {ex["נושא"]}'
        for ex in dynamic_examples
    ])

    # הפרומפט המלא עם הדוגמה החדשה לבדיקה
    final_prompt = (
        base_prompt  
        + "\n\nדוגמאות:\n" + examples_prompt
        + f"\n\nמשפט: {hebrew_sentence}\nניתוח:\nנושא:"
    )

    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    torch.cuda.empty_cache()
    outputs = model.generate(**inputs, max_new_tokens=32)

    print(tokenizer.decode(outputs[0]))
    
    return tokenizer.decode(outputs[0])
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


def create_prompt_for_finetuning(example):
    """
    Formats the sentence and topic into a structured prompt 
    for the Causal Language Model to learn the task.
    """
    
    # 1. Base Prompt (Instructions and Definitions)
    # Using a simplified version for fine-tuning, focusing on the core instruction.
    # The full, detailed instructions/examples can be omitted here 
    # to save token space if the model is robust enough to learn the pattern.
    # For a minimal, effective format:
    
    prompt = (
        f"משפט: {example['sentence']}\n"
        f"נושא: {example['topic']}"
    )
    
    return {"text": prompt}


if __name__ == "__main__":
    # ----------------------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------------------
    # dataset = load_dataset("biunlp/HeSum")['train']
    # dataset = load_dataset("HebArabNlpProject/HebNLI")['train'].select(range(5000))
    # # or using the dataset inside Data/dataset/wikipedia.jsonl (only the first 5000 samples)
    # dataset = load_dataset("json", data_files="./Data/datasets/Wikipedia.jsonl")['train'].select(range(5000))
    
    # print example from dataset
    # for i in range(3):
    #     print(f"Example {i+1}:")
    #     print("Text:", dataset[i]['article'])
    #     print("Summary:", dataset[i]['summary'])
    #     print("\n" + "="*50 + "\n")

        # print(f"Example {i+1}:")
        # print("Text:", dataset[i])
        # print("\n" + "="*50 + "\n")


    # Process dataset to split texts into sentences
    # processed_data = []
    # for item in dataset:
    #     sentences = split_into_sentences(item['translation2'])
    #     processed_data.append({
    #         'sentences': sentences,
    #     })

    # print(processed_data[0]['sentences'][0])

    # ----------------------------------------------------------------------------
    # Load model to fine-tune
    # ----------------------------------------------------------------------------
    quant_config_4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model_id = 'dicta-il/dictalm2.0'
    topic_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map='auto', 
        quantization_config=quant_config_4
    )
    topic_tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Fix tokenizer configuration
    if topic_tokenizer.pad_token is None:
        topic_tokenizer.pad_token = topic_tokenizer.eos_token
    # For CLMs, setting padding to the right is common for efficiency
    topic_tokenizer.padding_side = "right"
        

    # quant_config_8 = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,           # handles outlier values
    #     llm_int8_has_fp16_weight=False    # keep weights in int8
    # )

    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "google/gemma-2-9b",
    #     device_map="auto",
    #     quantization_config=quant_config_8,
    # )

    # context = dataset[0]['translation2']
    # hebrew_sentence = dataset['translation2'][0]

    # topic = get_topic_for_model(hebrew_sentence, model, tokenizer)
    
    # input_text = "Write me a poem about Machine Learning."
    # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    # outputs = model.generate(**input_ids, max_new_tokens=32)
    # print(tokenizer.decode(outputs[0]))
    

    # ----------------------------------------------------------------------------
    # Use LLM to detect topics
    # ----------------------------------------------------------------------------
#     labeled_data = []

#     TOPIC_DETECTION_PROMPT = """
# # Instructions:
# Given a sentence **in Hebrew**, you need to identify the *main* topic of the sentence.

# # Definitions:
# - "Topic" is the main area that the sentence deals with.
# - The topic should be a short phrase (try keep it up to 3 words).
# - Do not write sentences or explanations — only the topic.
# - If the topic cannot be understood — write: Not clear.
# - When there are several topics, choose the most central one.

# # Inupt:
# Sentence: {sentence}

# # Output:
# Topic:
# """

# # use tqdm for progress bar
#     for item in tqdm(dataset, desc="Labeling topics"):
#         prompt = TOPIC_DETECTION_PROMPT.format(
#             sentence=item["translation2"]
#         )
#         topic = get_completion(prompt, model="gpt-4o-mini")

#         labeled_data.append({
#             "sentence": item["translation2"],
#             "topic": topic.strip()
#         })
#     # print("משפט:", hebrew_sentence)
#     # print("נושא:", topic.strip())

#     # Save labeled data to csv file
#     output_path = "./scripts/labeled/labeled_topic_data.csv"
#     df = pd.DataFrame(labeled_data)
#     df.to_csv(output_path, index=False, encoding='utf-8-sig')
#     print(f"Labeled data saved to {output_path}")

    # ----------------------------------------------------------------------------
    # Fine tuning the model
    # ----------------------------------------------------------------------------
    df = pd.read_csv("./scripts/labeled/labeled_topic_data.csv")
    # remove rows with topic as "Not clear"
    df = df[df['topic'] != "Not clear."]

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Apply the formatting function to the dataset
    dataset = dataset.map(create_prompt_for_finetuning)

    # Split the dataset into train and test sets
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # ----------------------------------------------------------------------------
    # QLoRA Setup
    # ----------------------------------------------------------------------------
    # 1. Prepare model for k-bit training (important for stability)
    topic_model = prepare_model_for_kbit_training(topic_model)

    # 2. LoRA Configuration
    lora_config = LoraConfig(
        r=16, # LoRA attention dimension. Higher is generally better, but costs more memory/time.
        lora_alpha=32, # Scaling factor for the weights. 
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj"
            # Add other important layers for better performance (e.g., 'gate_proj', 'up_proj', 'down_proj')
        ],
        bias="none", # Only 'none' or 'all' supported for now
        lora_dropout=0.05,
        task_type="CAUSAL_LM", # We are fine-tuning a Causal Language Model
    )

    # ----------------------------------------------------------------------------
    # Training Arguments
    # ----------------------------------------------------------------------------
    output_dir = "./models/results_topic_detection"

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        save_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        report_to="none",
        dataset_text_field="text",  # This stays in SFTConfig
        packing=True,  # This stays in SFTConfig
    )

    # ----------------------------------------------------------------------------
    # SFTTrainer Setup and Start Training
    # ----------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=topic_model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=topic_tokenizer,
        peft_config=lora_config,
    )

    print("Starting fine-tuning...")
    # Train the model
    trainer.train()

    # ----------------------------------------------------------------------------
    # Save the Fine-Tuned Model (LoRA Adapters)
    # ----------------------------------------------------------------------------
    # Save the adapter weights (not the entire 4-bit model)
    adapter_save_path = os.path.join(output_dir, "final_adapters")
    trainer.model.save_pretrained(adapter_save_path)
    topic_tokenizer.save_pretrained(adapter_save_path)
    print(f"Fine-tuned LoRA adapters saved to {adapter_save_path}")
    
