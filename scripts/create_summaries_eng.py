import argparse
import os
import csv
import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import load_dataset
import google.generativeai as genai
import time

# Load API key from environment
_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_with_gpt(article):
    prompt = f"Summarize the following text briefly (in no more than 80 words):\n\n{article}\n\nSummary:"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You summarize texts in English briefly and clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()


def summarize_with_gemma(article, pipe):
    prompt = f"Summarize the following text briefly (in no more than 80 words):\n\n{article}\n\nSummary:"

    output = pipe(prompt, max_new_tokens=500, do_sample=False)
    
    # just return the generated text after the prompt
    generated_text = output[0]["generated_text"]
    summary = generated_text[len(prompt):].strip()
    return summary

def summarize_with_gemini(article, model_name="gemini-2.5-flash"):
    prompt = f"Summarize the following text briefly (in no more than 80 words):\n\n{article}\n\nSummary:"
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        err_str = str(e)
        if "429" in err_str and "retry" in err_str:
            # חיפוש זמן ההמתנה מתוך הודעת השגיאה
            import re
            match = re.search(r"retry in (\d+)", err_str)
            if match:
                delay = int(match.group(1))
                print(f"Quota hit. Retrying in {delay} seconds...")
                time.sleep(delay + 1)
                return summarize_with_gemini(article, model_name)
        print(f"Error in Gemini summarization: {e}")
        return f"שגיאה ביצירת סיכום: {str(e)}"
    
def process_articles(articles, output_file, summarizer, model_name=None):
    # אם הקובץ קיים, נטען אותו
    if os.path.isfile(output_file):
        existing_df = pd.read_csv(output_file, encoding="utf-8-sig")
        done_articles = set(existing_df["Article"].tolist())
    else:
        existing_df = pd.DataFrame(columns=["Article", "Summary", "Stance", "Error"])
        done_articles = set()

    # נעבור על המאמרים
    for i, article in enumerate(articles, start=1):
        if article in done_articles:
            print(f"[{i}] כבר טופל, מדלג...")
            continue

        print(f"[{i}] מסכם את המאמר...")

        try:
            if model_name:
                summary = summarizer(article, model_name=model_name)
            else:
                summary = summarizer(article)

            row = {"Article": article, "Summary": summary, "Stance": "נייטרלי", "Error": ""}
        except Exception as e:
            print(f"שגיאה במאמר {i}: {e}")
            row = {"Article": article, "Summary": "", "Stance": "", "Error": str(e)}

            # נשמור מיידית ונצא כדי לא לאבד התקדמות
            existing_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
            existing_df.to_csv(output_file, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
            print(f"✅ נשמר לקובץ {output_file}. אפשר להריץ שוב להמשך.")
            return

        # נוסיף שורה חדשה
        existing_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)

        # נשמור כל מאמר שעובד
        existing_df.to_csv(output_file, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)

    print(f"\n🎉 כל {len(articles)} המאמרים עובדו ונשמרו ל־{output_file}")

def summarize_with_gpt_oss(article, pipe):
    prompt = f"""
    Summarize the following text.
    The summay nust be accurate, with no more than 80 words.
    Do not include analysis, explanations, or introductory text. Just provide the final summary only. 
    
    Text:
    {article}
    
    Summary:
"""   

    try:
        # GPT-OSS expects chat format
        messages = [
            {"role": "user", "content": prompt}
        ]

        # outputs = pipe(
        #     messages,
        #     max_new_tokens=500,
        #     do_sample=False,
        # )
        outputs = pipe(
            messages,
            max_new_tokens=500,
        )
        
        generated_text = outputs[0]["generated_text"][-1]
        return generated_text
        
        
    except Exception as e:
        print(f"Error in GPT-OSS generation: {e}")
        return f"שגיאה ביצירת סיכום: {str(e)}"

def load_gpt_oss_model():
    """Load GPT-OSS model with different strategies"""
    try:
        model_id = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",  # allows automatic layer offload
            offload_folder="offload",  # creates folder for CPU/NVME offload
        )

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        print(f"Strategy failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create summary using LLMs")
    # parser.add_argument("--article_path", type=str, required=True, help="Path to article text file.")
    # parser.add_argument("--stance", type=str, help="Stance of the text.")
    parser.add_argument("--output-file", default="./scripts/output/eng_summaries_oss.csv", help="Where to save the summary.")
    parser.add_argument("--use-quantization", action="store_true", help="Use 4-bit quantization for memory efficiency")
    args = parser.parse_args()

    
    # gemma_pipe = pipeline(
    #     "text-generation",
    #     model="google/gemma-2-2b-it",
    #     device="cuda",
    #     torch_dtype=torch.bfloat16
    # )

    print("Loading GPT-OSS model...")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load the model
    gpt_oss_pipe = load_gpt_oss_model()
    
    if gpt_oss_pipe is None:
        print("Could not load GPT-OSS model. Exiting.")
        exit(1)
    

    # Load data
    name = 'EdinburghNLP/xsum'
    dataset = load_dataset(name)
    articles = dataset['train']['document'][:50]

    # Process articles
    processed_count = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # for i, article in enumerate(articles):
    #     try:
    #         # Get summary
    #         # summary = summarize_with_gpt(article)
    #         summary = summarize_with_gemma(article, gemma_pipe)

    #         print("\n=== SUMMARY ===")
    #         print(summary)

    #         # Write to CSV
    #         file_exists = os.path.isfile(args.output_file)
    #         with open(args.output_file, "a", encoding="utf-8-sig", newline='') as f:
    #             writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    #             if not file_exists:
    #                 writer.writerow(["Article", "Summary", "Stance"])
    #             writer.writerow([article, summary, "Neutral"])
                
    #         processed_count += 1
                
    #     except Exception as e:
    #         print(f"Error processing article {i+1}: {e}")
    #         continue

    # print(f"\nProcessing complete!")
    # print(f"Results saved to: {args.output_file}")

    for i, article in enumerate(articles):
        try:
            # Get summary
            summary = summarize_with_gpt_oss(article, gpt_oss_pipe)
            print("\n=== SUMMARY ===")
            print(summary)

            # Write to CSV
            file_exists = os.path.isfile(args.output_file)
            with open(args.output_file, "a", encoding="utf-8-sig", newline='') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
                if not file_exists:
                    writer.writerow(["Article", "Summary", "Stance"])
                writer.writerow([article, summary, "נייטרלי"])
                
            processed_count += 1
                
        except Exception as e:
            print(f"Error processing article {i+1}: {e}")
            continue

        # Clear cache periodically to prevent memory buildup
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            print(f"Cleared GPU cache after {i+1} articles")

    print(f"\nProcessing complete!")
    print(f"Results saved to: {args.output_file}")
    
    # Final memory cleanup
    torch.cuda.empty_cache()

    # process_articles(
    #     articles,
    #     output_file=args.output_file,
    #     summarizer=summarize_with_gemini,
    #     model_name="gemini-2.5-flash"
    # )
    

    # for article in articles:
    #     # Get summary
    #     summary = summarize_with_gpt(article)
    #     # summary = summarize_with_gptoss(article, oss_pipe)
    #     # summary = summarize_with_gemma(article, gemma_pipe)


    #     # print("=== ARTICLE ===")
    #     # print(article)
    #     print("\n=== SUMMARY ===")
    #     print(summary)

    #     # Ensure output directory exists
    #     os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    #     # Write to CSV
    #     file_exists = os.path.isfile(args.output_file)
    #     with open(args.output_file, "a", encoding="utf-8-sig", newline='') as f:
    #         writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    #         if not file_exists:
    #             writer.writerow(["Article", "Summary", "Stance"])
    #         writer.writerow([article, summary, "נייטרלי"])

