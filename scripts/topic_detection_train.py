from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import json
import torch

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

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


dataset = load_dataset("biunlp/HeSum")['train']

# model to fine-tune
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  
) 

topic_model = AutoModelForCausalLM.from_pretrained('dicta-il/dictalm2.0', device_map='cuda', quantization_config=quant_config)
topic_tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictalm2.0')

# Fix tokenizer configuration
if topic_tokenizer.pad_token is None:
    topic_tokenizer.pad_token = topic_tokenizer.eos_token

def split_into_sentences(text):
    """Split text into sentences."""
    if not isinstance(text, str):
        return []
    separators = r"[■|•.\n]"
    sentences = [sent.strip() for sent in re.split(separators, text) if sent.strip()]
    return sentences

if __name__ == "__main__":
    """
    Fine tuning the topic model to improve our platform.
    First stage - making the data:
        Taking the dataset and split the texts into sentences.
        Use LLM (gpt) to detect the topics in each sentence.
        Review the data manually to ensure quality - about 5%.
    Second stage - fine tuning the model:
        Using the data from the first stage to fine tune the model.
    """
    # print example from dataset
    for i in range(3):
        print(f"Example {i+1}:")
        print("Text:", dataset[i]['text'])
        print("Summary:", dataset[i]['summary'])
        print("\n" + "="*50 + "\n")

    # Process dataset to split texts into sentences
    processed_data = []
    for item in dataset:
        sentences = split_into_sentences(item['text'])
        processed_data.append({
            'sentences': sentences,
        })

    # Use LLM to detect topics
    prompt = """
    
    """



