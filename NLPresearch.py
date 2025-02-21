import time
import torch
import datasets
import evaluate
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import pandas as pd

# Device selection: use MPS (Apple Silicon) if available, else CUDA, else CPU
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define models and datasets
models = {
    "MHA": "t5-large",                   # Vanilla T5 (not fine-tuned for summarization)
    "MQA": "google/t5-large-lm-adapt"      # Fine-tuned for summarization
}

datasets_info = {
    "cnn_dailymail": {
        "name": "cnn_dailymail",
        "config": "3.0.0",
        "text_column": "article",
        "summary_column": "highlights"
    },
    "arxiv": {
        "name": "ccdv/arxiv-summarization",
        "config": None,
        "text_column": "abstract",
        "summary_column": "abstract"
    },
    "pubmed": {
        "name": "scientific_papers",
        "config": "pubmed",
        "text_column": "article",
        "summary_column": "abstract"
    }
}

# Load evaluation metric
rouge = evaluate.load("rouge")

def evaluate_model(model_name, dataset_name):
    """Evaluates a model on a dataset using minimal streamed data with a summarization prompt."""
    print(f"⚡ Evaluating {model_name} Model on {dataset_name}...")
    
    try:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f" Error loading model {model_name}: {e}")
        return None, None

    dataset_info = datasets_info[dataset_name]
    dataset_config = dataset_info["config"]

    # Load dataset in streaming mode to avoid downloading the entire dataset
    try:
        ds_stream = load_dataset(
            dataset_info["name"],
            dataset_config,
            split="test",
            streaming=True
        )
    except Exception as e:
        print(f" Error loading dataset {dataset_name}: {e}")
        return None, None

    # Retrieve only one sample from the streamed dataset
    ds_iter = iter(ds_stream)
    sample = next(ds_iter)
    # Wrap sample in a list for processing
    dataset = [sample]

    text_column = dataset_info["text_column"]
    summary_column = dataset_info["summary_column"]

    # Add the summarization prompt (this is crucial for t5-large)
    text = sample[text_column]
    prompt_text = "summarize: " + text
    texts = [prompt_text]
    references = [sample[summary_column]]

    start_time = time.time()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64  # Reduced max length for input tokens
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=32,   # Increased a bit for more complete output
            do_sample=False,
            num_beams=1      # Greedy decoding for speed
        )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    inference_time = round(time.time() - start_time, 3)

    scores = rouge.compute(predictions=generated_texts, references=references)
    avg_rouge1 = scores["rouge1"]

    return inference_time, avg_rouge1

# Run evaluation and store results
results = []
for dataset_name in datasets_info.keys():
    for model_label, model_path in models.items():
        infer_time, score = evaluate_model(model_path, dataset_name)
        if infer_time is not None and score is not None:
            results.append({
                "Model": model_label,
                "Dataset": dataset_name,
                "Inference Time (s)": infer_time,
                "ROUGE Score": score
            })

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("\n Results saved to evaluation_results.csv")
print(df)
