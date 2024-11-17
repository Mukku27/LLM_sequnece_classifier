
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import numpy as np

# Load dataset
dataset = load_dataset("ag_news")
df = pd.DataFrame(dataset['train'])

# Split data into train, validation, and test sets using scikit-learn
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Convert pandas DataFrames to Hugging Face Datasets
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'val': Dataset.from_pandas(val_df),
    'test': Dataset.from_pandas(test_df)
})

# Define class weights
class_weights = (1 / train_df['label'].value_counts(normalize=True).sort_index()).tolist()
class_weights = torch.tensor(class_weights)
class_weights = class_weights / class_weights.sum()

# Load pre-trained model and tokenizer
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=4,
    device_map='auto'  # Automatically choose device (CPU or GPU)
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")

# Define data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='sentiment_classification',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=10,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to="none"
)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions)
    }

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Generate predictions
def generate_predictions(model, df_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sentences = df_test['text'].tolist()
    batch_size = 32
    all_outputs = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs.logits)

    final_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
    df_test['predictions'] = np.argmax(final_outputs, axis=1)

generate_predictions(trainer.model, test_df)

# Evaluate the model
print(classification_report(test_df['label'], test_df['predictions']))
print(f"Accuracy: {accuracy_score(test_df['label'], test_df['predictions'])}")
print(f"Balanced Accuracy: {balanced_accuracy_score(test_df['label'], test_df['predictions'])}")
