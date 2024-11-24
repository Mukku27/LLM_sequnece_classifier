# # Install necessary libraries
# !pip install -q transformers accelerate trl bitsandbytes datasets evaluate
# !pip install -q peft scikit-learn

# # Login to HuggingFace hub
# !huggingface-cli login --token $YOUR_HF_TOKEN
from transformers import TrainingArguments # import TrainingArguments
from torch.nn import functional as F # import functional as F
import numpy as np # import numpy

# Load dataset from Excel file
import pandas as pd
df = pd.read_excel("code_labels.xlsx") # Load your excel file

# Check label distribution (optional, but recommended)
df.label.value_counts(normalize=True)

# Sample data and split into train, test, and validation sets (adjust sample sizes as needed)
train_df = df.sample(frac=0.7, random_state=42) # 70% for training
remaining_df = df.drop(train_df.index)
val_df = remaining_df.sample(frac=0.5, random_state=42) # 15% for validation
test_df = remaining_df.drop(val_df.index) # 15% for testing


# Convert to Hugging Face Dataset objects
from datasets import DatasetDict, Dataset
dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val,
    'test': dataset_test
})

# Calculate class weights
import torch
class_weights=(1/train_df.label.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()

# Download and prepare the model for training
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "crumb/nano-mistral"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=4,
    device_map='auto'
)

# Create a LoRA config for training
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

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

# Download the tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# Prepare test data for predictions
sentences = test_df.text.tolist()
batch_size = 32
all_outputs = []

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])

final_outputs = torch.cat(all_outputs, dim=0)
test_df['predictions'] = final_outputs.argmax(axis=1).cpu().numpy()

# Evaluate the model's performance
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, classification_report

def get_metrics_result(test_df):
    y_test = test_df.label
    y_pred = test_df.predictions

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

get_metrics_result(test_df)

# Data preprocessing function for training data
def data_preprocessing(row):
    return tokenizer(row['text'], truncation=True, max_length=512)

tokenized_data = dataset.map(data_preprocessing, batched=True, remove_columns=['text'])
tokenized_data.set_format("torch")

# Data collator for batch processing during training
from transformers import DataCollatorWithPadding

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

# Define custom metric function for training evaluation
def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)

    return {
        'balanced_accuracy': balanced_accuracy_score(predictions, labels),
        'accuracy': accuracy_score(predictions, labels)
    }

# Custom Trainer class definition for training with class weights
from transformers import Trainer
training_args = TrainingArguments(
    output_dir='sentiment_classification',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=1,
    weight_decay=0.01,
    eval_strategy='epoch',  # Updated from evaluation_strategy
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to="none"
)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Define training arguments
# training_args = TrainingArguments(
#     output_dir='sentiment_classification',
#     learning_rate=1e-4,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=1,
#     logging_steps=1,
#     weight_decay=0.01,
#     evaluation_strategy='epoch',
#     save_strategy='epoch',
#     load_best_model_at_end=True,
#     report_to="none"
# )




#Create trainer object and start training process
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['val'],
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
)



train_result = trainer.train()

# Generate predictions after training and evaluate performance metrics again
def generate_predictions(model, df_test):
    sentences = df_test.text.tolist()

    all_outputs = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs.to('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            outputs=model(**inputs)
            all_outputs.append(outputs['logits'])

    final_outputs=torch.cat(all_outputs,dim=0)

generate_predictions(model,test_df)

def get_performance_metrics(test_df):
    y_test = test_df.label
    y_pred = test_df.predictions

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
get_performance_metrics(test_df)