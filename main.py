

from datasets import load_dataset

dataset = load_dataset("ag_news")

import pandas as pd

df = pd.DataFrame(dataset['train'])

df.label.value_counts(normalize=True)

# Splitting the dataframe into 4 separate dataframes based on the labels
label_1_df = df[df['label'] == 0]
label_2_df = df[df['label'] == 1]
label_3_df = df[df['label'] == 2]
label_4_df = df[df['label'] == 3]

# Shuffle each label dataframe
label_1_df = label_1_df.sample(frac=1).reset_index(drop=True)
label_2_df = label_2_df.sample(frac=1).reset_index(drop=True)
label_3_df = label_3_df.sample(frac=1).reset_index(drop=True)
label_4_df = label_4_df.sample(frac=1).reset_index(drop=True)

# Splitting each label dataframe into train, test, and validation sets
label_1_train = label_1_df.iloc[:2000]
label_1_test = label_1_df.iloc[2000:2500]
label_1_val = label_1_df.iloc[2500:3000]

label_2_train = label_2_df.iloc[:2000]
label_2_test = label_2_df.iloc[2000:2500]
label_2_val = label_2_df.iloc[2500:3000]

label_3_train = label_3_df.iloc[:2000]
label_3_test = label_3_df.iloc[2000:2500]
label_3_val = label_3_df.iloc[2500:3000]

label_4_train = label_4_df.iloc[:2000]
label_4_test = label_4_df.iloc[2000:2500]
label_4_val = label_4_df.iloc[2500:3000]

# Concatenating the splits back together
train_df = pd.concat([label_1_train, label_2_train, label_3_train, label_4_train])
test_df = pd.concat([label_1_test, label_2_test, label_3_test, label_4_test])
val_df = pd.concat([label_1_val, label_2_val, label_3_val, label_4_val])

# Shuffle the dataframes to ensure randomness
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)

train_df.label.value_counts()

from datasets import DatasetDict, Dataset

# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)

# Combine them into a single DatasetDict
dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val,
    'test': dataset_test
})
dataset

import torch

class_weights=(1/train_df.label.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()
class_weights

from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True, 
    bnb_4bit_compute_dtype = torch.bfloat16 
)

model_name = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=4,
    device_map='auto'
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

lora_config = LoraConfig(
    r = 16, 
    lora_alpha = 8,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, 
    bias = 'none',
    task_type = 'SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

from transformers import AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

sentences = test_df.text.tolist()

batch_size = 32  

all_outputs = []

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]

    inputs = tokenizer(batch_sentences, return_tensors="pt", 
    padding=True, truncation=True, max_length=512)

    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])
        
final_outputs = torch.cat(all_outputs, dim=0)
test_df['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, classification_report

def get_metrics_result(test_df):
    y_test = test_df.label
    y_pred = test_df.predictions

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

get_metrics_result(test_df)

def data_preprocesing(row):
    return tokenizer(row['text'], truncation=True, max_length=512)

tokenized_data = dataset.map(data_preprocesing, batched=True, 
remove_columns=['text'])
tokenized_data.set_format("torch")

from transformers import DataCollatorWithPadding

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),
    'accuracy':accuracy_score(predictions,labels)}

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, 
            dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir = 'sentiment_classification',
    learning_rate = 1e-4,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 1,
    logging_steps=1,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    report_to="none"
)

trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
)

train_result = trainer.train()

def generate_predictions(model,df_test):
    sentences = df_test.text.tolist()
    batch_size = 32  
    all_outputs = []

    for i in range(0, len(sentences), batch_size):

        batch_sentences = sentences[i:i + batch_size]

        inputs = tokenizer(batch_sentences, return_tensors="pt", 
        padding=True, truncation=True, max_length=512)

        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') 
        for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
        
    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

generate_predictions(model,test_df)
get_performance_metrics(test_df)