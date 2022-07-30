import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn

# You can change this if you want hugginface to automatically log to wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

model_name = "microsoft/deberta-v3-base"

path = "train"

def get_essay(essay_id):
    essay_path = os.path.join("/home/kohtaro/work/essay_feedback/data/"+path+"/", f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

df = pd.read_csv("/home/kohtaro/work/essay_feedback/data/train.csv")
df['essay_text'] = df['essay_id'].apply(get_essay)

# This function helps strip extra whitespaces from the text, 
# convert it all to lower case for uniformity, and remove end of line characters
def normalise(text):
    text = text.lower()
    text = text.strip()
    text = re.sub("\n", " ", text)
    return text

df['discourse_text'] = df['discourse_text'].apply(normalise)
df['discourse_type'] = df['discourse_type'].apply(normalise)
df['essay_text'] = df['essay_text'].apply(normalise)
df['discourse_effectiveness'] = df['discourse_effectiveness'].apply(normalise)

#print(df.head())

# Get tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 512

df['text'] = df['discourse_text']+tokenizer.sep_token+df['essay_text']

classes_to_labels = {
    "adequate":0,
    "effective":1,
    "ineffective":2,
}
df['labels'] = df['discourse_effectiveness'].replace(classes_to_labels)

# Stratify by labels makes sure that the distribution of 
# the classes in both train and test remains the same
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["labels"])

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.shuffle(seed=42).map(tokenize_function, batched=True)
tokenized_test_dataset = valid_dataset.shuffle(seed=42).map(tokenize_function, batched=True)

#print(tokenized_train_dataset)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    warmup_ratio=0.1, 
    lr_scheduler_type='cosine',
    # Optimising
    auto_find_batch_size=True,
    # The num of workers may vary for different machines, if you are not sure, just comment this line out
    dataloader_num_workers=2,
    gradient_accumulation_steps=4,
    fp16=True,
)
# Calculating the weights
# Weightage = 1 - (num_of_samples_of_class)/(total_num_of_samples)
# less samples, more weightage

w_adequate = 1-len(df[df['discourse_effectiveness'] == 'adequate'])/len(train_df)
w_effective = 1-len(df[df['discourse_effectiveness'] == 'effective'])/len(train_df)
w_ineffective = 1-len(df[df['discourse_effectiveness'] == 'ineffective'])/len(train_df)

class_weights = torch.tensor(
    [w_adequate, w_effective, w_ineffective]
).cuda()

# huggingface has no straightforward way to incorparate class_weights as far as I know, 
# Hence we override the compute_loss function of the Trainer and introduce our class weighgts
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        # Class weighting
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

