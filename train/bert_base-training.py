import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from transformers import TrainingArguments
from torch.utils.data import TensorDataset

data = pd.read_csv("./Data/train_df.csv")
data = data.dropna(subset=['words'])
data = data.sample(10000)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
max_len = 128


def encode_text(texts, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_len,
                            padding='max_length',
                            return_attention_mask=True,
                            return_tensors='pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


input_ids, attention_masks = encode_text(data['words'], tokenizer, max_len)
train_inputs, validation_inputs, train_labels, validation_labels = \
    train_test_split(input_ids,
                     data['target'].values,
                     test_size=0.1,
                     random_state=42)
train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                       input_ids,
                                                       test_size=0.1,
                                                       random_state=42)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

train_dataset = TensorDataset(train_inputs,
                              train_masks,
                              train_labels)
eval_dataset = TensorDataset(validation_inputs,
                             validation_masks,
                             validation_labels)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=2)


def my_data_collator(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.tensor([f[2] for f in features])
    return batch


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_steps=0,
    weight_decay=0.01,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=my_data_collator
)

trainer.train()

model_path = "./model"
tokenizer_path = "./tokenizer"
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)
