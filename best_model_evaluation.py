from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    get_linear_schedule_with_warmup
from dataset_transform import raw_datasets
from transformers import DataCollatorForTokenClassification
import numpy as np
import torch
from torch.optim import AdamW
import random
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

# 1. 设置随机种子
random.seed(45)
np.random.seed(45)
torch.manual_seed(45)
torch.cuda.manual_seed_all(45)

# 2. 加载数据和分词器
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# 3. 定义标签对齐和分词函数
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# 4. 处理数据集
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# 5. 创建数据整理器
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# 6. 定义评估函数
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # 转换为实体标签
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 打印详细分类报告
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))

    metrics = {}
    return metrics


# 7. 标签映射
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# 8. 使用最佳参数
best_params = {
    "base_learning_rate": 5e-5,
    "batch_size": 2,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "layer_decay": 1
}

# 9. 创建模型
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)


# 10. 定义分层学习率函数
def get_optimizer_params(model, base_learning_rate, layer_decay=0.95):
    optimizer_grouped_parameters = []
    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)

    for i, layer in enumerate(layers):
        lr = base_learning_rate * (layer_decay ** (len(layers) - i - 1))
        optimizer_grouped_parameters += [
            {"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr}
        ]

    optimizer_grouped_parameters += [
        {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": base_learning_rate}
    ]

    return optimizer_grouped_parameters


# 11. 设置训练参数
training_args = TrainingArguments(
    output_dir="bert-finetuned-ner-best",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_params["base_learning_rate"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    num_train_epochs=3,
    weight_decay=best_params["weight_decay"],
    warmup_ratio=best_params["warmup_ratio"],
)

# 12. 设置优化器和调度器
optimizer_params = get_optimizer_params(
    model,
    best_params["base_learning_rate"],
    best_params["layer_decay"]
)
optimizer = AdamW(optimizer_params)

total_steps = len(tokenized_datasets["train"]) // best_params["batch_size"] * training_args.num_train_epochs
warmup_steps = int(best_params["warmup_ratio"] * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 13. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
)

if __name__ == "__main__":
    # 14. 训练模型
    print("开始训练模型...")
    trainer.train()

    # 15. 在测试集上评估
    print("\n在测试集上评估模型...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("\n测试集评估结果:")
    print(test_results)