# 可用，随机种子 可以复现内容，好代码
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    get_linear_schedule_with_warmup
from dataset_transform import raw_datasets
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
import pandas as pd
from torch.optim import AdamW
import random
import torch

random.seed(45)
np.random.seed(45)
torch.manual_seed(45)
torch.cuda.manual_seed_all(45)

# 加载数据和分词器
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# 定义标签对齐和分词函数
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


# 处理数据集
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# 确保 'test' 数据集存在
assert "test" in tokenized_datasets, "The 'test' dataset is missing from tokenized_datasets."

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = evaluate.load("seqeval")


# 定义评价指标计算函数
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# 标签映射
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


# 设置分层学习率的参数
def get_optimizer_params(model, base_learning_rate, layer_decay=1):
    optimizer_grouped_parameters = []
    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)

    # 对不同的层设置不同的学习率
    for i, layer in enumerate(layers):
        lr = base_learning_rate * (layer_decay ** (len(layers) - i - 1))
        optimizer_grouped_parameters += [
            {"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr}
        ]

    # 输出层使用基础学习率
    optimizer_grouped_parameters += [
        {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": base_learning_rate}
    ]

    return optimizer_grouped_parameters


# 训练和评估函数
def train_and_evaluate(base_learning_rate, batch_size, weight_decay, warmup_ratio=0.1, layer_decay=0.95):
    # 加载模型
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    # 设置训练参数
    args = TrainingArguments(
        output_dir="bert-finetuned-ner",
        evaluation_strategy="epoch",
        learning_rate=base_learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )

    # 获取优化器和调度器
    optimizer_params = get_optimizer_params(model, base_learning_rate, layer_decay)
    optimizer = AdamW(optimizer_params, lr=base_learning_rate)

    total_steps = len(tokenized_datasets["train"]) // batch_size * args.num_train_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
    )

    # 训练模型
    trainer.train()

    # 在验证集上评估
    validation_results = trainer.evaluate()
    # 在测试集上评估
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

    return {
        "learning_rate": base_learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "validation_loss": validation_results["eval_loss"],
        "validation_f1": validation_results["eval_f1"],
        "test_loss": test_results["eval_loss"],
        "test_f1": test_results["eval_f1"]
    }


# 定义要进行 Grid Search 的参数
learning_rates = [5e-5, 8e-5]
batch_sizes = [8, 2]
weight_decays = [0.0]

# 存储所有结果的列表
all_results = []

# 进行 Grid Search
for lr in learning_rates:
    for bz in batch_sizes:
        for wd in weight_decays:
            print(f"\nTraining with learning_rate={lr}, batch_size={bz}, weight_decay={wd}")
            results = train_and_evaluate(base_learning_rate=lr, batch_size=bz, weight_decay=wd)
            all_results.append(results)

# 转换结果为 DataFrame 并显示
results_df = pd.DataFrame(all_results)
print("\nAll Results:")
print(results_df)

# 找出 F1 得分最高的组合
best_result = results_df.loc[results_df["validation_f1"].idxmax()]
print("\nBest Model Parameters:")
print(best_result)
