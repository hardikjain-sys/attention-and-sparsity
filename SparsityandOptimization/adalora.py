import numpy as np
import evaluate
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import (
    AdaLoraConfig,
    get_peft_model,
    TaskType
)


tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base"
)

train = pd.read_csv(
    "GLUE-baselines/glue_data/CoLA/train.tsv",
    sep="\t",
    header=None
)

dev = pd.read_csv(
    "GLUE-baselines/glue_data/CoLA/dev.tsv",
    sep="\t",
    header=None
)

trainSentences = train[3].tolist()
trainLabels = train[1].tolist()

devSentences = dev[3].tolist()
devLabels = dev[1].tolist()

trainEncodings = tokenizer(
    trainSentences,
    truncation=True,
    padding=True,
    max_length=128
)

devEncodings = tokenizer(
    devSentences,
    truncation=True,
    padding=True,
    max_length=128
)


class ColaDataset(torch.utils.data.Dataset):

    def __init__(self, encoding, label):
        self.encoding = encoding
        self.label = label

    def __getitem__(self, index):

        x = {
            key: torch.tensor(val[index])
            for key, val in self.encoding.items()
        }

        x["labels"] = torch.tensor(self.label[index])

        return x

    def __len__(self):
        return len(self.label)


trainSet = ColaDataset(trainEncodings, trainLabels)

devSet = ColaDataset(devEncodings, devLabels)

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2
)

for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.pooler.parameters():
    param.requires_grad = True

adaLora = AdaLoraConfig(
    task_type=TaskType.SEQ_CLS,

    init_r=12,

    target_r=8,

    beta1=0.85,

    beta2=0.85,

    tinit=10,

    tfinal=1000,

    deltaT=50,

    lora_alpha=16,

    lora_dropout=0.1,

    target_modules=[
        "query_proj",
        "key_proj",
        "value_proj",
        "out_proj"
    ],

    total_step=1340
)

model = get_peft_model(model, adaLora)

model.print_trainable_parameters()

metric = evaluate.load("glue", "cola")

print(type(metric))


def computeMetrics(evalPred):

    logits, labels = evalPred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(
        predictions=predictions,
        references=labels
    )


trainingArgs = TrainingArguments(
    output_dir="./results/adalora",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-4,

    warmup_ratio=0.06,

    lr_scheduler_type="cosine",

    per_device_train_batch_size=16,

    gradient_accumulation_steps=2,

    per_device_eval_batch_size=16,

    num_train_epochs=8,

    weight_decay=0.01,

    logging_steps=50,

    max_grad_norm=1.0,

    load_best_model_at_end=True,

    metric_for_best_model="matthews_correlation",

    save_total_limit=2,

    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,

    args=trainingArgs,

    train_dataset=trainSet,

    eval_dataset=devSet,

    compute_metrics=computeMetrics
)

trainer.train()

results = trainer.evaluate()

print(results)

for name, module in model.named_modules():

    if hasattr(module, "lora_E"):

        print(name)

        E = module.lora_E["default"]

        effectiveRank = (E > 1e-6).sum().item()

        print(effectiveRank)

        print(E.detach().cpu().numpy())

model.save_pretrained("./adalora")