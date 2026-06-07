import time
import types
import numpy as np
import pandas as pd
import evaluate
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from peft.tuners.lora.layer import Linear as LoraLinear

RANK = 8
ALPHA = 16
LAMBDA = 0.05
GATE_THRESHOLD = 0.05

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

    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            k: torch.tensor(v[idx])
            for k, v in self.encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx]
        )

        return item

    def __len__(self):
        return len(self.labels)


trainSet = ColaDataset(
    trainEncodings,
    trainLabels
)

devSet = ColaDataset(
    devEncodings,
    devLabels
)

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2
)

lora = LoraConfig(
    task_type=TaskType.SEQ_CLS,

    r=RANK,

    lora_alpha=ALPHA,

    lora_dropout=0.1,

    bias="none",

    target_modules=[
        "query_proj",
        "key_proj",
        "value_proj",
        "out_proj",
    ],
)

model = get_peft_model(
    model,
    lora
)

soraLayers = []


def patchSoraLayer(module):

    if not isinstance(module, LoraLinear):
        return

    adapterName = "default"

    if adapterName not in module.lora_A:
        return

    rank = module.r[adapterName]

    module.gate = nn.Parameter(
        0.1 * torch.ones(rank)
    )

    oldForward = module.forward

    def soraForward(self, x, *args, **kwargs):

        result = oldForward(x, *args, **kwargs)

        A = self.lora_A["default"]

        B = self.lora_B["default"]

        dropout = self.lora_dropout["default"]

        scaling = self.scaling["default"]

        dtype = x.dtype

        h = A(dropout(x))

        h = h * self.gate.to(dtype)

        delta = B(h.to(dtype))

        return (
            result
            - scaling * B(A(dropout(x)))
            + scaling * delta
        )

    module.forward = types.MethodType(
        soraForward,
        module
    )

    soraLayers.append(module)


for module in model.modules():
    patchSoraLayer(module)

trainableParams = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(trainableParams)

model.print_trainable_parameters()

metric = evaluate.load(
    "glue",
    "cola"
)


def computeMetrics(evalPred):

    logits, labels = evalPred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return metric.compute(
        predictions=predictions,
        references=labels
    )


def getSoraStats():

    totalRank = 0
    activeRank = 0

    perLayer = []

    for i, layer in enumerate(
        soraLayers
    ):

        gate = layer.gate.detach()

        rank = gate.numel()

        effectiveRank = (
            gate.abs()
            > GATE_THRESHOLD
        ).sum().item()

        totalRank += rank
        activeRank += effectiveRank

        perLayer.append(
            (
                i,
                effectiveRank,
                rank
            )
        )

    sparsity = (
        1
        - activeRank / totalRank
    )

    return (
        perLayer,
        activeRank,
        totalRank,
        sparsity
    )


class SoRAProximalCallback(
    TrainerCallback
):

    def on_step_end(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs
    ):

        lr = args.learning_rate

        threshold = 5 * lr * LAMBDA

        with torch.no_grad():

            with torch.no_grad():

                for layer in soraLayers:

                    g = layer.gate.data.float()

                    g.copy_(
                        torch.sign(g)
                        * torch.clamp(
                            g.abs() - threshold,
                            min=0.0
                        )
                    )

                    layer.gate.data.copy_(g)

        return control


trainingArgs = TrainingArguments(
    output_dir="./sora_results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-4,

    warmup_ratio=0.06,

    lr_scheduler_type="cosine",

    per_device_train_batch_size=16,

    gradient_accumulation_steps=2,

    per_device_eval_batch_size=16,

    num_train_epochs=5,

    weight_decay=0.01,

    logging_steps=50,

    max_grad_norm=1.0,

    load_best_model_at_end=True,

    metric_for_best_model="matthews_correlation",

    save_total_limit=2,

    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,

    args=trainingArgs,

    train_dataset=trainSet,

    eval_dataset=devSet,

    compute_metrics=computeMetrics,

    callbacks=[
        SoRAProximalCallback()
    ]
)

start = time.time()

trainer.train()

end = time.time()

results = trainer.evaluate()

(
    perLayer,
    activeRank,
    totalRank,
    sparsity
) = getSoraStats()

print(results["eval_loss"])

print(
    results[
        "eval_matthews_correlation"
    ]
)

print(end - start)

print(trainableParams)

print(activeRank)

print(totalRank)

print(sparsity)

for layerId, eff, total in perLayer:

    print(layerId)

    print(eff)

    print(total)

model.save_pretrained("./sora")