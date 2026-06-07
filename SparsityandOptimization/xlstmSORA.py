

import math
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings("ignore")
torch.manual_seed(50)
np.random.seed(50)

def threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)

def effectiveRank(matrix: torch.Tensor, threshold: float = 0.01) -> int:
    
    with torch.no_grad():
        if matrix.dim() == 1:
            return int((matrix.abs() > threshold * matrix.abs().max()).sum().item())
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return int((S > threshold * S[0]).sum().item())

def trainableParams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SoRALinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        ) if bias else None

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.gate   = nn.Parameter(torch.empty(rank))  

        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        nn.init.normal_(self.gate, mean=0.0, std=0.1)
        
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        base = F.linear(x, self.weight, self.bias_param)

        lora_out = self.dropout(x) @ self.lora_A.T          
        lora_out = lora_out * self.gate                       
        lora_out = lora_out @ self.lora_B.T                  
        return base + self.scaling * lora_out

    def proximal_step(self, effective_lr: float, lam: float):
        
        with torch.no_grad():
            self.gate.data = threshold(self.gate.data, effective_lr * lam)

    def effectiveRank(self) -> int:
        return effectiveRank(self.gate)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, eff_rank={self.effectiveRank()}")

class SoRA_sLSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, rank: int = 8, lam: float = 1e-3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lam = lam

        self.R_z = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01, requires_grad=False)
        self.R_i = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01, requires_grad=False)
        self.R_f = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01, requires_grad=False)
        self.R_o = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01, requires_grad=False)

        self.W_z = SoRALinear(input_size, hidden_size, rank=rank)
        self.W_i = SoRALinear(input_size, hidden_size, rank=rank)
        self.W_f = SoRALinear(input_size, hidden_size, rank=rank)
        self.W_o = SoRALinear(input_size, hidden_size, rank=rank)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = x.size(0)
        if state is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)
            c = torch.zeros(B, self.hidden_size, device=x.device)
        else:
            h, c = state

        z = torch.tanh(self.W_z(x) + h @ self.R_z.T)
        i = torch.exp(torch.clamp(self.W_i(x) + h @ self.R_i.T, max=10.0))
        f = torch.exp(torch.clamp(self.W_f(x) + h @ self.R_f.T, max=10.0))
        o = torch.sigmoid(self.W_o(x) + h @ self.R_o.T)

        denom = f + i + 1e-8
        c_new = (f / denom) * c + (i / denom) * z

        c_norm = c_new / (torch.abs(c_new).max(dim=-1, keepdim=True).values.clamp(min=1.0))
        h_new = o * torch.tanh(c_norm)
        h_new = self.layer_norm(h_new)

        return h_new, (h_new, c_new)

    def proximal_step(self, gate_lr: float):
        
        for m in [self.W_z, self.W_i, self.W_f, self.W_o]:
            m.proximal_step(gate_lr, self.lam)

    def sora_layers(self) -> List[SoRALinear]:
        return [self.W_z, self.W_i, self.W_f, self.W_o]

class SoRA_mLSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, rank: int = 8, lam: float = 1e-3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lam = lam

        self.W_q = SoRALinear(input_size, hidden_size, rank=rank)
        self.W_k = SoRALinear(input_size, hidden_size, rank=rank)
        self.W_v = SoRALinear(input_size, hidden_size, rank=rank)

        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)
        self.W_o = nn.Linear(input_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, d = x.shape
        H = self.hidden_size

        if state is None:
            C = torch.zeros(B, H, H, device=x.device)
            n = torch.zeros(B, H, device=x.device)
        else:
            C, n = state

        q = self.W_q(x)                                          
        k = self.W_k(x) / math.sqrt(H)                          
        v = self.W_v(x)                                          

        i = torch.exp(torch.clamp(self.W_i(x), max=10.0))       
        f = torch.exp(torch.clamp(self.W_f(x), max=10.0))       
        o = torch.sigmoid(self.W_o(x))                          

        vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))          
        C_new = f.unsqueeze(2) * C + i.unsqueeze(2) * vk

        n_new = f * n + i * k

        num = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2)        
        denom = torch.abs(n_new * q).sum(dim=-1, keepdim=True).clamp(min=1.0)
        h = o * (num / denom)
        h = self.layer_norm(h)

        return h, (C_new, n_new)

    def proximal_step(self, gate_lr: float):
        for m in [self.W_q, self.W_k, self.W_v]:
            m.proximal_step(gate_lr, self.lam)

    def sora_layers(self) -> List[SoRALinear]:
        return [self.W_q, self.W_k, self.W_v]

class SoRA_xLSTM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        rank: int = 8,
        lam: float = 1e-3,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        from transformers import AutoModel

        deberta = AutoModel.from_pretrained(
            "microsoft/deberta-v3-base"
        )

        self.embedding = deberta.embeddings

        for param in self.embedding.parameters():
            param.requires_grad = False

        self.input_proj = nn.Linear(
            768,
            hidden_size
        )

        self.lam = lam

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.cells.append(SoRA_sLSTMCell(hidden_size, hidden_size, rank=rank, lam=lam))
            else:
                self.cells.append(SoRA_mLSTMCell(hidden_size, hidden_size, rank=rank, lam=lam))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        x = self.embedding(
            input_ids=input_ids
        )

        x = x.float()

        x = self.input_proj(x)

        states = [None] * len(self.cells)

        T = x.size(1)

        all_outputs = []

        for t in range(T):

            xt = x[:, t, :]

            for i, cell in enumerate(self.cells):
                xt, states[i] = cell(
                    xt,
                    states[i]
                )

                xt = self.dropout(xt)

            all_outputs.append(xt)

        outputs = torch.stack(
            all_outputs,
            dim=1
        )

        masked = outputs * attention_mask.unsqueeze(-1)

        pooled = masked.sum(dim=1)

        lengths = attention_mask.sum(
            dim=1,
            keepdim=True
        ).clamp(min=1)

        pooled = pooled / lengths

        logits = self.classifier(pooled)

        return logits

    def proximal_step(self, gate_lr: float):
        for cell in self.cells:
            cell.proximal_step(gate_lr)

    def all_sora_layers(self) -> List[SoRALinear]:
        layers = []
        for cell in self.cells:
            layers.extend(cell.sora_layers())
        return layers

def make_synthetic_cola(
    n_train: int = 800,
    n_val: int = 200,
    vocab_size: int = 500,
    max_len: int = 32,
    seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    
    rng = np.random.default_rng(seed)

    def make_sample(label: int) -> Dict:
        if label == 1:
            length = rng.integers(10, max_len)
            ids = rng.integers(1, vocab_size, size=length).tolist()
        else:
            length = rng.integers(3, 12)
            
            base = rng.integers(1, vocab_size, size=max(3, length // 3)).tolist()
            ids = (base * 4)[:length]
        pad = [0] * (max_len - len(ids))
        mask = [1] * len(ids) + [0] * len(pad)
        return {
            "input_ids": torch.tensor(ids + pad, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    class SyntheticCoLA(torch.utils.data.Dataset):
        def __init__(self, n):
            self.data = [make_sample(i % 2) for i in range(n)]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    return SyntheticCoLA(n_train), SyntheticCoLA(n_val)

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_main: torch.optim.Optimizer,  
    optimizer_gate: torch.optim.SGD,        
    device: torch.device,
    gate_lr: float,                         
    prune: bool = True,                     
) -> Dict[str, float]:
    
    model.train()
    total_loss = correct = total = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer_main.zero_grad()
        optimizer_gate.zero_grad()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss   = F.cross_entropy(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0
        )
        optimizer_main.step()

        optimizer_gate.step()
        if prune and hasattr(model, "proximal_step"):
            model.proximal_step(gate_lr)

        total_loss += loss.item() * len(labels)
        preds  = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total  += len(labels)

    return {"loss": total_loss / total, "accuracy": correct / total}

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    
    import evaluate as evaluate_lib
    cola_metric = evaluate_lib.load("glue", "cola")

    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss   = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    result = cola_metric.compute(predictions=all_preds, references=all_labels)
    mcc = result["matthews_correlation"]

    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "mcc":      mcc,
    }

def _collect_gate_params(model: nn.Module) -> List[nn.Parameter]:
    
    gate_params = []
    for m in model.modules():
        if isinstance(m, SoRALinear):
            gate_params.append(m.gate)
    return gate_params

def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
    gate_lr: float = 1e-2,  
    lam: float = 1e-3,
    warmup_epochs: int = 1,  
) -> Dict:
    
    model = model.to(device)

    gate_params = _collect_gate_params(model)
    gate_ids    = {id(p) for p in gate_params}

    optimizer_main = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids],
        lr=lr,
    )
    
    optimizer_gate = torch.optim.SGD(gate_params, lr=gate_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_main, T_max=epochs)

    n_params = trainableParams(model)
    n_gate   = sum(p.numel() for p in gate_params)
    print(model_name, n_params,  n_gate)
    print(lam ,gate_lr,gate_lr*lam, warmup_epochs)

    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        
        prune = (ep > warmup_epochs)
        train_metrics = train_epoch(
            model, train_loader, optimizer_main, optimizer_gate,
            device, gate_lr, prune=prune,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if hasattr(model, "all_sora_layers"):
            all_layers  = model.all_sora_layers()
            total_slots = sum(l.rank for l in all_layers)
            active      = sum((l.gate.data.abs() > 1e-9).sum().item() for l in all_layers)
            avg_er      = np.mean([l.effectiveRank() for l in all_layers])
            sparsity_str = f"  active={active}/{total_slots}  avgRank={avg_er:.1f}"
        else:
            sparsity_str = ""

        history.append({
            "epoch":       ep,
            "train_loss":  train_metrics["loss"],
            "train_acc":   train_metrics["accuracy"],
            "val_loss":    val_metrics["loss"],
            "val_acc":     val_metrics["accuracy"],
            "val_mcc":     val_metrics["mcc"],
        })
        print(f"  Ep {ep}/{epochs} | "
              f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} | "
              f"val_acc={val_metrics['accuracy']:.3f} mcc={val_metrics['mcc']:.3f}"
              f"{sparsity_str}")

    elapsed = time.time() - t0

    eff_ranks = []
    if hasattr(model, "all_sora_layers"):
        for layer in model.all_sora_layers():
            eff_ranks.append(layer.effectiveRank())

    final = history[-1]
    result = {
        "model":               model_name,
        "trainable_params":    n_params,
        "val_acc":             final["val_acc"],
        "val_mcc":             final["val_mcc"],
        "avg_effectiveRank":  np.mean(eff_ranks) if eff_ranks else None,
        "effectiveRanks":     eff_ranks,
        "training_time_sec":   elapsed,
        "history":             history,
    }
    avg_er = result["avg_effectiveRank"]
    print(f"\n{elapsed:.1f}s | valMcc={final['val_mcc']:.3f} | "
          f"avgRank={avg_er:.2f}" if avg_er is not None else
          f"\n{elapsed:.1f}s | valMCC={final['val_mcc']:.3f}")
    return result

def runtest():
    device = torch.device("mps")
    B, T, D, R = 4, 16, 32, 4
    VOCAB = 100

    layer = SoRALinear(D, D, rank=R)
    x = torch.randn(B, D)
    out = layer(x)
    assert out.shape == (B, D),{out.shape}
    
    layer.gate.data = torch.randn(R)
    layer.proximal_step(effective_lr=1.0, lam=10.0)
    assert layer.gate.data.abs().max() < 1e-6

    cell = SoRA_sLSTMCell(D, D, rank=R)
    x1 = torch.randn(B, D)
    h, state = cell(x1)
    assert h.shape == (B, D)
    h2, _ = cell(x1, state)
    assert h2.shape == (B, D)
    cell.proximal_step(gate_lr=1e-3)

    cell2 = SoRA_mLSTMCell(D, D, rank=R)
    h, state = cell2(x1)
    assert h.shape == (B, D)
    h2, _ = cell2(x1, state)
    assert h2.shape == (B, D)

    xlstm = SoRA_xLSTM(VOCAB, embed_dim=D, hidden_size=D, num_layers=2, rank=R)
    ids = torch.randint(0, VOCAB, (B, T))
    logits = xlstm(ids)
    assert logits.shape == (B, 2), {logits.shape}
    xlstm.proximal_step(gate_lr=1e-3)
    sl = SoRALinear(D, D, rank=8)
    sl.gate.data = torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.8])
    er = sl.effectiveRank()
    assert er == 4,er



import pandas as pd
from transformers import AutoTokenizer

class ColaDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.encodings["input_ids"][idx],      dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],                      dtype=torch.long),
        }

    def __len__(self):
        return len(self.labels)

def load_cola(
    train_path: str = "GLUE-baselines/glue_data/CoLA/train.tsv",
    dev_path:   str = "GLUE-baselines/glue_data/CoLA/dev.tsv",
    tokenizer_name: str = "microsoft/deberta-v3-base",
    max_length: int = 128,
    batch_size: int = 16,
):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_df = pd.read_csv(train_path, sep="\t", header=None)
    dev_df   = pd.read_csv(dev_path,   sep="\t", header=None)

    train_enc = tokenizer(train_df[3].tolist(), truncation=True, padding=True, max_length=max_length)
    dev_enc   = tokenizer(dev_df[3].tolist(),   truncation=True, padding=True, max_length=max_length)

    train_ds = ColaDataset(train_enc, train_df[1].tolist())
    dev_ds   = ColaDataset(dev_enc,   dev_df[1].tolist())

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(dev_ds,   batch_size=batch_size)

    vocab_size = tokenizer.vocab_size
    return train_loader, val_loader, vocab_size

def main():
    runtest()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    EMBED_DIM  = 128
    HIDDEN     = 256
    RANK       = 8     
    NUM_LAYERS = 4

    LAM        = 1e-3   
    LR         = 2e-4   
    GATE_LR    = 1e-3   
    WARMUP     = 1      
    EPOCHS     = 5      
    BATCH      = 16     

    train_loader, val_loader, VOCAB_SIZE = load_cola(
        batch_size=BATCH,
    )

    xlstm = SoRA_xLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        rank=RANK,
        lam=LAM,
    )

    r = run_experiment(
        "SoRA-xLSTM (CoLA)", xlstm, train_loader, val_loader,
        device, epochs=EPOCHS, lr=LR, gate_lr=GATE_LR, lam=LAM,
        warmup_epochs=WARMUP,
    )
    print(f"{r['model']:<25} "
          f"{r['trainable_params']:>10,} "
          f"{r['val_acc']:>9.3f} "
          f"{r['val_mcc']:>9.3f} "
          f"{r['training_time_sec']:>9.1f}")
    for i, layer in enumerate(xlstm.all_sora_layers()):
        n_active = (layer.gate.data.abs() > 1e-9).sum().item()

        print(
            i,
            n_active,
            layer.rank,
            layer.effectiveRank(),
            [f"{v}" for v in layer.gate.data.tolist()]
        )
    return r

if __name__ == "__main__":
    main()