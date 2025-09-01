# HyLIGHT-Cache Experiment Pipeline

This repository is a **ready-to-run** skeleton for your HyLIGHT-Cache study:
parser → feature/label → training (Ensemble + LR-SGD with decay) → full metrics → cache replay.

It includes:
- A **unified trace schema** and adapters for **LRB/Wikipedia CDN** and **Twitter KV (TTL)**,
- Label generation via **Relaxed Belady** (standalone) and **hooks** for **libCacheSim/Cachebench**,
- Ensemble (LightGBM/GaussianNB/ELM) with **soft voting**, plus an **explicit LR + SGD (with decay)** baseline,
- Evaluation: classification metrics (Accuracy/Precision/Recall/F1/AUC) **and** cache metrics (Hit Rate, Byte Hit Rate, Avg Latency, Bandwidth).

> ✅ You can run a complete **synthetic** end-to-end demo without any external data.
> 🧪 Replace the synthetic/data adapters with real traces to reproduce your experiments.
