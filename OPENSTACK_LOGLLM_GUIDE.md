# Training LogLLM on OpenStack Dataset - Complete Guide

This guide walks you through training LogLLM for anomaly detection on your OpenStack log data.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Model Setup](#model-setup)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM (recommended: 24GB+)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space for models and data

### Software Requirements
```bash
# Python 3.8+
python --version

# CUDA 12.1+ (for GPU training)
nvcc --version

# Required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate
pip install pandas numpy tqdm scikit-learn
```

---

## Data Preparation

### Step 1: Verify Your Data Files

You should have the following files in `/Users/nakulbhardwaj/Developer/PersonalProjects/BetterLogs/OpenStackData/`:

```
openstack_normal1.log (52,312 lines)
openstack_normal2.log (137,074 lines)
openstack_abnormal.log (18,434 lines)
```

### Step 2: Run Data Preparation Script

```bash
cd /Users/nakulbhardwaj/Developer/PersonalProjects/BetterLogs/LogLLM/prepareData
python3 prepare_openstack.py
```

**Expected Output:**
```
Normal windows: 1894, Abnormal windows: 185
Train windows: 1663 (Normal: 1515, Abnormal: 148) - 8.90% anomaly ratio
Test windows: 416 (Normal: 379, Abnormal: 37) - 8.89% anomaly ratio
```

**Generated Files:**
- `OpenStackData/train.csv` - Training data
- `OpenStackData/test.csv` - Test data
- `OpenStackData/openstack_combined.log` - Combined raw logs
- `OpenStackData/openstack_combined.log_structured.csv` - Parsed logs

---

## Model Setup

### Step 3: Download Pre-trained Models

You need two base models:

#### 1. BERT Model (bert-base-uncased)
```bash
# Option A: Let transformers download automatically (easier)
# Just run the training script and it will download

# Option B: Manual download
huggingface-cli download bert-base-uncased
```

#### 2. Llama 3 8B Model (Meta-Llama-3-8B)

**Important:** This requires access approval from Meta.

1. Request access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B
2. After approval:
```bash
huggingface-cli login  # Login with your token
huggingface-cli download meta-llama/Meta-Llama-3-8B
```

**Alternative:** If you can't get Llama 3 access, you can try:
- Llama 2 7B: `meta-llama/Llama-2-7b-hf` (also requires approval)
- Or modify the code to use a different LLM

---

## Training

### Step 4: Configure Training Script

Edit `train_openstack.py` if needed:

```python
# Update these paths if you downloaded models locally
Bert_path = r"bert-base-uncased"  # or local path
Llama_path = r"meta-llama/Meta-Llama-3-8B"  # or local path

# Adjust batch size based on your GPU memory
batch_size = 16          # Full batch size
micro_batch_size = 4     # Reduce to 2 or 1 if OOM errors

# Training epochs (increase for better performance)
n_epochs_1 = 1      # Stage 1
n_epochs_2_1 = 1    # Stage 2.1
n_epochs_2_2 = 1    # Stage 2.2
n_epochs_3 = 2      # Stage 3 (main training)
```

### Step 5: Run Training

```bash
cd /Users/nakulbhardwaj/Developer/PersonalProjects/BetterLogs/LogLLM
python3 train_openstack.py
```

**Training Stages:**

1. **Stage 1 (1 epoch)**: Fine-tune Llama answer template with ~1,000 samples
2. **Stage 2.1 (1 epoch)**: Train the projector layer (BERT → Llama embedding alignment)
3. **Stage 2.2 (1 epoch)**: Train projector + BERT together
4. **Stage 3 (2 epochs)**: Fine-tune the entire model end-to-end

**Expected Training Time:**
- With 24GB GPU: ~2-4 hours
- With 16GB GPU: ~4-6 hours (with reduced batch size)
- CPU: Not recommended (would take days)

**Output:**
- Fine-tuned model saved to: `LogLLM/ft_model_OpenStack/`

---

## Evaluation

### Step 6: Evaluate the Model

```bash
cd /Users/nakulbhardwaj/Developer/PersonalProjects/BetterLogs/LogLLM
python3 eval_openstack.py
```

**Expected Output:**
```
Precision: 0.XXXX
Recall:    0.XXXX
F1-Score:  0.XXXX

Confusion Matrix:
                Predicted
               Normal  Anomalous
Actual Normal    XXX     XXX
      Anomalous  XXX     XXX
```

**Evaluation Metrics:**
- **Precision**: Of all predicted anomalies, how many were actually anomalous?
- **Recall**: Of all actual anomalies, how many did we detect?
- **F1-Score**: Harmonic mean of precision and recall (overall performance)

**Results saved to:** `LogLLM/evaluation_results_OpenStack.txt`

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)
**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# In train_openstack.py, reduce:
micro_batch_size = 2  # or even 1
```

#### 2. Models Not Found
**Error:** `OSError: meta-llama/Meta-Llama-3-8B does not exist`

**Solution:**
```bash
# Login to Hugging Face
huggingface-cli login

# Download model
huggingface-cli download meta-llama/Meta-Llama-3-8B
```

#### 3. No GPU Available
**Warning:** `CUDA not available. Training on CPU...`

**Solution:**
- Install CUDA toolkit and cuDNN
- Install PyTorch with CUDA:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

#### 4. Dataset Loading Error
**Error:** `FileNotFoundError: train.csv`

**Solution:**
```bash
# Re-run data preparation
cd LogLLM/prepareData
python3 prepare_openstack.py
```

#### 5. Import Errors
**Error:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
pip install transformers datasets peft accelerate tqdm scikit-learn
```

---

## Understanding the Results

### What is Good Performance?

For log anomaly detection:
- **F1-Score > 0.85**: Excellent
- **F1-Score 0.70-0.85**: Good
- **F1-Score 0.50-0.70**: Fair (consider retraining)
- **F1-Score < 0.50**: Poor (check data/config)

### Improving Performance

If results are poor:

1. **Increase Training Epochs**
   ```python
   n_epochs_3 = 5  # Increase main training epochs
   ```

2. **Balance Dataset Better**
   ```python
   min_less_portion = 0.4  # Increase minority class to 40%
   ```

3. **Adjust Learning Rates**
   ```python
   lr_3 = 1e-5  # Try smaller learning rate
   ```

4. **Get More Data**
   - Download full OpenStack dataset from Loghub
   - Current dataset is relatively small

---

## File Structure

```
BetterLogs/
├── OpenStackData/
│   ├── openstack_normal1.log (original data)
│   ├── openstack_normal2.log
│   ├── openstack_abnormal.log
│   ├── train.csv (generated - training data)
│   └── test.csv (generated - test data)
├── LogLLM/
│   ├── prepareData/
│   │   └── prepare_openstack.py (data preparation script)
│   ├── train_openstack.py (training script)
│   ├── eval_openstack.py (evaluation script)
│   ├── ft_model_OpenStack/ (generated - fine-tuned model)
│   └── evaluation_results_OpenStack.txt (generated - results)
└── OPENSTACK_LOGLLM_GUIDE.md (this guide)
```

---

## Next Steps

1. ✅ Data prepared and verified
2. ⏳ Download base models (BERT + Llama)
3. ⏳ Run training script
4. ⏳ Evaluate model
5. ⏳ Use for anomaly detection on new logs

---

## References

- **LogLLM Paper**: https://arxiv.org/abs/2411.08561
- **LogLLM GitHub**: https://github.com/guanwei49/LogLLM
- **Loghub Dataset**: https://github.com/logpai/loghub
- **BERT Model**: https://huggingface.co/bert-base-uncased
- **Llama 3 Model**: https://huggingface.co/meta-llama/Meta-Llama-3-8B

---

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Review the LogLLM repository issues
3. Check CUDA/PyTorch installation

**Good luck with your training!** 🚀
