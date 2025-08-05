# Medical Chatbot Using DeepSeek-R1 (Fine-tuning with Unsloth, Colab, and HuggingFace)

This project demonstrates how to build and fine-tune a medical chatbot leveraging the DeepSeek-R1-Distill-Llama-8B model using the [Unsloth](https://github.com/unslothai/unsloth) framework in Google Colab. The workflow walks through environment setup, data preparation, model loading, fine-tuning, and inference, focusing on clinical reasoning and chain-of-thought (CoT) style answers for medical questions.

## ⚠️ Note

> My previous GitHub account was unexpectedly suspended. This project was originally created earlier and has been re-uploaded here. All work was done gradually over time, and original commit history has been preserved where possible.


## Table of Contents

- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Workflow Overview](#workflow-overview)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Fine-tuning Approach](#fine-tuning-approach)
- [Inference Example](#inference-example)
- [Troubleshooting](#troubleshooting)
- [Library Versions](#library-versions)
- [Credits](#credits)

---

## Features

- **Colab-compatible** workflow with GPU (e.g., Tesla T4)
- Uses HuggingFace and Unsloth for efficient LLM fine-tuning
- LoRA (Low-Rank Adaptation) for parameter-efficient training
- Chain-of-thought (CoT) reasoning for medical Q&A
- Integration with Weights & Biases (wandb) for experiment tracking

---

## Setup & Installation

1. **Install Unsloth and dependencies:**
    ```python
    !pip install unsloth
    !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
    ```

2. **Import libraries:**
    ```python
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    from trl import SFTTrainer
    from huggingface_hub import login
    from transformers import TrainingArguments
    from datasets import load_dataset
    import wandb
    ```

3. **Authenticate with HuggingFace & wandb:**
    - Store your HF token in Colab's `userdata` and login.
    - Store your wandb API key similarly and login.

---

## Workflow Overview

1. **Environment Setup**
    - Install necessary libraries
    - Check GPU and CUDA availability
2. **Model Loading**
    - Load DeepSeek-R1-Distill-Llama-8B via Unsloth in 4-bit mode
3. **Prompt Engineering**
    - System prompt guides clinical reasoning and chain-of-thought answers
4. **Inference (Zero-shot)**
    - Run initial inference to test default model behavior
5. **Dataset Preparation**
    - Use [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) for training
    - Preprocess with custom prompt formatting (CoT style)
6. **Fine-tuning**
    - Apply LoRA
    - Setup SFTTrainer with custom formatting and training arguments
    - Track with wandb
7. **Post-training Inference**
    - Evaluate model on medical questions and observe improvements

---

## Dataset

- **Source:** [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Contents:** Medical questions, step-by-step reasoning (Complex_CoT), and final answers (Response)
- **Usage:** Only a subset (e.g., `train[:500]`) is loaded for demo purposes

---

## Model Details

- **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Framework:** Un
