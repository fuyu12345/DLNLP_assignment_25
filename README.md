# DLNLP_assignment_25 - SR-R1 Training Pipeline

This repository contains the implementation of the Elo-based Self-Rewarding GRPO R1 (SR-R1) training pipeline, an innovative framework designed to enhance the alignment and reasoning capabilities of Large Language Models (LLMs) without relying on costly human-labeled data. SR-R1 is an extension of the Group Relative Policy Optimization (GRPO) algorithm, introducing a novel self-rewarding mechanism that enables LLMs to evaluate their own outputs during training.





## 🚨 Important Notification

This project is heavily built upon two primary external libraries:

1. [**TRL (Transformers Reinforcement Learning)**](https://github.com/huggingface/trl) — A robust framework provided by HuggingFace for implementing Reinforcement Learning with Language Models. It forms the backbone for **GRPO-based optimization** in SR-R1.

2. [**Open-R1**](https://github.com/huggingface/open-r1) — The main framework that **SR-R1** extends upon. Most of the directory structure, training utilities, and model handling are derived from Open-R1, with important modifications specifically for the **self-rewarding mechanism** and **Elo-based ranking**.

### 📁 **SR-R1 Folder Structure**
The core implementation of SR-R1 is located in the `SR-R1` folder, which is an extension of the Open-R1 framework. Key modifications include:


### 📌 `accelerate_config/`
- **zero3.yaml**: 
  Configuration file for DeepSpeed's ZeRO-3 optimization. This setup enables memory-efficient training for large-scale models by partitioning model states across multiple GPUs. 



### 📌 `config_setting/`
- **config_demo.yaml**:
  This YAML file contains the configuration settings required for SR-R1 training. Key parameters include:
  - `model_name`: Specifies the model to be used, such as `Qwen2.5-1.5B-Instruct`.
  - `batch_size`: Defines the batch size for both training and evaluation.
  - `learning_rate`: The learning rate for the optimizer.
  - `num_epochs`: Number of epochs to iterate over the training dataset.
  - `num_generations`: Number of completions generated for each prompt during training.
  - `reward_weights`: Weights for different types of rewards, including self-judging and accuracy-based metrics.

This file serves as the main configuration interface for fine-tuning and optimizing the SR-R1 pipeline.



### 📌 `utils/`
The `utils` folder contains utility scripts and configurations essential for the SR-R1 training process. 

### 📌 `__init__.py` 
  - Unchanged from the original Open-R1 implementation. It simply marks the folder as a Python module.

### 📌 `configs.py` 
  - This file remains identical to the Open-R1 repository.
  - It provides configuration parsing and initialization logic for the training environment.

### 📌 `grpo_new.py` 
  - This is a **heavily modified version** of the original GRPO trainer to accommodate SR-R1's enhancements:
    - **Import Path Modifications**:
      Updated import paths to integrate with the self-judging mechanism and the new reward calculation methods.
    - **Self-Judge Reward Function Registration**:
      Introduced the `self_judge_reward_func` into the `REWARD_FUNCS_REGISTRY`, enabling it as a valid reward function during training.
    - **GRPO Trainer Initialization**:
      - Cut the dataset into smaller chunks, specifically splitting the training dataset into four equal parts.
      - These chunks are sequentially fed into the `GRPOTrainer()` for batched processing.
    

### 📌 `rewards.py` 
This is a **heavily modified version** of the original `rewards.py` to accommodate SR-R1's self-rewarding enhancements. It introduces the **Elo-based self-rewarding mechanism** and several utility functions to manage prompt extraction, pairwise ranking, and reward calculation. Below is a list of the main functions related to the self-rewarding mechanism:

- `extract_prompt_text()`: Extracts the main content from the prompt.  
- `make_meta_judge_chat_prompt()`: Generates the structured judge prompt for pairwise comparisons.  
- `parse_winner()`: Parses the winner from the judge's output.  
- `run_elo()`: Executes the Elo ranking algorithm on pairwise results.  
- `self_judge_reward_func()`: The core of the self-rewarding mechanism, integrating Elo-based scoring into the GRPO framework.  


---



## 🚀 Installation and Usage

To install all required packages:

```bash
pip install -r requirements.txt
```

## 🧪 Running the Code

Once your environment is set up, you can run different scripts by modifying the script paths as needed in your Python code.

To execute the main test pipeline, run:

```bash
python main.py
```

This will execute the `run.sh` script, which contains important configurations for paths and environment variables. **You need to adjust these paths based on your local setup**, including:

- The path for your **Hugging Face cache directory (`HF_HOME`)**. Make sure it points to your desired location to avoid storage issues.

- The **dataset path**, **model checkpoint paths**, and **output directories** specified in the script should be verified and modified if necessary.

For more details and reference commands, you can also check the **Open-R1 repository**, as it provides a comprehensive set of running scripts and configurations.

If you want to evaluation the model, replce the run.sh into eval.sh in the  `main.py`
