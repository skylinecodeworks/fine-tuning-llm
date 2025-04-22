# LLM FINETUNING HOW-TO!

This project provides a complete pipeline to fine-tune the EleutherAI GPT-Neo 1.3B causal language model on your own prompt–response dataset using parameter-efficient LoRA adapters. By leveraging data stored in MongoDB and Hugging Face's Transformers and PEFT libraries, you can achieve style and tone control without retraining the full model.

## Features

- **Custom data loading** from a MongoDB collection (`datasets.smoltest`).
- **Data validation**: checks for empty datasets, null values, and duplicates.
- **Train/eval split** with configurable ratio and label distribution reporting.
- **Efficient tokenization** with a custom function that masks prompts and appends EOS tokens.
- **Disk caching** of tokenized datasets to speed up repeated runs.
- **LoRA adapters** (PEFT) to fine-tune only a small subset of adapter weights.
- **Hugging Face Trainer** API for streamlined training, evaluation, and checkpointing.
- **Inference demo** at the end of training to verify generation quality.

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- PEFT (parameter-efficient fine-tuning)
- Datasets
- pandas
- pymongo

## Installation with Astral UV (https://docs.astral.sh/uv/)

1. Initialize and activate a UV environment (Astral UV should be installed)
   ```bash
   uv init venv
   uv activate venv
   ```

2. Install all project dependencies via UV:
   ```bash
   uv pip install torch transformers peft datasets pandas pymongo
   ```

3. Ensure MongoDB is running locally on `mongodb://localhost:27017/` before proceeding.

## Data Preparation

1. Insert your prompt–response JSON documents into MongoDB. Each document should have the keys:
   - `prompt` (string)
   - `response` (string)

2. The script will load the data, remove the `_id` field, and convert it into a pandas DataFrame.
3. Basic checks for empty or duplicate entries will be printed to the console.

## How It Works

1. **Tokenization**: `tokenize_function` builds input sequences as:
   ```
   [ prompt tokens..., EOS, response tokens..., EOS ]
   ```
   and masks labels for the prompt portion to focus loss on the response.

2. **Caching**: Tokenized datasets are saved to `./cache_tok_train` and `./cache_tok_eval` for reuse.

3. **LoRA Setup**: A `LoraConfig` targets the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with:
   - rank `r = 4`
   - `lora_alpha = 16`
   - dropout `0.1`

4. **Training**: Uses `TrainingArguments` with:
   - `learning_rate = 5e-6`
   - `num_train_epochs = 4`
   - gradient accumulation to fit larger effective batch sizes
   - `EvaluationStrategy.EPOCH` for evaluation at each epoch

5. **Evaluation & Checkpointing**: The best model is loaded at the end, and final evaluation loss is printed.

6. **Inference**: A sample query runs at the end to demonstrate generation:
   ```python
   input_text = "{{here_goes_the_question}}"
   model.generate(...)
   ```

## Usage

Run the training script:
```bash
source .venv/bin/activate
uv run fine_tune.py
```

Monitor logs in `./logs`. The fine‑tuned model and tokenizer will be saved under `./finetuned`.

## Configuration

- **MongoDB URI**: Modify the connection string in `load_data_from_mongodb()` if needed.
- **Model directory**: Update `model_dir` to point to your local GPT-Neo weights.
- **Batch sizes, epochs, learning rate**: Adjust in the `TrainingArguments` block.
- **LoRA parameters**: Tweak `r`, `lora_alpha`, and `dropout` in the `LoraConfig`.

## Project Structure

```
├── fine-tuned-training.py          # Main training script (this file)
├── get_model.py                    # Download the model from huggingface
├── cache_tok_train/                # Cached tokenized training set
├── cache_tok_eval/                 # Cached tokenized evaluation set
├── logs/                           # Training logs
└── finetuned/                      # Output directory for model checkpoints
```

*Created with love by Tomás Pascual*
