import os
from pathlib import Path
import pandas as pd
import torch
from pymongo import MongoClient
from datasets import Dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import EvaluationStrategy, set_seed
from peft import LoraConfig, get_peft_model, TaskType


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# --- FUNCIÓN DE TOKENIZACIÓN AJUSTADA ---

def tokenize_function(example, tokenizer, max_length=512):
    # tokenizamos prompt y response por separado (devuelven listas de ints)
    p_ids = tokenizer(
        example["prompt"],
        truncation=True,
        padding=False,
        add_special_tokens=False
    )["input_ids"]
    r_ids = tokenizer(
        example["response"],
        truncation=True,
        padding=False,
        add_special_tokens=False
    )["input_ids"]

    # construimos secuencia [prompt..., EOS, response..., EOS]
    ids = p_ids + [tokenizer.eos_token_id] + r_ids + [tokenizer.eos_token_id]
    # generamos labels enmascarando el prompt
    labels = [-100] * (len(p_ids) + 1) + r_ids + [tokenizer.eos_token_id]

    # recortamos si excede max_length
    if len(ids) > max_length:
        ids = ids[:max_length]
        labels = labels[:max_length]

    # máscara de atención y padding
    attention_mask = [1] * len(ids)
    pad_len = max_length - len(ids)
    ids += [tokenizer.pad_token_id] * pad_len
    labels += [-100] * pad_len
    attention_mask += [0] * pad_len

    return {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }




def load_data_from_mongodb():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["datasets"]
    collection = db["smoltest"]
    docs = list(collection.find({}))
    for doc in docs:
        doc.pop("_id", None)
    df = pd.DataFrame(docs)

    if df.empty:
        print("Alerta: Se han cargado datos vacíos.")
    if df.isnull().values.any():
        print("Advertencia: Existen valores nulos en los datos.")
    if df.astype(str).duplicated().sum() > 0:
        print(f"Advertencia: Se detectaron {df.astype(str).duplicated().sum()} filas duplicadas.")
    return df


def split_train_eval(df, train_ratio=0.8):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    eval_df = df[split_idx:]

    if train_df.empty:
        print("Error: El conjunto de datos de entrenamiento está vacío.")
    if eval_df.empty:
        print("Error: El conjunto de datos de evaluación está vacío.")

    if "label" in df.columns:
        print("Distribución de etiquetas en entrenamiento:")
        print(train_df["label"].value_counts())
        print("Distribución de etiquetas en evaluación:")
        print(eval_df["label"].value_counts())

    return train_df, eval_df


def process_data_batch(tokenizer):
    # 1. Carga y chequeos básicos
    df = load_data_from_mongodb()
    print("Longitudes de prompt (DATA):", df["prompt"].str.len().describe())
    print("Longitudes de response (DATA):", df["response"].str.len().describe())

    # 2. Split train/eval
    train_df, _ = split_train_eval(df, train_ratio=0.8)

    # 3. Cache path
    cache_path = "./cache_tok_train"
    if os.path.isdir(cache_path):
        # Carga ya tokenizado
        return Dataset.load_from_disk(cache_path)

    # 4. Tokenización ejemplo a ejemplo (batched=False para mantener tipos homogéneos)
    dataset = Dataset.from_pandas(train_df)
    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=False,
        remove_columns=dataset.column_names
    )

    # 5. Guarda el cache
    tokenized.save_to_disk(cache_path)
    return tokenized


def process_eval_batch(tokenizer):
    # 1. Carga y chequeos básicos
    df = load_data_from_mongodb()
    print("Longitudes de prompt (EVAL):", df["prompt"].str.len().describe())
    print("Longitudes de response (EVAL):", df["response"].str.len().describe())

    # 2. Split train/eval
    _, eval_df = split_train_eval(df, train_ratio=0.8)

    # 3. Cache path
    cache_path = "./cache_tok_eval"
    if os.path.isdir(cache_path):
        return Dataset.load_from_disk(cache_path)

    # 4. Tokenización elemento a elemento
    dataset = Dataset.from_pandas(eval_df)
    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=False,
        remove_columns=dataset.column_names
    )

    # 5. Guarda el cache
    tokenized.save_to_disk(cache_path)
    return tokenized


def main():
    set_seed(42)
    output_dir = "./finetuned"
    os.makedirs(output_dir, exist_ok=True)

    transformers.logging.set_verbosity_debug()
    model_dir = Path.home() / "eleutherai_models" / "gpt-neo-1.3B"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = get_device()

    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=model_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # --- PEFT LoRA Setup ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        # Incluimos todas las proyecciones de atención para habilitar gradientes
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    # Verificar que hay parámetros entrenables
    model.print_trainable_parameters()
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_count}")

    model.to(device)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        seed=42,
        learning_rate=5e-6,
        num_train_epochs=4,
        warmup_ratio=0.1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        weight_decay=0.0,
        output_dir=output_dir,
        eval_strategy=EvaluationStrategy.EPOCH,
        logging_dir="./logs",
        report_to=None,  # Desactivar TensorBoard o configurar tras instalar
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
    )

    train_dataset = process_data_batch(tokenizer)
    eval_dataset = process_eval_batch(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)

    eval_metrics = trainer.evaluate()
    print(f"Eval loss: {eval_metrics['eval_loss']:.3f}")

    # Prueba de generación
    input_text = "¿Qué vecinos tiene el nodo 21?"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    output_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=30,
        num_beams=5,
        temperature=0.2,
        top_p=0.9,
        top_k=50,
        early_stopping=True,
    )
    print("Salida generada:", tokenizer.decode(output_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
