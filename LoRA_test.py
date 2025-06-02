import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from tqdm import tqdm


# 1. 加载模型和分词器
model_name = "llama-3.2-1b"
model_path = f"./LLMs/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        # 尝试使用 GPU 1
        device = torch.device("cuda:0")
        # 检查是否能成功使用 GPU 1
        torch.cuda.get_device_name(0)  # 如果 GPU 1 不可用，可能会抛出 RuntimeError
    except RuntimeError:
        # 如果 GPU 1 不可用，选择 GPU 2
        device = torch.device("cuda:1")
else:
    # 如果没有 GPU，使用 CPU
    device = torch.device("cpu")

model.to(device)

# 2. 加载数据集
data_list = ['agnews']
# dataset = load_dataset("ag_news")
for dataset_name in data_list:
    print(f"Processing dataset: {dataset_name}")
    dataset = load_from_disk(f'./datasets/{dataset_name}')
    train_data = dataset["train"].shuffle(seed=42).select(range(1000))  # 选一部分微调样本
    # test_data = dataset["test"].select(range(100))  # 验证集

    # 3. 配置LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # 具体模块需根据实际模型结构确认
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    # 4. 数据预处理
    def preprocess(example):
        return {
            "text": f"Classify this news: {example['text']} Category: {example['label']}"
        }

    train_data = train_data.map(preprocess)
    # test_data = test_data.map(preprocess)

    # 5. LoRA微调参数
    training_args = TrainingArguments(
        output_dir="./lora_llama3_agnews",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none"
    )

    # 6. SFTTrainer 进行训练
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=test_data, 可选
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text"
    )

    trainer.train()

    # 7. 保存LoRA微调后的模型
    directory = "./finetuned_models_lora/"
    # datasets_name = '_'.join(data_list)
    directory_name = f"{directory}{model_name}_lora_{datasets_name}/"
    model.save_pretrained(directory_name)
    tokenizer.save_pretrained(directory_name)

# model.save_pretrained("./lora_llama3_agnews/peft")
# tokenizer.save_pretrained("./lora_llama3_agnews/peft")
print(f"11111111111111111111111111111111111")