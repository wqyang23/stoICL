import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


model_name = "llama-3.2-3b"
model_path = f"./LLMs/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        # 尝试使用 GPU 1
        device = torch.device("cuda:2")
        # 检查是否能成功使用 GPU 1
        torch.cuda.get_device_name(2)  # 如果 GPU 1 不可用，可能会抛出 RuntimeError
    except RuntimeError:
        # 如果 GPU 1 不可用，选择 GPU 2
        device = torch.device("cuda:3")
else:
    # 如果没有 GPU，使用 CPU
    device = torch.device("cpu")

model.to(device)

data_list = ['hotel']

results = []
prediction_data = []

label0_shots_list = [2, 5, 10, 20, 30, 40]
default_other_shots_list = [2, 5, 10, 20, 30, 40]


for dataset_name in data_list:
    print(f"Processing dataset: {dataset_name}")
    dataset = load_from_disk(f'./datasets/ICL_classification/{dataset_name}')
    
    for default_other_shots in default_other_shots_list:
        for label0_shots in label0_shots_list:    
            print(f"\n==== Evaluating with label0_shots={label0_shots}, other_labels={default_other_shots} ====\n")

            columns = dataset['train'].features
            label_column_name = 'label' if 'label' in columns else 'labels'
            labels = sorted(set(dataset['train'][label_column_name]))  # 排序以确保第一个是label0
            label0 = labels[0]

            # Step 1: 收集每个标签对应数量的样本
            label_to_examples = defaultdict(list)
            for example in dataset['train']:
                label = example[label_column_name]
                limit = label0_shots if label == label0 else default_other_shots
                if len(label_to_examples[label]) < limit:
                    label_to_examples[label].append(example)

            # Step 2: 构建 context（按标签顺序）
            context = ""
            for label in sorted(label_to_examples.keys()):
                for example in label_to_examples[label]:
                    if 'text' in example:
                        text_field = 'text'
                    elif 'sentence' in example:
                        text_field = 'sentence'
                    elif 'sentence1' in example:
                        text_field = 'sentence1'
                    else:
                        raise KeyError("No valid text field found in example.")

                    context += f"Text: {example[text_field]}\nLabel: {label}\n"
    
            print(context)

            context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)

            test_data = dataset['validation'] if 'validation' in dataset else dataset['test']
            predictions, true_labels = [], []

            for test_example in tqdm(test_data, desc=f"Testing (label0={label0_shots})"):
                if 'text' in test_example:
                    text_field = 'text'
                elif 'sentence' in test_example:
                    text_field = 'sentence'
                elif 'sentence1' in test_example:
                    text_field = 'sentence1'
                else:
                    raise KeyError("No valid text field found in test example.")

                input_text = context + f"Text: {test_example[text_field]}\nLabel: "
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

                with torch.no_grad():
                    outputs = model(input_ids)

                logits = outputs.logits[:, -1, :]
                predicted_token = torch.argmax(logits, dim=-1)
                try:
                    predicted_label = int(tokenizer.decode(predicted_token.item()).strip())
                except ValueError:
                    predicted_label = -1

                predictions.append(predicted_label)
                true_labels.append(test_example[label_column_name])

            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)

            result = {
                'dataset': dataset_name,
                'label0_shots': label0_shots,
                'other_shots': default_other_shots,
                'accuracy': accuracy,
                'precision': precision,
                'true_labels': true_labels,
                'predictions': predictions,
                'context': context,
            }

            results.append(result)
            print(f"Finished {dataset_name} with label0={label0_shots} - Accuracy: {accuracy:.4f}")
            print(f"Results for {dataset_name} saved temporarily")

        directory = "./results_ICL_classification/"
        # datasets_name = '_'.join(data_list)
        filename = f"{directory}{model_name}_ICL_classification_on_{dataset_name}_2classes_unequal_square.csv"

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

print(f"All results saved to classification_results.csv")
