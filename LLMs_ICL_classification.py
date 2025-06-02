import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


model_name = "llama-3.2-1b"
model_path = f"./LLMs/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        # 尝试使用 GPU 1
        device = torch.device("cuda:1")
        # 检查是否能成功使用 GPU 1
        torch.cuda.get_device_name(1)  # 如果 GPU 1 不可用，可能会抛出 RuntimeError
    except RuntimeError:
        # 如果 GPU 1 不可用，选择 GPU 2
        device = torch.device("cuda:2")
else:
    # 如果没有 GPU，使用 CPU
    device = torch.device("cpu")

model.to(device)


data_list = ['agnews', 'sst2', 'sst5', 'subj', 'hotel']

results = []
prediction_data = []


for dataset_name in data_list:
    print(f"Processing dataset: {dataset_name}")
    dataset = load_from_disk(f'./datasets/{dataset_name}')

    print(f"Fields for dataset {dataset_name}: {dataset['train'].features}")

    columns = dataset['train'].features

    label_column_name = None
    if 'label' in columns:
        label_column_name = 'label'
    elif 'labels' in columns:
        label_column_name = 'labels'

    label_column = dataset['train'].features[label_column_name]
    labels = set(dataset['train'][label_column_name])
    num_classes = len(labels)
    label_to_data = defaultdict(list)
    label_count = defaultdict(int)

    for example in tqdm(dataset['train'], desc="Selecting training data"):
        label = example[label_column_name]
        if label_count[label] < 10:
            label_to_data[label].append(example)
            label_count[label] += 1

    context = ""
    recent_labels = []

    for example in dataset['train']:
        label = example[label_column_name]

        if label_to_data[label]:
            text_field = None
            if 'text' in example:
                text_field = 'text'
            elif 'sentence' in example:
                text_field = 'sentence'
            elif 'sentence1' in example:
                text_field = 'sentence1'

            if text_field:
                if recent_labels.count(label) < 3:
                    context += f"Text: {example[text_field]}\nLabel: {example[label_column_name]}\n"
                    recent_labels.append(label)
                    label_to_data[label].pop(0)
                else:
                    continue
            else:
                raise KeyError("No valid text field found in example.")

    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)

    if 'validation' in dataset:
        test_data = dataset['validation']
    else:
        test_data = dataset['test']

    predictions = []
    true_labels = []

    for test_example in tqdm(test_data, desc="Processing test data"):
        text_field = None
        if 'text' in test_example:
            text_field = 'text'
        elif 'sentence' in test_example:
            text_field = 'sentence'
        elif 'sentence1' in test_example:
            text_field = 'sentence1'

        if text_field:
            input_text = context + f"Text: {test_example[text_field]}\nLabel: "
        else:
            raise KeyError("No valid text field found in test example.")

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits[:, -1, :]
        predicted_token = torch.argmax(logits, dim=-1)

        predicted_label = int(tokenizer.decode(predicted_token.item()))

        predictions.append(predicted_label)
        true_labels.append(test_example[label_column_name])

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')

    print(f"Results for {dataset_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    result = {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'true_labels': true_labels,
        'predictions': predictions,
        'context': context,
    }

    results.append(result)

    print(f"Results for {dataset_name} saved temporarily")

directory = "./results/"
datasets_name = '_'.join(data_list)
filename = f"{directory}{model_name}_{datasets_name}_classification_results.csv"

df = pd.DataFrame(results)
df.to_csv(filename, index=False)

print(f"All results saved to classification_results.csv")
