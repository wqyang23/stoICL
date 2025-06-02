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

data_list = ['agnews', 'hotel', 'sst2', 'sst5', 'subj', 'trec6', 'twitterfn']

results = []
prediction_data = []

shots_list = [3, 6, 9, 12]

for dataset_name in data_list:
    print(f"Processing dataset: {dataset_name}")
    dataset = load_from_disk(f'./datasets/ICL_classification/{dataset_name}')

    for label_sample_limit in shots_list:
        print(f"\n==== Evaluating with {label_sample_limit} shots per label ====\n")

        print(f"Fields for dataset {dataset_name}: {dataset['train'].features}")

        columns = dataset['train'].features

        label_column_name = None
        if 'label' in columns:
            label_column_name = 'label'
        elif 'labels' in columns:
            label_column_name = 'labels'

        label_column = dataset['train'].features[label_column_name]
        labels = set(dataset['train'][label_column_name])
    # num_classes = len(labels)

        # Step 1: 收集每个标签的前 10 个样本（按出现顺序）
        label_to_examples = defaultdict(list)
        for example in dataset['train']:
            label = example[label_column_name]
            if len(label_to_examples[label]) < label_sample_limit:
                label_to_examples[label].append(example)

        # Step 2: 按标签排序后构建上下文
        context = ""
        for label in sorted(label_to_examples.keys(), reverse=True):
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
    # recall = recall_score(true_labels, predictions, average='weighted')

        print(f"Results for {dataset_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
    # print(f"  Recall: {recall:.4f}")

        result = {
            'number_each_label': label_sample_limit,
            'dataset': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            # 'recall': recall,
            'true_labels': true_labels,
            'predictions': predictions,
            'context': context,
        }

        results.append(result)

        directory = "./results_ICL_classification/"
        # datasets_name = '_'.join(data_list)
        # filename = f"{directory}{model_name}_ICL_classification_on_{datasets_name}_36912.csv"
        filename = f"{directory}{model_name}_ICL_classification_on_{dataset_name}_36912_reverse.csv"

        print(f"Results for {dataset_name} saved temporarily")

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

print(f"All results saved to classification_results.csv")
