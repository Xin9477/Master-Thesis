import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

warnings.filterwarnings("ignore")

# 设置 Hugging Face 缓存路径
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"

class MyDataset(Dataset):
    def __init__(self, sentences, targets):
        self._X = sentences
        self._y = targets

    def __getitem__(self, index):
        return self._X[index], self._y[index]

    def __len__(self):
        return len(self._X)

if __name__ == '__main__':
    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载微调后的模型和 tokenizer
    model_path = '/tmp/hongloumeng/hf_compatible_fenleiFTed_5'
    tokenizer_path = '/tmp/hongloumeng/hf_compatible_fenleiFTed_5'

    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    model.eval()
    print("微调后的完整模型已成功加载！")

    # 数据路径
    folder_path_y = '/tmp/datasets/chapt_y'
    folder_path_n = '/tmp/datasets/chapt_n'
    file_list_y = sorted(os.listdir(folder_path_y), key=lambda x: int(''.join(filter(str.isdigit, x))))
    file_list_n = sorted(os.listdir(folder_path_n), key=lambda x: int(''.join(filter(str.isdigit, x))))

    # 验证文件列表长度
    print("Length of file_list_y:", len(file_list_y))
    print("Length of file_list_n:", len(file_list_n))

    # 测试集范围
    y_test_range = file_list_y[64:80]  # Y-dataset 第64到80章
    n_test_range = file_list_n[32:40]  # N-dataset 第113到120章

    # 验证切片是否正确
    print("Y test range files:", y_test_range)
    print("N test range files:", n_test_range)

    # 输出文件路径
    performance_file = '/tmp/hongloumeng/trained_bert_performance5.txt'
    percentage_file = '/tmp/hongloumeng/trained_bert_percentage5.txt'

    with open(performance_file, 'w') as perf_f, open(percentage_file, 'w') as perc_f:
        datasets = [
            (y_test_range, folder_path_y, 65),  # Y-dataset 第65到80章
            (n_test_range, folder_path_n, 113)  # N-dataset 第113到120章
        ]

        for file_list, folder_path, start_index in datasets:
            for index, file_name in enumerate(file_list):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    data = [line for line in content.split('\n') if line.strip()]
                    labels = [1 if start_index < 81 else 0] * len(data)  # Y-dataset 为1，N-dataset 为0
                    test_dataset = MyDataset(data, labels)
                    test_dataload = DataLoader(test_dataset, shuffle=False, batch_size=16)

                    acc_test, pre_test, rec_test, f1_test = 0.0, 0.0, 0.0, 0.0
                    predicted = []

                    for inputs, labels in test_dataload:
                        inputs = list(inputs)
                        labels = labels.tolist()
                        batch_tokenized = tokenizer.batch_encode_plus(
                            inputs,
                            add_special_tokens=True,
                            max_length=148,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt"
                        )
                        input_ids = batch_tokenized['input_ids'].to(device)
                        attention_mask = batch_tokenized['attention_mask'].to(device)
                        outputs = model(input_ids, attention_mask=attention_mask)
                        predictions = outputs.logits.argmax(dim=1).tolist()
                        predicted += predictions
                        acc = accuracy_score(labels, predictions)
                        pre = precision_score(labels, predictions, average='macro')
                        rec = recall_score(labels, predictions, average='macro')
                        f1 = f1_score(labels, predictions, average='macro')
                        acc_test += acc
                        pre_test += pre
                        rec_test += rec
                        f1_test += f1

                    total_sentences = len(predicted)
                    positive_percentage = (predicted.count(1) / total_sentences) * 100
                    negative_percentage = (predicted.count(0) / total_sentences) * 100
                    acc_test /= len(test_dataload)
                    pre_test /= len(test_dataload)
                    rec_test /= len(test_dataload)
                    f1_test /= len(test_dataload)

                    chapter_number = start_index + index
                    perf_f.write(f'第{chapter_number}章: acc_test={acc_test:.5f}, pre_test={pre_test:.5f}, rec_test={rec_test:.5f}, f1_test={f1_test:.5f}\n')
                    perc_f.write(f'第{chapter_number}章: Positive={positive_percentage:.2f}%, Negative={negative_percentage:.2f}%\n')

    print(f"模型表现记录已保存到 {performance_file}")
    print(f"句子标签百分比记录已保存到 {percentage_file}")
