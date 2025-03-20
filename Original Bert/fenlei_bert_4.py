import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertModel, BertTokenizer
import warnings

warnings.filterwarnings("ignore")

# 设置 Hugging Face 缓存路径
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, targets):
        self.sentences = sentences
        self.targets = targets

    def __getitem__(self, idx):
        return self.sentences[idx], self.targets[idx]

    def __len__(self):
        return len(self.sentences)

class BertClassification(nn.Module):
    def __init__(self, tokenizer, device, dropout1=0.2, dropout2=0.5):
        super(BertClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir="/tmp/huggingface_cache")
        self.tokenizer = tokenizer
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(),
            nn.Dropout(dropout1), nn.Linear(256, 16), nn.GELU(),
            nn.Dropout(dropout2), nn.Linear(16, 2)
        )

    def forward(self, x):
        batch_tokenized = self.tokenizer.batch_encode_plus(
            x, add_special_tokens=True, max_length=148,
            padding='max_length', truncation=True, return_tensors="pt"
        )
        input_ids = batch_tokenized['input_ids'].to(self.device)
        attention_mask = batch_tokenized['attention_mask'].to(self.device)
        hidden_states = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        outputs = self.fc(hidden_states[:, 0, :])
        return outputs

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    folder_path_y = '/tmp/datasets/chapt_y'
    folder_path_n = '/tmp/datasets/chapt_n'
    file_list_y = os.listdir(folder_path_y)
    file_list_n = os.listdir(folder_path_n)

    datas, labels = [], []

    # 处理正样本数据，排除第49到第64章
    for index, file_name in enumerate(file_list_y):
        if 48 <= index < 64:  # 排除范围
            continue
        file_path = os.path.join(folder_path_y, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            positive_data = [line for line in content.split('\n') if line.strip()]
            datas.extend(positive_data)
            labels.extend([1] * len(positive_data))  # 确保标签数量与数据一致

    # 处理负样本数据，排除第105到第112章
    for index, file_name in enumerate(file_list_n):
        if 104 <= index < 112:  # 排除范围
            continue
        file_path = os.path.join(folder_path_n, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            negative_data = [line for line in content.split('\n') if line.strip()]
            datas.extend(negative_data)
            labels.extend([0] * len(negative_data))  # 确保标签数量与数据一致

    # 检查数据和标签数量
    print(f"数据数量: {len(datas)}, 标签数量: {len(labels)}")
    assert len(datas) == len(labels), "数据和标签数量不一致！"

    # 数据划分
    train_data, val_data, train_labels, val_labels = train_test_split(
        datas, labels, test_size=0.2, random_state=42
    )

    train_dataset = MyDataset(train_data, train_labels)
    val_dataset = MyDataset(val_data, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 初始化 Tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="/tmp/huggingface_cache")
    model = BertClassification(tokenizer=tokenizer, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # 训练模型
    best_f1 = 0.0
    save_directory = '/tmp/hongloumeng/originalBert/hf_compatible_fenleiFTed_4'

    for epoch in range(11):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_dataloader:
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证集评估
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, targets).item()
                preds = outputs.argmax(dim=1).tolist()
                all_preds.extend(preds)
                all_labels.extend(targets.tolist())

        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            # 保存 Bert 模型、Tokenizer 和分类头
            model.bert.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            torch.save(model.fc.state_dict(), os.path.join(save_directory, "classification_head.bin"))
            print(f"模型已保存到 {save_directory}")

# 执行训练
if __name__ == '__main__':
    train_model()
