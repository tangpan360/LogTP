from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn


class LogClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super(LogClassifier, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        # 定义一个简单的线性分类器
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 通过BERT模型获取输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS]标记的输出作为分类的基础
        cls_output = outputs.pooler_output
        # 通过分类器获取最终的分类结果
        logits = self.classifier(cls_output)
        return logits


# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = LogClassifier('bert-base-uncased')

# 假设的20条日志
logs = ["Error found in module 1", "System reboot scheduled", "User login successful" "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful", "User login successful"]  # 请根据实际情况填充

# 将日志拼接起来，每条日志之间使用[SEP]分隔
logs_string = " [SEP] ".join(logs)

logss = [logs_string, logs_string, logs_string, logs_string, logs_string, logs_string, logs_string]

# 对拼接后的日志字符串进行分词处理
inputs = tokenizer(logss, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 将处理后的输入送入模型进行分类
with torch.no_grad():
    logits = model(inputs.input_ids, inputs.attention_mask)

# 使用softmax函数计算每个分类的概率
probabilities = torch.softmax(logits, dim=-1)

print(probabilities)
