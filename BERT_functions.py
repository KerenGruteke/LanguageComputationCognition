import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class SentenceRepresentation(nn.Module):
    def __init__(self):
        super(SentenceRepresentation, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 300)  # Adjust the input and output size as needed

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sentence_representation = outputs.last_hidden_state[:, 0, :]
        sentence_representation = self.linear(sentence_representation)
        return sentence_representation


def extract_sentence_representation(sentences):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = SentenceRepresentation()
    representations = []

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([token_ids])
        attention_mask = torch.ones(input_ids.shape)

        with torch.no_grad():
            representation = model(input_ids, attention_mask)

        representations.append(representation)

    return torch.cat(representations, dim=0)
