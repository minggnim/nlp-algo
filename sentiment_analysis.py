from typing import List
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
LABELS = ['negative', 'neutral', 'positive']

@dataclass(frozen=True)
class Prediction:
    proba: float
    label: str

class Sentiment(object):
    def __init__(self, model: str = MODEL, labels: List[str] = LABELS):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.labels = labels
        
    def __call__(self, text: str) -> Prediction:
        return self.sentiment(text)

    def sentiment(self, text: str) -> Prediction:
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        outputs = self.model(**inputs).logits
        label_id = outputs.argmax().item()
        probas = softmax(outputs, dim=-1).detach().numpy()[0]
        return Prediction(
            label=self.labels[label_id],
            proba=probas[label_id]
        )


def conversation_sentiment(senti):
    if senti.positive and senti.negative:
        return 'mixed'
    elif senti.positive:
        return 'positive'
    elif senti.negative:
        return 'negative'
    else:
        return 'neutral'
