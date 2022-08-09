from typing import Union
from sentence_transformers import SentenceTransformer, util

MODEL_CARD = 'multi-qa-mpnet-base-dot-v1'


class SentenceEncoder(object):
    def __init__(self, model_card=MODEL_CARD):
        self.encoder = SentenceTransformer(model_card)

    def __call__(self, sent):
        return self.encode_sentence(sent)

    def encode_sentence(
            self,
            sentence: Union[str, list],
            normalize_sentence: bool = True,
            normalize_embeddings: bool = True,
            convert_to_tensor: bool = True,
            show_progress_bar: bool = False):
        if isinstance(sentence, str):
            sentence = [sentence]
        if normalize_sentence:
            sentence = [*map(preprocess_sentence, sentence)]
        return self.encoder.encode(
            sentence,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar)


def preprocess_sentence(sent):
    norm_sent = str(sent).lower().strip().strip(string.punctuation).strip().capitalize()
    norm_sent = replace_unicode_quote(norm_sent)
    return norm_sent


def dot_score(query_embedding, corpus_embeddings):
    return util.dot_score(query_embedding, corpus_embeddings)
