import torch
from typing import Union
import numpy as np
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from faq_model.io import save_augmented_corpus
from faq_model.sentence_encoder import SentenceEncoder

TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_paraphrase_model():
    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(TORCH_DEVICE)
    return tokenizer, model


def generate_paraphrases(input_text, tokenizer, model, num_return_sequences=50, num_beams=50):
    '''
    Generate paraphrases on input_text
    tokenizer, model: paraphrasing model to initiate externally
    num_return_sequences: the number of paraphrases to generate
    num_beams: the number of beams used in beam search
    # Example
    question = 'How do I get a new quote for insurance?'
    generate_paraphrases(question, num_return_sequences, num_beams)
    '''
    batch = tokenizer(
        [input_text],
        truncation=True,
        padding='longest',
        max_length=60,
        return_tensors="pt"
    ).to(TORCH_DEVICE)

    translated = model.generate(
        **batch,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=1.5
    )

    paraphrases = tokenizer.batch_decode(
        translated,
        skip_special_tokens=True
    )
    return paraphrases


def corpus_paraphrases(
        corpus: Union[list, pd.Series],
        sent_encoder: SentenceEncoder,
        num_return_sequences: int = 50
        ):
    '''
    Augmenting corpus by generating paraphrases for corpus
    '''
    tokenizer, model = get_paraphrase_model()
    augmented_corpus = dict()
    augmented_corpus_encoded = np.ndarray(
        (
            len(corpus),
            num_return_sequences,
            sent_encoder.encoder.get_sentence_embedding_dimension()
        ),
        dtype=np.float32
    )

    for idx, question in enumerate(corpus):
        paraphrases = generate_paraphrases(
            question,
            tokenizer, model,
            num_return_sequences
        )
        augmented_corpus[question] = paraphrases
        paraphrases = sent_encoder(paraphrases)
        augmented_corpus_encoded[idx] = paraphrases

    augmented_corpus = pd.DataFrame.from_dict(
        augmented_corpus,
        orient='index'
    )
    return augmented_corpus, augmented_corpus_encoded


def generate_augmented_data(
        question_list: Union[list, pd.Series],
        sent_encoder: SentenceEncoder,
        output_filename: str,
        num_return_sequences: int = 50):
    '''
    Generate augmented data and save to a pickle file
    '''
    augmented_corpus, augmented_corpus_encoded = corpus_paraphrases(
        question_list,
        sent_encoder,
        num_return_sequences=num_return_sequences
    )
    save_augmented_corpus([augmented_corpus, augmented_corpus_encoded], output_filename)
