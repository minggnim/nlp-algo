from keybert import KeyBERT
from typing import Optional
from dataclasses import dataclass


@dataclass
class Keyword:
    keyword: str
    score: float
    

class ExtractKeywords(object):
    def __init__(self, model_card='distilbert-base-nli-mean-tokens'):
        self.kw_model = KeyBERT(model_card)
        
    def __call__(self, text):
        return self.extract_keywords_maxsum(text)
    
    def extract_keywords_maxsum(self, text: str, fill_empty_return: bool = False) -> Optional[Keyword]:
        bi_keywords = self.kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(2, 3), 
            stop_words='english',
            use_maxsum=True,
            nr_candidates=20,
            top_n=1,
        ) 
        uni_keywords = self.kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 1), 
            stop_words='english',
            use_maxsum=True,
            nr_candidates=20,
            top_n=1,
        )
        if bi_keywords:
            keywords = bi_keywords if bi_keywords[0][1] > uni_keywords[0][1] else uni_keywords
        else:
            keywords = uni_keywords
        # handle the full text is from stopwords
        if keywords:
            return Keyword(
                keyword=keywords[0][0],
                score=keywords[0][1]
            )
        elif fill_empty_return:
            return Keyword(keyword=text.lower(), score=0)
        else:
            return None   

    
    def extract_keywords_mmr(self, text: str, fill_empty_return: bool = False) -> Optional[Keyword]:
        keywords = self.kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(2, 3), 
            stop_words='english',
            use_mmr=True,
            diversity=0.2,
            top_n=1,
        )
        # handle the full text is from stopwords
        if keywords:
            return Keyword(
                keyword=keywords[0][0],
                score=keywords[0][1]
            )
        elif fill_empty_return:
            return Keyword(keyword=text.lower(), score=0)
        else:
            return None   