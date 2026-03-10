import re
import html
import unicodedata
from typing import List, Optional
from collections import Counter

pattern_http = re.compile(r'https?://\S+|www\.\S+')
pattern_html = re.compile(r'<[^>]+>')

def clean_refs(text: str) -> str:

    text = html.unescape(text)
    text = pattern_html.sub('', text)
    text = pattern_http.sub('', text)

    return text

RE_TOKEN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|\d+",
    flags=re.IGNORECASE
)

def tokenization(text:str, word_counts: Optional[Counter] = None) -> List[str]:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"\s#39;[sS]\b", "'s", text)
    text = text.lower()
    text = clean_refs(text)
    tokens = RE_TOKEN.findall(text)

    if word_counts is not None:
        word_counts.update(tokens)

    return tokens

def encode_text(text:str, word2id) -> List[int]:
    tokens = tokenization(text)

    tokenized_texts = []
    for token in tokens:
        if token in word2id:
            tokenized_texts.append(word2id[token])

    return tokenized_texts