import re
import string
import bisect

# ==========================================
# Text Splitter Module (NLP Tokenization)
# ==========================================
class BaseSegmentSplitter:
    def get_spans(self, text: str):
        """Return token spans and start offsets as (spans, starts)."""
        raise NotImplementedError

class RegexSegmentSplitter(BaseSegmentSplitter):
    """Original regex-based tokenizer for whitespace and selected separators."""
    def __init__(self):
        ABBREV_RE = r"(?:[A-Za-z]{1,4}\.){2,}(?:[A-Za-z]{1,4}\.?)?"
        CJK_RE = r"[\u4E00-\u9FFF\u3400-\u4DBF]+"
        EN_WORD_RE = r"[A-Za-z0-9]+(?:[-_'\u2019\u2018\u02BC][A-Za-z0-9]+)*"
        self.word_re = re.compile(rf"{ABBREV_RE}|{CJK_RE}|{EN_WORD_RE}")

    def get_spans(self, text: str):
        matches = list(self.word_re.finditer(text))
        spans = [(m.start(), m.end()) for m in matches]
        starts = [s for s, _ in spans]
        return spans, starts

class JiebaSegmentSplitter(BaseSegmentSplitter):
    """Jieba-based tokenizer, suitable for Chinese and mixed-language text."""
    def __init__(self):
        try:
            import jieba
            self.jieba = jieba
        except ImportError:
            raise ImportError("Please install jieba: pip install jieba")

    def get_spans(self, text: str):
        spans = []
        starts = []
        # tokenize returns (word, start, end).
        for word, start, end in self.jieba.tokenize(text):
            if word.strip():  # Skip pure whitespace.
                spans.append((start, end))
                starts.append(start)
        return spans, starts

class NltkSegmentSplitter(BaseSegmentSplitter):
    """Standard English tokenizer based on NLTK."""
    def __init__(self):
        try:
            from nltk.tokenize import TreebankWordTokenizer
            self.tokenizer = TreebankWordTokenizer()
        except ImportError:
            raise ImportError("Please install nltk: pip install nltk")

    def get_spans(self, text: str):
        # span_tokenize directly returns [(start, end), ...].
        spans = list(self.tokenizer.span_tokenize(text))
        starts = [s for s, e in spans]
        return spans, starts

class SpacySegmentSplitter(BaseSegmentSplitter):
    """Production-oriented NLP tokenizer based on spaCy."""
    def __init__(self, model="en_core_web_sm"):
        try:
            import spacy
            try:
                self.nlp = spacy.load(model)
            except OSError:
                # Fallback to a lightweight tokenizer-only pipeline when the
                # full model package is missing in runtime environments.
                self.nlp = spacy.blank("en")
                print(
                    f"[Warn] spaCy model '{model}' not found. "
                    "Falling back to spacy.blank('en') tokenizer. "
                    f"Install model for full pipeline: python -m spacy download {model}"
                )
        except ImportError:
            raise ImportError("Please install spacy: pip install spacy")

    def get_spans(self, text: str):
        doc = self.nlp(text)
        spans = [(token.idx, token.idx + len(token)) for token in doc if not token.is_space]
        starts = [s for s, e in spans]
        return spans, starts

# ==========================================
