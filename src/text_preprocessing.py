import re
import nltk
import string
import gensim
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

stop_words = set(stopwords.words('english') + \
                 list(string.punctuation) + \
                 ['\\n'] + ['quot'])

regex_str = ["http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|"
             "[!*\(\),](?:%[0-9a-f][0-9a-f]))+",
             "(?:\w+-\w+){2}",
             "(?:\w+-\w+)",
             "(?:\\\+n+)",
             "(?:@[\w_]+)",
             "<[^>]+>",
             "(?:\w+'\w)",
             "(?:[\w_]+)",
             "(?:\S)"]

# Create the tokenizer which will be case insensitive and will ignore space.
tokens_re = re.compile(r'('+'|'.join(regex_str)+')',
                       re.VERBOSE | re.IGNORECASE)


def tokenize_document(text):
    """Preprocess a whole raw document.

    Args:
        text (str): Raw string of text.

    Return:
        List of preprocessed and tokenized documents
    """

    return [clean_and_tokenize(sentence)
            for sentence in nltk.sent_tokenize(text)]


def clean_and_tokenize(text):
    """Preprocess a raw string/sentence of text.

    Args:
       text (str): Raw string of text.

    Return:
       tokens (list, str): Preprocessed tokens.
    """

    tokens = tokens_re.findall(text)
    _tokens = [t.lower() for t in tokens]
    filtered_tokens = [token.replace('-', '_') for token in _tokens
                       if len(token) > 2
                       and token not in stop_words
                       and not any(x in token for x in string.digits)
                       and any(x in token for x in string.ascii_lowercase)]
    return filtered_tokens

def build_ngrams(documents, n=2, **kwargs):
    """Create ngrams using Gensim's phrases.

    Args:
        documents (:obj:`list` of token lists): List of preprocessed and
                                                tokenized documents
        n (int): The `n` in n-gram.
    """
    # Check whether "level" was passed as an argument
    if "level" not in kwargs:
        level = 2
    else:
        level = kwargs["level"]
    # Generate sentences, as required for gensim Phrases
    sentences = []
    for doc in documents:
        sentences += doc
    # Get the bigrams
    phrases = gensim.models.Phrases(sentences, min_count=2, delimiter=b'_')
    bigram = gensim.models.phrases.Phraser(phrases)
    docs_bi = [[bigram[sentence] for sentence in doc] for doc in documents]
    # If finished
    if level == n:
        return docs_bi
    # Otherwise, keep processing until n-grams satisfied
    return build_ngrams(docs_bi, n=n, level=level+1)

def keras_tokenizer(texts, num_words=None, mode=None, maxlen=None, sequences=False):
    """Preprocess text with Keras Tokenizer.

    Args:
        texts (list, str): Collections of documents to preprocess.
        num_words (int): Length of the vocabulary.
        mode (str): Can be "count" or "tfidf".

    Returns:
        encoded_docs (array, int | float): Can be Bag-of-Words or TF-IDF weight matrix, depending
                                            on "mode".

    """
    t = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ', num_words=num_words)
    t.fit_on_texts(texts)

    if sequences:
        seq = t.texts_to_sequences(texts)
        encoded_docs = pad_sequences(seq, maxlen=maxlen)
    else:
        encoded_docs = t.texts_to_matrix(texts, mode=mode)

    print('Documents count: {}'.format(t.document_count))
    print('Found %s unique tokens.' % len(t.word_index))
    print('Shape of encoded docs: {}'.format(encoded_docs.shape))

    return encoded_docs
