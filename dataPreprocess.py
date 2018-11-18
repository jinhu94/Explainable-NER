import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

"""
For the new data file, build word dict and tag dict for it and save to pickle
"""

def load_data_and_labels(filename):
    """
    Loads data and label from a file.
    Args:
        filename (str): path to the file.
        The file format is tab-separated values.
        A blank line is required at the end of a sentence.
        For example:
        ```
        EU	B-ORG
        rejects	O
        German	B-MISC
        call	O
        to	O
        boycott	O
        British	B-MISC
        lamb	O
        .	O
        Peter	B-PER
        Blackburn	I-PER
        ...
        ```
    Returns:
        tuple(numpy array, numpy array): data and labels.
    Example:
        >>> filename = 'conll2003/en/ner/train.txt'
        >>> data, labels = load_data_and_labels(filename)
    """
    sents, labels = [], []
    with open(filename) as f:
        words, tags = [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                word, tag = line.split('\t')
                words.append(word.lower())
                tags.append(tag)
    return np.asarray(sents), np.asarray(labels)

def get_words_and_tags_set(x_seq, y_seq):
    """
    Get all distinct words and tags of data
    Args:
        x_seq: an array of sentences consists of words, output from load_data_and_labels
        y_seq: an array of respond sequence of tags to words in x_seq, output from load_data_and_labels
        for example:
            ```
            x_seq: np.array(['word1','word2', ..., 'word5'], ['w1','w2', ..., 'w9'], ...)
            y_seq: np.array(['tag1','tag2',...,'tag5'],['t1','t2',...,'t5'], ...)
            ```
    Returns: list of distinct words and tags
    Example: words_set, tags_set = get_words_and_tags_set(x_seq, y_seq)
    """
    words, tags = [], []
    for i, j in zip(x_seq, y_seq):
        for x, y in zip(i, j):
            words.append(x)
            tags.append(y)
    return list(set(words)), list(set(tags))

def get_length(li):
    """ 
    Get length of word or tag set
    Args:
        li: a list of distinct words or tags
    Returns: the length of the list
    """
    return len(li)

def get_dict(words_set, tags_set):
    """ 
    Get dict of words and tags
    Args:
        words_set: a list of distinct words
        tags_set: a list of distinct tags
    Returns: two dicts of words and tags
    Example:
        {'word1': 0, 'word2': 5, ...}
        {'tag0': 1, 'tag2': 0, ...}
    """
    word2idx = {w: i + 1 for i, w in enumerate(words_set)}
    tag2idx = {t: i for i, t in enumerate(tags_set)}
    return word2idx, tag2idx

def saveOrLoadPickle(filename, opeartion, *dictToSave):
    """ 
    Save or load word/tag dicts
    Args:
        filename: the filename you want to save as or load to
        operation: 's' to save, 'l' to load
        *dictToSave: the dict object that needs to be saved, only works for 's' mode
    """
    if 'operation' == 's':
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(dictToSave, f)
        return 'pickle saved'
    if 'operation' == 'l':
        with open(filename+'.pkl', 'rb') as f:
            return pickle.load(f)


def map_words_and_tags(x_seq, y_seq):
    """ 
    Map words and respond tags
    Args: 
        x_seq: an array of sentences consists of words, output from load_data_and_labels
        y_seq: an array of respond sequence of tags to words in x_seq, output from load_data_and_labels
        for example:
            x_seq: np.array(['word1','word2', ..., 'word5'], ['w1','w2', ..., 'w9'], ...)
            y_seq: np.array(['tag1','tag2',...,'tag5'],['t1','t2',...,'t5'], ...)
    Returns: a list of mapped sentence
    Example: sentences[1]: [('word1', 'tag0'), ('word2', 'tag5'), ...]
    """
    sentences = []
    s = [(w, t) for w, t in zip(x_seq, y_seq)]
    for x in range(len(s)):
        sentences.append([(w, t) for w, t in zip(s[x][0],s[x][1])])
    return sentences

def pad_seq(word2idx, tag2idx, sentences, n_tags, maxlen):
    """
    Pad the sentences and corresponding tag sequences
    """
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=maxlen, sequences=X, padding="post", value=0)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    return X,y

def make_and_save_dict(filename):
    """
    Get sets contain distinct words or tags from the raw data file, save or load sets
    """
    x_seq, y_seq = load_data_and_labels(filename)
    words_set, tags_set = get_words_and_tags_set(x_seq, y_seq)
    words_set.append('ENDPAD')
    n_words = get_length(words_set)
    n_tags = get_length(tags_set)
    sentences = map_words_and_tags(x_seq, y_seq)
    word2idx, tag2idx = get_dict(words_set, tags_set)
    saveOrLoadPickle('word_dict1', 's', word2idx)
    saveOrLoadPickle('tag_dict1', 's', tag2idx)
    return word2idx, tag2idx


def prepare_data(filename, num_tags, maxlen, word2idx, tag2idx):
    """
    Prepare the formated data which can be used to train the model
    """
    x_seq, y_seq=load_data_and_labels(filename)
    sent = map_words_and_tags(x_seq, y_seq)
    X, y = pad_seq(word2idx, tag2idx, sent, num_tags, maxlen)
    return X, y

