import operator
from collections import defaultdict, Counter
from mynlplib.preproc import conll_seq_generator
from mynlplib.constants import OFFSET, START_TAG, END_TAG, UNK


# Deliverable 1.1
def get_tag_word_counts(trainfile):
    """
    Produce a Counter of occurences of word for each tag
    
    Parameters:
    trainfile: -- the filename to be passed as argument to conll_seq_generator
    :returns: -- a default dict of counters, where the keys are tags.
    """
    all_counters = defaultdict(lambda: Counter())
    
    raise NotImplementedError
    

    
    return all_counters


def get_tag_to_ix(input_file):
    """
    creates a dictionary that maps each tag (including the START_TAG and END_TAG to a unique index and vice-versa
    :returns: dict1, dict2
    dict1: maps tag to unique index
    dict2: maps each unique index to its own tag
    """
    tag_to_ix={}
    for i,(words,tags) in enumerate(conll_seq_generator(input_file)):
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    #adding START_TAG and END_TAG
    #if START_TAG not in tag_to_ix:
    #    tag_to_ix[START_TAG] = len(tag_to_ix)
    #if END_TAG not in tag_to_ix:
    #    tag_to_ix[END_TAG] = len(tag_to_ix)
    
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    
    return tag_to_ix, ix_to_tag


def get_word_to_ix(input_file, max_size=100000):
    """
    creates a vocab that has the list of most frequent occuring words such that the size of the vocab <=max_size, 
    also adds an UNK token to the Vocab and then creates a dictionary that maps each word to a unique index, 
    :returns: vocab, dict
    vocab: list of words in the vocabulary
    dict: maps word to unique index
    """
    vocab_counter=Counter()
    for words,tags in conll_seq_generator(input_file):
        for word,tag in zip(words,tags):
            vocab_counter[word]+=1
    vocab = [ word for word,val in vocab_counter.most_common(max_size-1)]
    vocab.append(UNK)
    
    word_to_ix={}
    ix=0
    for word in vocab:
        word_to_ix[word]=ix
        ix+=1
    
    return vocab, word_to_ix


def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    weights = defaultdict(float)
    weights[('NOUN'),OFFSET] = 1.
    return weights


# Deliverable 2.2
def get_most_common_word_weights(trainfile):
    """
    Return a set of weights, so that each word is tagged by its most frequent tag in the training file.
    If the word does not appear in the training data,
    the weights should be set so that the tagger outputs the most common tag in the training data.
    For the out of vocabulary words, you need to think on how to set the weights so that you tag them by the most common tag.
    
    Parameters:
    trainfile: -- training file
    :returns: -- classification weights
    :rtype: -- defaultdict

    """
    weights = defaultdict(float)
    
    raise NotImplementedError
    

    
    return weights


# Deliverable 4.1
def get_tag_trans_counts(trainfile):
    """compute a dict of counters for tag transitions

    :param trainfile: name of file containing training data
    :returns: dict, in which keys are tags, and values are counters of succeeding tags
    :rtype: dict

    """
    
    tot_counts = defaultdict(lambda : Counter())
    
    raise NotImplementError
    

    
    return dict(tot_counts)
