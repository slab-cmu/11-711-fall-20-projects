from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal
import torch
from mynlplib.constants import * 
from mynlplib import preproc, bilstm, hmm, viterbi, most_common, scorer
import numpy as np

def setup():
    global word_to_ix, tag_to_ix, X_tr, Y_tr, model
    
    vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE, max_size=6900)
    tag_to_ix={}
    for i,(words,tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE)):
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    
    if START_TAG not in tag_to_ix:
        tag_to_ix[START_TAG]=len(tag_to_ix)
    if END_TAG not in tag_to_ix:
        tag_to_ix[END_TAG]=len(tag_to_ix)
        
    X_tr = []
    Y_tr = []
    for i,(words,tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE)):
        X_tr.append(words)
        Y_tr.append(tags)
    
    torch.manual_seed(711);
    
    embedding_dim=30
    hidden_dim=30
    model = bilstm.BiLSTM_CRF(len(word_to_ix),tag_to_ix,embedding_dim, hidden_dim)

#6.1
def test_forward_alg():
    global model, X_tr, word_to_ix
    torch.manual_seed(711);
    
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    alpha = model.forward_alg(lstm_feats)
    assert_almost_equal(alpha.item(), 104.916992, places=4)

    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[1], word_to_ix))
    alpha = model.forward_alg(lstm_feats)
    assert_almost_equal(alpha.item(), 65.290924, places=4)

#6.2
def test_score_sentence():
    global model, X_tr, Y_tr, word_to_ix, tag_to_ix
    torch.manual_seed(711);
    
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    score = model.score_sentence(lstm_feats, bilstm.prepare_sequence(Y_tr[0], tag_to_ix))
    print(tag_to_ix)
    assert_almost_equal(score.item(), 2.659940, places=4)
    
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[1], word_to_ix))
    score = model.score_sentence(lstm_feats, bilstm.prepare_sequence(Y_tr[1], tag_to_ix))
    assert_almost_equal(score.item(), -2.397999, places=4)

#6.3
def test_predict():
    global model, X_tr, Y_tr, word_to_ix, tag_to_ix
    torch.manual_seed(711);
    best_tags = model.predict(bilstm.prepare_sequence(X_tr[5], word_to_ix))
    eq_(best_tags[0:5],['NOUN', 'ADJ', 'CONJ', 'ADV', 'ADJ'])
    
    best_tags = model.predict(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    eq_(best_tags[0:5],['X', 'NUM', 'INTJ', 'PART', 'AUX'])

#6.4
def test_neg_log_likelihood():
    global model, X_tr, Y_tr, word_to_ix, tag_to_ix
    torch.manual_seed(711);
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[5], word_to_ix))
    loss = model.neg_log_likelihood(lstm_feats, bilstm.prepare_sequence(Y_tr[5], tag_to_ix))
    assert_almost_equal(loss.item(),50.326389, places=4)

    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    loss = model.neg_log_likelihood(lstm_feats, bilstm.prepare_sequence(Y_tr[0], tag_to_ix))
    assert_almost_equal(loss.item(),102.239616, places=4)
