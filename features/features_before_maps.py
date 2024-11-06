import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import language_tool_python
from pattern.nl import sentiment
from nltk import ngrams
import string
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def get_lang_tool_feats(sent, tool):
    '''Counts number of times error occurs based on selected langtool categories'''
    beledigend = overig = eigennamen = grammar = repetitions = leenwoorden = leesbaarheid = whitespace_rule = typos = ouderwets = dyslexie = vormfouten = confused_words = unexpected = test = full_sentences = 0
    # Remove filtered items and replace by nothing
    sent = sent.replace('<FILTERED>', '')
    sent = sent.replace('  ', ' ')
    matches = tool.check(sent)
    for item in matches:
        if item.category == 'BELEDIGEND':
            beledigend = beledigend + 1
        elif item.category == 'OVERIG':
            overig = overig + 1
        elif item.category == 'EIGENNAMEN':
            eigennamen = eigennamen + 1
        elif item.category == 'GRAMMAR':
            grammar = grammar + 1
        elif item.category == 'REPETITIONS':
            repetitions = repetitions + 1
        elif item.category == 'LEENWOORDEN':
            leenwoorden = leenwoorden + 1
        elif item.category == 'LEESBAARHEID':
            leesbaarheid = leesbaarheid + 1
        elif item.category == 'WHITESPACE_RULE':
            whitespace_rule = whitespace_rule + 1
        elif item.category == 'TYPOS':
            typos = typos + 1
        elif item.category == 'OUDERWETS':
            ouderwets = ouderwets + 1
        elif item.category == 'DYSLEXIE':
            dyslexie = dyslexie + 1
        elif item.category == 'VORMFOUTEN':
            vormfouten = vormfouten + 1
        elif item.category == 'CONFUSED_WORDS':
            confused_words = confused_words + 1
        elif item.category == 'UNEXPECTED':
            unexpected = unexpected + 1
        elif item.category == 'TEST':
            test = test + 1
        elif item.category == 'FULL_SENTENCES':
            full_sentences = full_sentences + 1
        else:
            pass
    return beledigend, overig, eigennamen, grammar, repetitions, leenwoorden, leesbaarheid, whitespace_rule, typos, ouderwets, dyslexie, vormfouten, confused_words, unexpected, test, full_sentences

def commonness(OODdata, ngram):
    '''Checks if the ngrams from the utterances are in the out of domain ngrams'''
    checks = []
    commoncounts = 0
    for n in ngram:
        # Create string from tuple, lower and remove punctuation
        n = ' '.join(n).lower()
        n = n.translate(str.maketrans('', '', string.punctuation))
        if n in OODdata:
            commoncounts = commoncounts + 1

    return commoncounts

def outofdomain(file):
    '''Creates list for the out of domain ngrams'''
    OODdata = []
    file = open(file, encoding='utf-8')
    for line in file:
        OODdata.append(line.rstrip())

    return OODdata


def createfeats(utterance, tool, OODdata1, OODdata2, OODdata3, OODdata4, OODdata5, ref_words):
    feat_vec = []
    # create ngrams for sentences
    ngram1 = list(ngrams(utterance.lower().split(), 1))
    ngram2 = list(ngrams(utterance.lower().split(), 2))
    ngram3 = list(ngrams(utterance.lower().split(), 3))
    ngram4 = list(ngrams(utterance.lower().split(), 4))
    ngram5 = list(ngrams(utterance.lower().split(), 5))

    # get languagetool error counts: 16 categories
    (beledigend, overig, eigennamen, grammar, repetitions, leenwoorden, leesbaarheid,
     whitespace_rule, typos, ouderwets, dyslexie, vormfouten, confused_words, unexpected,
     test, full_sentences) = get_lang_tool_feats(utterance, tool)

    # get counts for commonness
    commoncount1 = commonness(OODdata1, ngram1)
    commoncount2 = commonness(OODdata2, ngram2)
    commoncount3 = commonness(OODdata3, ngram3)
    commoncount4 = commonness(OODdata4, ngram4)
    commoncount5 = commonness(OODdata5, ngram5)
    commoncounts = commoncount5 + commoncount4 + commoncount3 + commoncount2 + commoncount1

    s = utterance.split()
    lowersent = utterance.lower().split()
    references = len(set(lowersent).intersection(ref_words))
    nrsent = len(sent_tokenize(utterance))
    pol = sentiment(utterance)[0]
    subj = sentiment(utterance)[1]
    # len(s) is the number of words in the sentence
    # len(sent) is the number of characters in the sentence
    feat_vec.append([len(s), len(utterance), nrsent, pol, subj, references, beledigend, overig, eigennamen, grammar, repetitions, leenwoorden, leesbaarheid, whitespace_rule, typos, ouderwets, dyslexie, vormfouten, confused_words, unexpected, test, full_sentences, commoncounts])
    X = np.array(feat_vec)
    return X

def getfeats_csv(row_num, name, final_data, data, tool, OODdata1, OODdata2, OODdata3, OODdata4, OODdata5, ref_words, list_of_feature_vectors):
    for item in row_num:
        error, sesid_error, from_error = (data.iloc[item][['removed for anonymity']])
        before_error, sesid_before_error, from_before_error = (data.iloc[item-1][['removed for anonymity']])

        # Remove newlines and change to string
        if isinstance(before_error, str):
            before_error = before_error.replace('\n', '')
        else:
            before_error = str(before_error)

       # if the utterance is 'nan' use an empty feature array
        if before_error != 'nan':
            before_feat_vec = (createfeats(before_error, tool, OODdata1, OODdata2, OODdata3, OODdata4, OODdata5, ref_words))
        else:
            before_feat_vec = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # Create a list of the feature vectors of the items before a repair
        try:
            before_feat_vec = before_feat_vec.tolist()[0]
        except AttributeError:
            before_feat_vec = before_feat_vec[0]
        before_feat_vec.append(before_error)
        list_of_feature_vectors.append(before_feat_vec)
    #return final_data
    return list_of_feature_vectors

def main():

    # Read the OODdata
    OODdata1 = outofdomain('OODoutputcommon1_grams.txt')
    OODdata2 = outofdomain('OODoutputcommon2_grams.txt')
    OODdata3 = outofdomain('OODoutputcommon3_grams.txt')
    OODdata4 = outofdomain('OODoutputcommon4_grams.txt')
    OODdata5 = outofdomain('OODoutputcommon5_grams.txt')

    tool = language_tool_python.LanguageTool('nl')

    # Load dutch referencing words
    file_ref = open('verwijswoorden.txt', 'r')
    ref_words = []
    for line in file_ref:
        ref_words.append(line.rstrip())

    datafiles = ['removed for anonymity']

    final_data = pd.DataFrame(columns=['Sender', 'Session_ID', 'Feature_Vector'])
    list_of_feature_vectors = []
    column_names = ['Words', 'characters', 'sent', 'pol', 'subj',
                    'ref', 'beledig', 'overig', 'eigennaam', 'grammar',
                    'rep', 'leen', 'leesb', 'whites', 'typos',
                    'ouder', 'dysl', 'vorm', 'conf', 'unex',
                    'test', 'full_s', 'common', 'utterance_text']
    for fi in datafiles:
        data = pd.read_excel(fi)
        row_num_sorry = data[data['Msg Attachment'] == 'removed for anonymity'].index.tolist()
        row_num_helaas = data[data['Msg Attachment'] == 'removed for anonymity'].index.tolist()

        final_data = getfeats_csv(row_num_sorry, 'sorry', final_data, data, tool, OODdata1, OODdata2, OODdata3, OODdata4, OODdata5, ref_words, list_of_feature_vectors)

        final_data = getfeats_csv(row_num_helaas, 'helaas', final_data, data, tool, OODdata1, OODdata2, OODdata3,
                                  OODdata4, OODdata5, ref_words, list_of_feature_vectors)

        feat_df = pd.DataFrame(final_data, columns=column_names)
    feat_df.to_csv('features/all_data_with_anonymous_test.csv', index=False)



if __name__ == '__main__':
    main()
