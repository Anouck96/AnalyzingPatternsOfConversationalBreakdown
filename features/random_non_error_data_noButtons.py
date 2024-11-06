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
import random
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

def getfeats_csv(utterance, tool, OODdata1, OODdata2, OODdata3, OODdata4, OODdata5, ref_words, list_of_feature_vectors):
    for item in utterance:

       # if the utterance is 'nan' use an empty feature array
        if item != 'nan':
            feats = (createfeats(item, tool, OODdata1, OODdata2, OODdata3, OODdata4, OODdata5, ref_words))
        else:
            feats = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        try:
            feats = feats.tolist()[0]
        except AttributeError:
            feats = feats[0]
        feats.append(item)
        list_of_feature_vectors.append(feats)
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

    buttons = ['removed for anonymity']
    user_utt = []
    total_list = []
    for i, file_path in enumerate(datafiles):
        data = pd.read_excel(file_path)
        row_filter_sorry = data[data['Msg Attachment'] == 'removed for anonymity']
        row_filter_helaas = data[data['Msg Attachment'] == 'removed for anonymity']
        session_ids_sorry = row_filter_sorry['Session'].tolist()
        session_ids_helaas = row_filter_helaas['Session'].tolist()

        # Obtain al list of all sessions that have any repair
        combined = list(set(session_ids_sorry + session_ids_helaas))

        # Exclude data that has a session ID that is in the list
        filtered_data = data[~data['Session'].isin(combined)]
        norepair_dataframe = pd.DataFrame(filtered_data)

        # Count the occurrences of each session row
        session_counts = norepair_dataframe['Session'].value_counts()
        filtered_8 = norepair_dataframe[norepair_dataframe['Session'].isin(session_counts[session_counts > 8].index)]
        norepair_eightplus_dataframe = pd.DataFrame(filtered_8)

        # Get the row number of the welcome intent
        row_number = norepair_eightplus_dataframe[norepair_eightplus_dataframe['Msg Text'] == 'removed for anonymity'].index.tolist()
        # Get the first user utterance
        index_start_utt = [utt + 2 for utt in row_number]

        second_turn = []
        third_turn = []
        for item in index_start_utt:
            if data.iloc[item + 3]['From'] == 'bot':
                second_turn.append(item + 2)
            else:
                pass
            if data.iloc[item + 5]['From'] == 'bot':
                third_turn.append(item + 4)
            else:
                pass

        total_list = second_turn + third_turn + index_start_utt


        # remove the \n from the utterances
        for id in total_list:
            user_utt.append(str(data['Msg Text'].iloc[id]).replace('\n', ''))

        # Remove welcome from the user_utt list. This means the conversation has already begun anew.
        user_utt = [utt for utt in user_utt if 'removed for anonymity' not in utt]

        # Remove items that are probably buttons
        user_utt = [utt for utt in user_utt if not any(phrase in utt for phrase in buttons)]


        # Remove item that is testing
        user_utt = [utt for utt in user_utt if 'removed for anonymity' not in utt]

        if i == 0 or i == 1 or i == 2 or i == 3:
            selected_items = random.sample(user_utt, 916)
        else:
            selected_items = random.sample(user_utt, 917)

        final_data = getfeats_csv(selected_items, tool, OODdata1, OODdata2, OODdata3, OODdata4, OODdata5, ref_words, list_of_feature_vectors)

        feat_df = pd.DataFrame(final_data, columns=column_names)
        feat_df = feat_df.iloc[:-5]
    feat_df.to_csv('features/nonrepairs_features_nobuttons.csv', index=False)




if __name__ == '__main__':
    main()
