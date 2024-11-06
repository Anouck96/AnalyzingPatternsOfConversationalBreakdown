import pandas as pd
from nltk import ngrams
import string

def chatbottriggers(n):
    '''Reads chatbot trigger data and creates list of ngrams'''
    # read chatbot trigger data
    triggerdf = pd.read_excel('x')

    if n == 1:
        triggerlist = []
        for index, row in triggerdf.iterrows():
            # if possible get ngrams from each trigger and lowercase
            ngram = list(ngrams(row['Trigger'].lower().split(), n))

            # Hardcode case where punct is not working
            for item in ngram:
                if item == ('stress…',):
                    item = ('stress',)
                triggerlist.append(item[0].translate(
                    str.maketrans('', '', string.punctuation)))  # for every item remove punctuation, and create list

        # remove duplicate items from list
        triggerlist = set(triggerlist)
    if n == 2 or 3 or 4 or 5:
        triggerlist = []
        for index, row in triggerdf.iterrows():
            # if possible get ngrams from eacht trigger
            ngram = ngrams(row['Trigger'].split(), n)
            joinedngrams = [' '.join(tup) for tup in ngram]
            # remove empty lists from list
            joinedngrams = [it for it in joinedngrams if it != []]

            # create one list with all ngrams
            triggerlist = triggerlist + joinedngrams

    return triggerlist

def commonness(file, cutoff):
    '''Reads commonness file and creates list of ngrams'''
    f = open(file, 'r')
    commonngrams = []
    # Creates a list of all ngrams with a commonness above threshold
    for line in f:
        if float(line.split()[-1]) > cutoff:
            commonngrams.append(line.split('\t')[0].lower())
    # remove duplicates from list
    commonngrams = set(commonngrams)
    return commonngrams


def main():
    files = [['outputcommon1_grams.txt',1, 0.25],['outputcommon2_grams.txt',2, 0.5],['outputcommon3_grams.txt',3, 0.75],['outputcommon4_grams.txt',4, 0.75], ['outputcommon5_grams.txt',5, 0.75]]
    # Loop through set of file, n and cutoff per ngram
    for fnc in files:
        file = fnc[0]
        n = fnc[1]
        cutoff = fnc[2]
        commonngrams = commonness(file, cutoff)
        triggerlist = chatbottriggers(n)

        #Get a list of common ngrams that are not in triggerdata
        OOD = list(filter(lambda item: item not in triggerlist, commonngrams))

         # Write the list to a txt file
        with open('OOD'+file, 'w') as f:
            for line in OOD:
                f.write("%s\n" % line)


if __name__ == '__main__':
    main()