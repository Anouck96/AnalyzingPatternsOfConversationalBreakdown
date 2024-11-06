import numpy as np
import pandas as pd
from collections import Counter
import hdbscan
from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import StandardScaler
import csv


def cluster(feature_array, user_utt_single, savefile):
    '''Clusters the feature array and writes the clusters to a csv file'''
    # Vectorize using hdbscan and created features
    hdb = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=20, min_samples=5).fit(feature_array)

    # Get labels and count datapoints and noise
    labels = hdb.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Save clusters and write to csv
    cluster_map = pd.DataFrame()
    cluster_map['sentence'] = user_utt_single
    cluster_map['cluster'] = labels
    cluster_map.to_csv(savefile, quoting=csv.QUOTE_ALL)

    # Give extra info about the clusters
    print("Total datapoints: %d" % len(user_utt_single))
    print("Amount of clusters: %d" % n_clusters_)
    print("Amount of noise points: %d" % n_noise_)
    print(Counter(labels))


def main():

    # Load Bertmodel for embeddings
    tok = BertTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased')
    bertmodel = BertModel.from_pretrained('wietsedv/bert-base-dutch-cased')
    bertfeatvec = []

    # Read utterances from feature data
    data = pd.read_csv('x.csv')
    utterances = data['utterance_text'].tolist()

   # Remove nan from list
    utterances = [u for u in utterances if str(u) != 'nan']

    # Remove double items from list
    user_utt_single = list(dict.fromkeys([utt.strip() for utt in utterances]))

    for sent in user_utt_single:
        # Get sentence embedding from Bert
        bertoutput = tok(sent, return_tensors='pt', truncation=True)
        with torch.no_grad():
            out = bertmodel(**bertoutput)
        sentEmbedding = out.last_hidden_state.mean(dim=1).squeeze()
        sentEmbedding = sentEmbedding.numpy()
        bertfeatvec.append(sentEmbedding)

    print(len(bertfeatvec))


    # Execute clustering and create savefile
    print("Info clustering with bert sentence embeddings")
    savefilebert = 'y.csv'
    cluster(bertfeatvec, user_utt_single, savefilebert)


if __name__ == '__main__':
    main()
