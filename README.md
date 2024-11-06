# Analyzing Patterns Of Conversational Breakdown

This repository contains the code and (anonymous) data for the paper "Analyzing Patterns of Conversational Breakdown in Human-Chatbot Customer Service Conversations".

## Data
Contains three files:
- all_data_anonymous_test.csv. This data file contains featurizations for the test data (utterances before a repair).
- nonrepair_features_nobuttons_anon.csv This data file contains featurization for a random control data set which consists out of utterances that do not occur together with a repair. This selection also removed some utterances that were buttons.
- finalfeatanonymoussesID.csv Contains a larger selection of featurized utterances before and after repairs (labeled 'sorry' and 'helaas') and a bot response to a complaint (labeled 'vervelend'). 

## Create data
Contains the code for the creation of the anonymous data from the non-anonymous original data.

## Features
This directory contains code and files that are needed to create features.
- verwijswoorden.txt Contains words that refer to external entities or previous sentences. This is adapted from https://www.scribbr.nl/taalregels-schrijftips/verwijswoorden/.
- score_commonness_wikidump.py Scores the commonness of entities based on wikipedia dump files. Script adapted from Florian Kunneman's script. Instead of using annotations from the json file it reads these out of the text based on the html tags. This script uses colibri-core (https://github.com/proycon/colibri-core). For entity extraction the following script from Florian Kunneman can be used (https://github.com/fkunneman/DTE/blob/master/dte/classes/commonness.py).
- createOODlist.py Creates a list of n-grams that are common based on commonness scores, but not in the chatbots training data.

- features.py Creates finalfeatanonymoussesID.csv
- features_before_maps.py Creates all_data_anonymous_test.csv but retains the original sentence (which is not in this csv file because of anonymity reasons).
- random_non_error_data_noButtons.py Creates nonrepair_feature_nobuttons_anon.csv 

## Statistics and figures
This directory contains code to create the table from the paper (including statistics test), create the heatmap and code that counts some other descriptives of the total dataset (such as average conversation length).

- average_lengthconversations.py Calculates average length of total data set.
- countrecognizedEnglish.py Calculates how often the chatbot recognizes a user speaking English.
- Create_comparetable.py Creates the table from the paper, including the statistical test.
- make_heatmap_feature_correlation.py Creates the heatmap from the paper by calculating correlations between features.

## Clustering 
- bert_cluster.py Contains the code to apply clustering to the user utterances before a repair by using bert sentence embeddings.
- table.txt Contains results of the clustering.





Note: GitHub Copilot was used to help write the code.
