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


## Statistics and figures

## Clustering 







Note: GitHub Copilot was used to help write the code.
