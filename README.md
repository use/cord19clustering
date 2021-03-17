## Authors

  * Will Hall
  * Colin Middleton

## Overview

This repository contains tools used during our research project "The Efficient K-Means Clustering of Documents Using TF IDF", for Dr. Dan Li's CSCD 530: Big Data Analytics at Eastern Washington University during Winter Quarter 2021.

The project uses the k-means clustering technique on a sparse matrix of TF-IDF scores to group documents by subject.

## Data

The data is a parsed set of ~150,000 papers relating to COVID-19, available on Kaggle:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=568

> In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 400,000 scholarly articles, including over 150,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.

Specifically, we focused on the English language papers found in document_parses/pdf_json.

## Source Files

  * **library.py**  
    Preprocessing functions.

  * **remove_low_frequency_words.py**  
    Another preprocessing step which removes one-off words from the vocabulary and the corpus.

  * **kmeans.py**  
    Clustering functions.

  * **run_kmeans.py**  
    Performs the actual clustering once preproccessing has been completed. It reads pickled data which was generated during the preprocessing phase.

  * **analyze_results.py**  
    Analyzes the program output and prints out something formatted more nicely.

  * **find_optimal_k.py**  
    Finds optimal k by clustering a subcorpus with increasing k values.

  * **avg_word_overlaps.py**  
    Produces some stats about documents.

  * **binary_lsh.py**  
    Some experiments that didn't work out.

## Profiling with cProfile

python -m cProfile -o profile.txt run_kmeans.py  
python analyze_stats.py > analysis_output.txt
