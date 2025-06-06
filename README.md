# Psychological Reactance to Vaccine Mandates on Twitter: A Study of Sentiments in the United States

Hsieh, PH. Psychological reactance to vaccine mandates on Twitter: a study of sentiments in the United States. J Public Health Pol 46, 269–283 (2025). https://doi.org/10.1057/s41271-025-00554-0

The fine-tuned model for classifying tweets related to vaccine mandates is available on [Hugging Face](https://huggingface.co/phsieh/vaccine-mandate-bertweet-classifier).

This repository contains the following files:

* `training_no_texts.csv`: Dataset with human-annotated labels used for fine-tuning classifiers to identify tweets related to vaccine mandates.
    + `index`: Original index from the annotated dataset before shuffling for training/testing splits.
    + `id`: Tweet ID.
    + `mandate_m`: Human-annotated label indicating whether a tweet is related to vaccine mandates (1 = related, 0 = not related).
    + `usage`: Indicates whether the tweet was used for training, evaluation, or testing.
    + `pred`: Predicted label from the fine-tuned model.
    + `score0`: Model score for the "not related to vaccine mandates" class. Can be transformed into a probability using the softmax function.
    + `score1`: Model score for the "related to vaccine mandates" class. Can be transformed into a probability using the softmax function.

* `Mandate_tweets_Bertweet_training.ipynb`: Code for fine-tuning BERTweet to classify tweets about vaccine mandates. Originally run on Google Colab.

* `Mandate_tweets_supervised_learning.ipynb`: Code for training a machine learning model using a bag-of-words approach to classify tweets. Originally run on Google Colab.

* `Bertweet_inference.py`: Code for classifying tweets using `bertweet_mandate_classifier`. Originally run on high-performance computing clusters.

* `Vax_Dictionary_coding.py`: Code for dictionary-based tweet classification. Originally run on high-performance computing clusters.

* `Vaccine_roBERTa.py`: Code for using TweetNLP’s [sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and [emotion](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion-multilabel-latest) classifiers. Originally run on high-performance computing clusters.

