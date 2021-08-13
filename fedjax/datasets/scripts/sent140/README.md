
# Sent140

Convenience scripts for downloading, preprocessing, and creating a Sent140
dataset for use with FedJAX.

Users should run build_dataset.sh and pass two flags that give the path to the
directories that they want the output files written to and intermediate steps
written to. An optional third flag for the fraction of the dataset that the user
wants is also available (defaults to 0.1 or ten percent).

e.g.

```
./build_dataset.sh -o Users/username/Desktop -t /tmp/sent140 -f 0.1
```

In the end, the user is left with three files in the output_dir (-o):
normalized_lines.txt, vocab.txt, and sent140_dataset.sqlite.
normalized_lines.txt is a text file where every line is a normalized tweet
from the sent140 dataset. vocab.txt is a text file where every line is a
word from the sent140 dataset sorted from most occurring to least occurring.
sent140_dataset.sqlite is an SQLite file that contains the Sent140 dataset.

### About Sent140:

Sent140 is a sentiment classification dataset, where approximately 1.6 million
tweets were collected and labeled as positive or negative.

#### Data Collection:

The Sent140 training data set contains 1.6 million tweets collected through the
Twitter Search API. 800,000 tweets with positive emojis, and 800,000 tweets with
negative emojis. Only tweets that contained positive and/or negative emojis were
retrieved through keyword search. The data was automatically labeled, as opposed
to being annotated by a human. The polarity respective to the emoji was assigned
to the tweet. The presence of a positive emoji led to a polarity of 4, of a
negative emoji to that of 0. Neutral tweets weren’t considered in the training
data, many tweets do not have sentiment, however, so it is a current limitation.
Any tweet containing both positive and negative emojis was removed. Retweets
were removed. Tweets with “:P” were removed. Repeated tweets were removed. Only
tweets in English were used.

After the tweets were labeled, they were stripped out of any emojis. This
causes the classifier to learn from the other features present in the tweet. The
classifier uses these non-emoticon features to determine the sentiment. If the
test data contains an emoji, it does not influence the classifier because emoji
features are not part of the training data. This is a current limitation because
it would be useful to take emojis into account when classifying test data.

The test data was manually collected. A small set (in the hundreds) of negative
tweets, positive tweets, and neutral tweets were manually labeled. It is
important to note that the training data only contains positive and negative
labeled tweets, 4 and 0, respectively. However, the testing data contains
positive, neutral, and negative labels, 4, 2, and 0, respectively. To maintain
consistency with the FedProx repository, this file converts positive labels (4)
to 1 and neutral (2) and negative labels (0) to 0. Not all the test data has
emoticons.

#### Table 1: List of Emoticons

Emoticons mapped to positive sentiment:

> :)
>
> :-)
>
> \: )
>
> :D
>
> =)

Emoticons mapped to negative sentiment:

> :(
>
> :-(
>
> \: (


#### Labels:

Each Tweet has six fields associated with it:

0. the polarity of the tweet (0 = negative, 1 = positive)

1. the id of the tweet (2087)

2. the date of the tweet (Sat May 16 23:58:44 UTC 2009)

3. the query (lyx). If there is no query, then this value is NO_QUERY.

4. the user that tweeted (robotickilldozr)

5. the text of the tweet (Lyx is cool)

#### Current Benchmarks:

Pang and Lee reported 81.0%, 80.4%, and 82.9% accuracy for Naive Bayes, MaxEnt,
and SVM, respectively. This is very similar to the Stanford paper results of
81.3%, 80.5%, and 82.2% for the same set of classifiers.

#### Miscellaneous Notes on Data:

Tweets are more casual and limited to 140 characters of text.

#### Resources:

[Original Paper](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

[FedProx Repository](https://github.com/litian96/FedProx/tree/master/data/sent140)

[LEAF Repository](https://github.com/TalwalkarLab/leaf/tree/master/data/sent140)
