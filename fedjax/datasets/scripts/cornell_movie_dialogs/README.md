# Cornell Movie Dialogs
[TOC]

build_dataset.txt downloads, normalizes, and moves the Cornell Movie Lines
federated dataset into SQLite database.
The script also makes a vocab.txt of the most to least common words in the dataset.

Simply run build_dataset.sh with:

```
sh build_dataset.sh -d your_data_directory -o your_output_directory -k
[optional to save intermediate output]
```

in the command line.

Flags:

- d: The data directory where files will be downloaded to

- o: The output directory

- k: If k is a flag, the files numbered.txt, normalized_lines.txt, cornell.zip,
  and folder cornell_dataset are not deleted at the end of the script


## Objective


The objective is to make a script that downloads and preprocesses the Cornell
Movie-Dialogs dataset. This dataset is then converted into clients and numpy
arrays which are then moved into an SQLite database.

Requirements:

- Download and unzip cornell.zip file from
http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip

- normalize dialogue (training data)

- move data so that dialogue and metadata from a speific character
(client) is mapped to character (data is stored in numpy arrays)

- move data into SQLite database file

## Background

[FedJAX](https://fedjax.readthedocs.io/en/latest/) is a library for developing
federated learning algorithms using JAX. FedJAX has a collection of various
federated datatsets to run federated learning algorithms on, however more
datasets will increase the usability of the library. In this project, the
Cornell Movie Dialogs dataset will be added to the FedJAX datasets. Adding a new
and popular dataset will allow users to have more baselines and comparison
points. Having this new dataset will also make FedJAX more useful for research.

## Overview

The [Cornell Movie-Dialogs corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
contains a large collection of fictional
conversations from raw movie scripts. The lines can be slpit up into clients
(individual characters) and the corpus (dialogue ffrom the characters) can be
used as federated data. Due to legal reasons, Google cannot host the dataset
directly, so a script will be made to download and preprocess the text files,
and then the training data and metadata from the files will need to be moved to
numpy dictionaries and eventually into a SQLite database.

## Detailed Design

### build_dataset.sh
This shell script downloads the orignal cornell movie-dialgoue zip file into
the passed data directory. The script will use wget for downloading and then
unzip the file as well. It calls data_to_sqlite.py, create a vocab file and then
clean up the intermediate files when -k is not a flag.

### data_to_sqlite.py

This script reads from both the movie_lines.txt file and the
movie_characters_metadata.txt file to add the data into a python
dictionary, with the outer keys as client ID and the inner keys as 'line_id',
'line', 'name', 'movie', and 'gender'. The client ID, line ID and line are all
taken from movie_lines.txt and the name, movie and gender are all taken from
movie_characters_metadata.txt. The motivation for keeping the metadata along
with the training data (lines), is that the metadata information may be used for
some learning algorithms, like gender was used in the Chameleons in imagined
conversations paper.

### normalize_dialogue_string()

This method takes in a line of dialogue and normalizing
it by removing puncutation and decasing the string. Ellipses,
dashes and html tags are removed. Numbers are replaced by \<NUMERIC\>.


## Related Work

The Cornell Movie-Dialogs corpus dataset has been used in the
[original paper](https://arxiv.org/abs/1106.3077), as well as the Google paper
[Agnostic Federated Learning](https://arxiv.org/abs/1902.00146). The latter
paper processed the dataset using scripts that remove
`" , -  ( ) * _ : " '" ^ + $ . ? ! < > ` characters, all
spaces in the beginning, double spaces and empty lines, and HTML tags. It also
replaces ". ", "; ',  "? " and "! " with next line. The output file is each line
of dialogue preprocessed. More information and details about the dataset can be found at
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html.




