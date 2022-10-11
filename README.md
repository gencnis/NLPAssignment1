# NLPAssignment1
## Probabilistic Language Model for Finding Sentence Likelihood

# Authors: Nisanur Genc and Alex Wills
Performance Analysis of N-gram Language Models Assignment.


## How to run program
1. Open a terminal in this directory.
2. Run `py assignment1.py`.
3. Results will be printed to the terminal. 

### Configuring the program
In order for this program to generate a model from a training corpus, and to test its effectiveness on a test corpus, you must have two subdirectories containing the training files. In our program, the training set is in `DUC 2005 Dataset/TrainingSet/` and the testing set is in `UDUC 2005 Dataset/TestSet/`. If you wish to name the directories differently, you may edit `assignment1.py`'s main function, where the filepaths are specified as the variable `rootdir`.

### Outputs
This program will generate a unigram probability model and a bigram probability model from the Training Corpus. Then it will apply these models on every sentence in every file of the Test Corpus and keep track of the average sentence likelihood that it outputs. At the end of the program, it will print out a report stating what the average likelihoods for the bigram and unigram models are.

## How program works

### Probabilistic model

To build a probabilistic model, we scan through a training corpus to see what words appear, how frequently they appear, and what words apper around them. With these counts, we can generate a probability table for every word, and every 2 words. With this model, we can take in a sample sentence and use our probaility tables to determine the likelihood that the sentence is a valid sentence (according to the training set).

Given a sentence {w1, w2, w3, w4} where each w is a word, the probability of the sentence = P(w1) * P(w2 | w1) * P(w3 | w1, w2) * P(w4 | w1, w2, w3). This takes every previous word in the sentence into account.

In our model, we only look at the previous few words, estimating probability as P(w1) * P(w2 | w1) * P(w3 | w2) * P(w4 | w3). The number of words we look at is determined by the n in N-gram


### N-gram model
Our model uses a unigram model and a bigram model, for looking at sets of 1 and 2 words, respectively. 

The unigram model analyzes sentences based on each word individually, and their probability of showing up in the training set. Because it only looks at words one at a time, the order of the words does not matter in determining probability.

The bigram model analyzes sentences based on pairs of words, and so word order is taken into account.

With the sentence {w1, w2, w3, w4, w5}, for a trigram model, for example, we look at P(w1), P(w2 | w1), P(w3 | w1, w2), P(w4 | w2, w3), ...

As you can see, the begining of the sentence is treated as a unigram, then the second word is a bigram, then the third word is a trigram, up until we reach whatever n-gram our model is running.
As a result, our trigram model additionally requires a bigram and unigram model for the first probabilities of each sentence.

One way around this, which we did not implement, is by making every word in the sentence an n-gram with tags for the start and end of the sentence. So instead of P(w1), for a trigram you would have P(w1 | start, start).
P(w2 | w1) -> P(w2 | start, w1). For the calculations, you will always need a unigram model, but this method of inserting tags into sentences removes the need for any intermediate models.

Since our program does not take this approach, we represent an n-gram model as a list of dictionaries, where each dictionary is a probability table for a different n-gram, from 1 to n. With this approach, creating any sized n-gram model is not too difficult, but it could be simplified in future code.