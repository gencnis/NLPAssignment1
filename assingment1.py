'''
Natural Language Processing Assignment 1
Performance Analysis of N-gram Language Models

@author Alex Wills
@author Nisanur Genc
10/10/22
'''

import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams


def preprocess(text):
    ''' Preprocesses the text and returns a list of processed tokens. 
    Removes punctuation, casefolds the text, tokenizes it, and lemmatizes it.
    @param: text string - the text to process
    @returns: list - a list of lemmatized tokens from the text '''

    # Casefold and remove punctuation
    text = text.casefold()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    
    for character in punc:
        text = text.replace(character, "")

    # Tokenize and lemmatize
    tokens = nltk.word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    lemmatizedTokens = []
    for token in tokens:
        lemmatizedTokens.append(lemmatizer.lemmatize(token))

    return lemmatizedTokens
    

def create_ngram_dict(tokens, n):
    ''' Creates a frequency dictionary of ngrams.
    @param: tokens list - a list of preprocessed tokens 
    @param: n int - the n part of the n gram
    @returns: gram_dict dictionary - dictionary where the keys are ngrams and the values are frequency counts'''
    grams = ngrams(tokens, n)

    # Make a frequency dicitonary
    gram_dict = {}
    for gram in grams:
        if gram in gram_dict:
            gram_dict[gram] += 1
        else:
            gram_dict[gram] = 1

    return gram_dict


def print_dict_stats(dictionary: dict, total_size: int, count = 10):
    ''' Pretty prints the ngram dictionary statistics with and without smoothing.
    @param: dictionary dictionary - the dictionary of ngrams where each key is an ngram and each value is the frequency of the ngram
    @param: total_size int - the total number of ngrams in the dictionary
    @param: count int (default 10) - the number of words to display statisitics for 
    
    @return: nothing. Output prints to the console.'''
    
    ngrams = list(dictionary.keys())
    print("|-------------------------------------------------------------------------------------------------------------------|")
    print("|                       Training Set Statistics                                                                     |")
    print("|-------------------------------------------------------------------------------------------------------------------|")
    print("|                      Ngram                       | Occurences | Probability (smoothed) | Probability (unsmoothed) |")
    print("|--------------------------------------------------|------------|------------------------|--------------------------|")

    for i in range( min( len(dictionary), count)):

        ngram = ngrams[i]
        smooth_probability = (dictionary[ngram] + 1) / (total_size + len(dictionary))
        rough_probability = (dictionary[ngram]) / (total_size)

        print("|{word:^50s}|{count:^12d}|   {smooth!s:.15s}      |   {rough!s:.15s}        |".format(word = str(ngram), smooth = smooth_probability, rough = rough_probability, count = dictionary[ngram]))
        # print(str(ngram[0]) + " - smooth: " + str(smooth_probability) + "\t\t\trough: " + str(rough_probability))

    print("|--------------------------------------------------|------------|------------------------|--------------------------|")
    if(len(dictionary) > count):
        print("|        ... Not all ngrams displayed ...                                                                           |")
    print("|--------------------------------------------------|------------|------------------------|--------------------------|")



def ngram_probability_smooth(ngram, ngram_dict, total_size):
    ''' Uses an n-gram dictionary to calculuate the probability of an n-gram showing up with Laplace Smoothing. 
    @param: ngram tuple - the ngram to look up in the dictionary
    @param: ngram_dict dictionary - a dictionary where the keys are ngrams (tuples the same length as the ngram parameter), and the values are the number of occurences of that ngram in the training set
    @param: total_size int - the total number of ngrams in the dictionary (the sum of all the frequencies) 
    
    @return: probability float - the smoothed probability of the ngram in the model '''
    probability: float

    # Calculate probability with smoothing
    if(ngram in ngram_dict):
        probability = (ngram_dict[ngram] + 1.0) / (total_size + len(ngram_dict))
    else:
        # If the ngram is not in the dictoinary, assign a low probability
        probability = 1.0 / (total_size + len(ngram_dict))

    return probability



def sentence_probability(sentence: str, ngram_model: list[dict], total_size):
    ''' Uses an ngram model to calculate the likelihood of a sentence being a valid sentence.
    @param: sentence str - the sentence to test
    @param: ngram_model - a list of ngram dictionaries. Element 0 should be a monogram dictionary, element 1 should be a bigram dictionary, etc. For an n-gram model, there should be n dictionaries in this list.
        Each dictionary should have keys that are tuples of n-grams, and values that are the frequencies of the ngram
    @param: total_size int - the total number of ngrams in the training set (the length of the training set) 
    
    @return: probability float - the probability of the sentence being a valid sentence based on the ngram model '''


    ngram_size = len(ngram_model)
    sentence_tokens = preprocess(sentence)
    

    conditional_prob: float

    sliding_window = []
    sliding_window.append(sentence_tokens[0])

    # Add the first word, which is not a conditional probability
    cumulative_probability = ngram_probability_smooth((sentence_tokens[0]), ngram_model[0], total_size)

    # Calculate the first ngram probabilities until the sliding window is full (based on ngram size)
    # NOTE: right now, this model does not place tags to start the sentence / end the sentence.
    # So for the beginning of the sentence "a b c d" with a trigram model, we are looking at (a), (a b), (a b c), (b c d) INSTEAD OF ('' '' a), ('' a b) (a b c) etc.
    for i in range(1, min(len(sentence_tokens), ngram_size)):
        # Add the next word and find the probability of the n-gram
        sliding_window.append(sentence_tokens[i])


        conditional_prob = ngram_probability_smooth(tuple(sliding_window), ngram_model[len(sliding_window) - 1], total_size )  # Uses the ngram model based on the size of the window. Monogram -> bigram -> ... -> n-gram
        # For the conditional part, we divide by the probability of everything before the new word (this should be greater than or equal to the probability of the window with the new word)
        conditional_prob /= ngram_probability_smooth(tuple( sliding_window[0:-1]), ngram_model[len(sliding_window) - 2], total_size )
        
        cumulative_probability *= conditional_prob

    # Now slide the window the rest of the way
    for i in range(ngram_size, len(sentence_tokens)):
        # Slide the window one word over
        sliding_window.pop(0)
        sliding_window.append(sentence_tokens[i])

        # Find probability of the new word in the sentence, given the previous words in the window.
        if(len(ngram_model) == 1):
            # Monogram model does not use conditional probability
            cumulative_probability *= ngram_probability_smooth(tuple(sliding_window), ngram_model[0], total_size)
        else:
            cumulative_probability *= ngram_probability_smooth(tuple(sliding_window), ngram_model[ngram_size - 1], total_size) \
                / ngram_probability_smooth(tuple( sliding_window[0:-1]), ngram_model[ngram_size - 2], total_size)

    return cumulative_probability



def main():
     
   # ---------------------------------------------------------------------
   # OPENS TRANING SET FILES
   # ---------------------------------------------------------------------

    rootdir = 'DUC 2005 Dataset/TrainingSet/' # the root of the path 
    folder_list = []

    # this loop goes through the small folders in the Training Set folder 
    # and append the path it to the folder list
    # also adds the the original path to the folder path list
    for file in os.listdir(rootdir):
            folder_list.append( os.path.join(rootdir, file))
        
    # creates empty list for the text file paths
    textfile_list = []

    # goes through every little folder in the folder list to add to the text path list
    for folder in folder_list:
        text_files = os.listdir(folder)
        for text_file in text_files:
                textfile_list.append( os.path.join(folder, text_file))

    # goes through every element of the list to open the every file
    for i in textfile_list:
        # opens the file 
        with open(i, 'r', encoding = "utf8") as f:
            lines = f.read()

            tokens = preprocess(lines)
            total_size = len(tokens)

            unigrams = create_ngram_dict(tokens, 1)
            bigrams = create_ngram_dict(tokens, 2)
            trigrams = create_ngram_dict(tokens, 3)
            fourgrams = create_ngram_dict(tokens, 4)

            # print_dict_stats(unigrams, total_size, 100)

            bigram_model = [unigrams, bigrams]

            print(str(sentence_probability("Cystic fibrosis is a cringe illness according to the journals.", bigram_model, total_size)))
            print(str(sentence_probability("Cystic cringe according the to journals a is fibrosis illness.", bigram_model, total_size)))

            print(str(sentence_probability("Cystic fibrosis is a cringe illness according to the journals.", [unigrams], total_size)))
            print(str(sentence_probability("Cystic cringe according the to journals a is fibrosis illness.", [unigrams], total_size)))

    # ---------------------------------------------------------------------
    # TRAINING CODE ENDS
    # ---------------------------------------------------------------------



    # ---------------------------------------------------------------------
    # OPENS TEST SET FILES
    # ---------------------------------------------------------------------

    rootdir = 'DUC 2005 Dataset/TestSet/'
    folder_list = [] # this is the list for the folder 

    # this loop goes through the small folders in the Test Set folder 
    # and append the path it to the folder list
    # also adds the the original path to the folder path list
    for file in os.listdir(rootdir):
            folder_list.append( os.path.join(rootdir, file))
        
    # creates empty list for the text file paths
    textfile_list = []

    # goes through every little folder in the folder list to add to the text path list
    for folder in folder_list:
        text_files = os.listdir(folder)
        for text_file in text_files:
                textfile_list.append( os.path.join(folder, text_file))

    # goes through every element of the list to open the every file
    for i in textfile_list:
        # opens the file 
        with open(i, 'r', encoding = "utf8") as f:
            lines = f.read()

            tokens = preprocess(lines) 
            total_size = len(tokens)
            

            unigrams = create_ngram_dict(tokens, 1)
            bigrams = create_ngram_dict(tokens, 2)
            trigrams = create_ngram_dict(tokens, 3)
            fourgrams = create_ngram_dict(tokens, 4)

            # print_dict_stats(unigrams, total_size, 100)

            bigram_model = [unigrams, bigrams]

            print(str(sentence_probability("Cystic fibrosis is a cringe illness according to the journals.", bigram_model, total_size)))
            print(str(sentence_probability("Cystic cringe according the to journals a is fibrosis illness.", bigram_model, total_size)))

            print(str(sentence_probability("Cystic fibrosis is a cringe illness according to the journals.", [unigrams], total_size)))
            print(str(sentence_probability("Cystic cringe according the to journals a is fibrosis illness.", [unigrams], total_size)))

    # ---------------------------------------------------------------------
    # TEST CODE ENDS
    # ---------------------------------------------------------------------






if __name__ == "__main__":
    main()