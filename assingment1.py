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


def print_dict_stats(dictionary, total_size):
    
    for ngram in dictionary:
        smooth_probability = (dictionary[ngram] + 1) / (total_size + len(dictionary))
        rough_probability = (dictionary[ngram]) / (total_size)

        print(str(ngram) + " - smooth: " + str(smooth_probability) + "\t\t\trough: " + str(rough_probability))



def main():
    
    file_name = "scienceShort.txt"

    assert os.path.exists(file_name), "I was not able to find the file at, "+str(file_name)

    # Open the file as f
    with open(file_name, 'r', encoding = "utf8") as f:
        lines = f.read()

        tokens = preprocess(lines)
        total_size = len(tokens)
        

        unigrams = create_ngram_dict(tokens, 1)
        bigrams = create_ngram_dict(tokens, 2)
        trigrams = create_ngram_dict(tokens, 3)
        fourgrams = create_ngram_dict(tokens, 4)

        print_dict_stats(unigrams, total_size)

        




if __name__ == "__main__":
    main()