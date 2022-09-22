import os

def preprocess(text):
    ''' Preprocesses the text '''

    # Casefold and remove punctuation
    text = text.casefold()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    


    # Removing punctuations in string
    # Using loop + punctuation string
    # for character in text:
    #     if character in punc:
    #         text = text.replace(character, "")

    for character in punc:
        text = text.replace(character, "")

    
    print(text)
    



def main():
    print("Example file hehe")


    file_name = "scienceShort.txt"

    assert os.path.exists(file_name), "I was not able to find the file at, "+str(file_name)

    # Open the file as f
    with open(file_name, 'r', encoding = "utf8") as f:
        lines = f.read()

        preprocess(" Here is a string with slashes \\\\ \' \" {} ")




if __name__ == "__main__":
    main()