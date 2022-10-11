from cgitb import text
from logging import root
import os


rootdir = 'DUC 2005 Dataset/TrainingSet/'
folder_list = []

for file in os.listdir(rootdir):
        folder_list.append( os.path.join(rootdir, file))
    
textfile_list = []

for folder in folder_list:
    text_files = os.listdir(folder)
    for text_file in text_files:
            textfile_list.append( os.path.join(folder, text_file))

a = 0
for i in textfile_list:
    with open(i, 'r', encoding = "utf8") as f:
        lines = f.read()
        # print(lines)
        a+=1

print(a)