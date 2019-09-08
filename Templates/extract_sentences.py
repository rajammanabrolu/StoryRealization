import time
import re
import sys
from tqdm import tqdm
from nltk.corpus import wordnet as wn

##usage: python extract_sents.py input.txt output.txt

fh = open(sys.argv[1], 'r')
fh2 = open(sys.argv[2], 'w')
lines = fh.readlines()

for i in tqdm(range(len(lines))):
    parts = lines[i].strip().split(';')
    fh2.write(parts[0] + '\n')

fh.close()
fh2.close()
        
        
        

