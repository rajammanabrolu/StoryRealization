import ast
from tqdm import tqdm
import time

fh = open('scifi-events-fullSent-nameChange.txt')
lines = fh.readlines()
fh.close()

with open('outfile.txt', 'w') as outf:
    for line in tqdm(lines):
        parts = line.strip().split(';')
        if len(parts) == 3:
            events, generalized_events, sentence = parts
        else:
            continue
        events = ast.literal_eval(events.replace("u'", "'"))


        #for event in events[0]:
        outf.write(event +'\t'+sentence + '\n')
        outf.write()
        
