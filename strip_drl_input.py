
import sys
##usage: python abstract_dataset.py input.txt output.txt

fh = open(sys.argv[1], 'r')
fh2 = open(sys.argv[2], 'w')

lines = fh.readlines()
fh.close()


for line in lines:
    if '----------' not in line:
        fh2.write(line.replace('starting event: ', ''))
fh2.close()