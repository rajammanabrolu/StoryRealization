import sys
fh = open(sys.argv[1])
lines = fh.readlines()
fh.close()


totallen = 0.0
for line in lines:
    words = line.strip().split(' ')
    totallen += len(words)
print(float(totallen) / float(len(lines)))