import sys
fh = open(sys.argv[1])
lines = fh.readlines()
fh.close()

dicto = {}
dicto['MCTS:'] = 0
dicto['RETEDIT:'] = 0
dicto['TEMPLATES:'] = 0
dicto['FSM:'] = 0
dicto['VANILLA:'] = 0

for line in lines:
    parts = line.split(' ')
    dicto[parts[0]] += 1

total = 0.0
for x in dicto:
    total += dicto[x]
for x in dicto:
    print (x[:-1].lower() + ": {:0.2f}%".format((dicto[x]/total)*100))