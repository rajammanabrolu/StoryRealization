fh = open('full_data/stripped-mini-discover-ACL-epoch4.txt')
lines = fh.readlines()
for line in lines:
    if len(line.split(' ')) != 5:
        print line

fh.close()