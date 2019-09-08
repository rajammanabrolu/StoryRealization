lines = []
with open("test_output_unpadded.txt") as f:
    lines = f.readlines()
totallen = 0
for line in lines:
    words = line.split()
    totallen += len(words)
print(float(totallen) / float(len(lines)))