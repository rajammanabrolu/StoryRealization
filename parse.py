import random
import sys

full_data = sys.argv[1]
with open(full_data) as f:
    data = f.readlines()
    
    
data = [d for d in data if (not d.startswith("%") and not d.startswith("<EOS>"))]

def parse_and_write(d):
    base = full_data[:full_data.rfind(".")]
    with open(base + "_input.txt", "w") as input_data, open(base + "_output.txt", "w") as output_data:
        for line in d:
            _, event, _, sentence = line.split("|||")
	    event = " ".join(eval(event)[0])
            input_data.write(event + "\n")
            output_data.write(sentence)
            
parse_and_write(data)
