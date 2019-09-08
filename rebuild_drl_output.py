import sys
##usage: python abstract_dataset.py input.txt output.txt

fh = open('full_data/mini-discover-ACL-epoch3-take2-only50.txt')
fh2 = open('drl2_ensemble_output.txt')

input_stories = fh.readlines()
output_lines = fh2.readlines()
fh.close()
fh2.close()

fh3 = open('paper_output/mini-discover-ACL-epoch3-take2-only50-ensemble-output.txt', 'w')


output_idx = 0
for i in range(len(input_stories)):
    if '----------' not in input_stories[i]:
        fh3.write(output_lines[output_idx])
        output_idx += 1
    else:
        fh3.write('-----------------------------------------------\n')
fh3.close()
