lines = []
ground = []
with open("test_output.txt") as f:
    lines = f.readlines()
with open("../full_data/all-sci-fi-data-test_output.txt") as f:
    ground = f.readlines()
arglines = [i for i in range(len(lines)) if lines[i] != "<pad>\n"]
new_lines = [lines[i] for i in arglines]
new_ground = [ground[i] for i in arglines]
with open("test_output_unpadded.txt", "w") as fo:
    fo.writelines(new_lines)
with open("ground_test_unpadded.txt", "w") as fo:
    fo.writelines(new_ground)