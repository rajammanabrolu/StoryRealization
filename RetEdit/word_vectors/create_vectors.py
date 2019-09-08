import numpy as np
import random

file = open("all-sci-fi-data.50d.txt", "w")

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def loadWords(storyFile):
    print("Loading sentences")
    f = open(storyFile,'r')
    model = {}
    for line in f:
        splitLine = line.strip().split()
        for word in splitLine:
            model[word] = 1
    return model


model = loadGloveModel("glove.6B.50d.txt")
words = loadWords("all-sci-fi-data.tsv")
vocab = words.keys()
print(len(vocab))
for w in vocab:
    if w not in model.keys():
        newVector =  w + ' ' + ' '.join([str(random.uniform(-1,1)) for i in range(len(model['unk']))]) + '\n'
        file.write(newVector)

    else:
        newVector =  w + ' ' + ' '.join([str(x) for x in model[w]]) + '\n'
        file.write(newVector)
