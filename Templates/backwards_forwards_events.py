import argparse

import torch
from torch.autograd import Variable

import data
import ast
import time
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/scifi',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./bigru_4.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')  
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--input_event_file', type=str, default='event_data/scifi-events-fullSent-nameChange.txt',
                    help='input events')
parser.add_argument('--sample', action='store_true',
                    help='sample instead of chosing top chose next/back word')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

print ("reloading model")
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location='cpu')[0]
print ("model successfully loaded")
model.eval()
if args.model == 'QRNN':
    model.reset()

model.cpu()
print ('ARE WE SAMPLING: ' + str(args.sample))
print ("initializing corpus")
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
print ('ntokens: ' + str(ntokens))
hidden = model.init_hidden(1)
# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
print ("finished initialzing corpus")


# print(input)
# if args.cuda:
#     input.data = input.data.cuda()
print (model)

input_file = open(args.input_event_file, 'r')
lines = input_file.readlines()
input_file.close()

with open(args.outf, 'w') as outf:
    outfile = open('to_grammar_corrector.txt', 'w')
    for line in tqdm(lines):
    #for line in (lines):
        parts = line.strip().split(';')
        if len(parts) == 3:
            events, generalized_events, sentence = parts
        else:
            continue
        events = ast.literal_eval(events.replace("u'", "'"))
        

        for event in events:
            sent = []
            m_phrase = []
            # doc = nlp(unicode(' '.join(event), 'utf-8'))
            # for z in range(5):
            #     print event[z] + ': ' + doc[z].pos_
            for i, word in enumerate(event):
                orig_word = word
                try:
                    doc = nlp(orig_word)
                except:
                    doc = nlp('cat')
                #print orig_word + ': ' + doc[0].pos_

                if word == 'EmptyParameter':
                    continue

                word = word.lower()
                if word in corpus.dictionary.word2idx:
                    input = Variable(torch.tensor([[corpus.dictionary.word2idx[word]]]).long(), volatile=True)
                    #print ('real: ' + str(input))
                else:

                    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
                    #word = corpus.dictionary.idx2word[input.data.numpy()[0][0]]
                    #print ("rando generate: " + str(input))
                
                output_forwards, output_backwards, _ = model(input, hidden, return_h=False)
                



                word_weights = output_forwards.squeeze().data.div(args.temperature).exp().cpu()
                #print (word_weights)
                if args.sample:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                else:
                    _, word_idx = torch.max(word_weights, 0)
                #print ('next idx: ' + str(word_idx))
                forw_word = corpus.dictionary.idx2word[word_idx]
                while forw_word == '<eos>':
                    word_weights[word_idx] = 0
                    _, word_idx = torch.max(word_weights, 0)
                    forw_word = corpus.dictionary.idx2word[word_idx]
                #print ("forwards word: " + forw_word)

                word_weights = output_backwards.squeeze().data.div(args.temperature).exp().cpu()
                if args.sample:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                else:
                    _, word_idx = torch.max(word_weights, 0)
                #print ('back idx: ' + str(word_idx))
                back_word = corpus.dictionary.idx2word[word_idx]
                while back_word == '<eos>':
                    word_weights[word_idx] = 0
                    _, word_idx = torch.max(word_weights, 0)
                    back_word = corpus.dictionary.idx2word[word_idx]
                #print ("backwards word: " + back_word)
                #print ('\n')

                
                del input, word_weights

                if i == 0:
                    if orig_word[0] == orig_word[0].upper() or doc[0].pos_ == 'PROPN' or doc[0].pos_ == 'PRON':
                        sent.append(orig_word)
                    else:
                        sent.append(back_word)
                        sent.append(orig_word)
                elif i == 2:
                    if doc[0].pos_ == 'PROPN' or doc[0].pos_ == 'PRON':
                        sent.append(orig_word)
                    else:
                        sent.append(back_word)
                        sent.append(orig_word)

                    # sent.append(back_word)
                    # sent.append(orig_word)
                    #sent.append(forw_word)
                elif i == 1 or i == 4:
                    sent.append(orig_word)
                elif i == 3:
                    if doc[0].pos_ == 'PROPN' or doc[0].pos_ == 'PRON':
                        m_phrase.append(orig_word)
                    else:
                        m_phrase.append(back_word)
                        m_phrase.append(orig_word)
                    # m_phrase.append(back_word)
                    # m_phrase.append(orig_word)

                
                #outf.write(back_word + ' ' + word + ' ' + forw_word + ' ')
                # print ('backward: ' + str(back_word))
                # print ('current word: ' + str(word))
                # print ('foreward: ' + str(forw_word))
            sent.extend(m_phrase)
            outf.write(' '.join(sent))
            outfile.write(' '.join(sent) + '\n')
            outf.write('; ' + str(event) + '\n')
