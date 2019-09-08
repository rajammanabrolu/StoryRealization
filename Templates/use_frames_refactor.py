import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn

import data
import ast
import time
from tqdm import tqdm
import spacy
import re

from TahaManipulateState import ManipulateState

all_losses = []
nlp = spacy.load('en_core_web_sm')

dont_end_list = ['ADP', 'DET', 'CCONJ', 'PUNCT']


parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')
parser.add_argument('--data', type=str, default='./data/gen_scifi',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./gen_full_2.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')  
parser.add_argument('--seed', type=int, default=111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--input_event_file', type=str, default='event_data/5arg_train_input.txt',
                    help='input events')
parser.add_argument('--sample', action='store_true',
                    help='sample instead of chosing top chose next/back word')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
args = parser.parse_args()


print ("Initializing corpus...")
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()
print ("Corpus initialized!")

def isNoun(word):
    return (("Synset" in word and word.split('.')[1]=='n') or nlp(word)[0].pos_ == "NOUN" or re.match('<.*?>.*', word) != None)

def sample_for_word(word_weights, topn=40, sample=True):
    if sample:
        sample_weights, sample_idx = torch.topk(word_weights, topn)
        s_idx = torch.multinomial(sample_weights, 1)[0]
        word_idx = sample_idx[s_idx]
    else:
        _, word_idx = torch.max(word_weights, 0)
    next_word = corpus.dictionary.idx2word[word_idx]
    return next_word, word_idx


#returns predicted beam in specified direction
def predict_beam(next_word, model, forwards=True, maxlen=20, noun=True, ending=False, sample=True):
    new_beam = []
    hidden = model.init_hidden(1)

    total_loss = 0.0
    keep_going = True
    last_was_noun = noun
    last_word = next_word
    last_pos = "NOUN"
    for i in range(maxlen):
        if not keep_going:
            return new_beam, total_loss
        if next_word in corpus.dictionary.word2idx:
            word_input = Variable(torch.tensor([[corpus.dictionary.word2idx[next_word]]]).long())
        else:
            word_input = Variable(torch.rand(1, 1).mul(ntokens).long())

        if args.cuda:
            word_input.data = word_input.data.cuda()
        if forwards:
            model_output, _, hidden, rnn_hs, dropped_rnn_hs = model(word_input, hidden, return_h=True)
        else:
            _, model_output, hidden, rnn_hs, dropped_rnn_hs = model(word_input, hidden, return_h=True)
        
        
        loss = sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])



        word_weights = model_output.squeeze().data.div(args.temperature).exp().cpu()
        next_word, word_idx = sample_for_word(word_weights, 60, sample)

            #or (nlp(next_word)[0].pos_ == last_pos) \
        while next_word == last_word or next_word == '<eos>' \
            or (nlp(next_word)[0].pos_ == "PUNCT") \
            or (last_was_noun and isNoun(next_word)) \
            or (ending and i == maxlen - 1 and (nlp(next_word)[0].pos_ in dont_end_list)) \
            or (forwards == False and nlp(next_word)[0].pos_ == "DET" and not last_was_noun) \
            or (forwards and last_pos == "DET" and not isNoun(next_word)) \
            or (ending and i == maxlen - 1 and (nlp(next_word)[0].pos_ == "VERB" and (last_pos in dont_end_list or len(new_beam) == 0)))\
            or sample == False and next_word == 'to':

            # print ('******************************************')
            #print ('reject: ' + next_word)
            # print ('no "s" (punct): ' + str(nlp(next_word)[0].pos_ == "PUNCT"))
            # print ('no noun then noun: ' + str(last_was_noun and isNoun(next_word)))
            # #print (nlp(next_word)[0].pos_ == last_pos)
            # print ('no ending and end list: ' + str(ending and i == maxlen - 1 and (nlp(next_word)[0].pos_ in dont_end_list)))
            # print ('no noun then det: ' + str(forwards == False and nlp(next_word)[0].pos_ == "DET" and not last_was_noun))
            # print ('determiner and then not noun: ' +str(forwards and last_pos == "DET" and not isNoun(next_word)))
            # print ('dont end on bad word: ' + str(ending and i == maxlen - 1 and (nlp(next_word)[0].pos_ == "VERB" and (last_pos in dont_end_list or len(new_beam) == 0))))


            
            word_weights[word_idx] = 0
            next_word, word_idx = sample_for_word(word_weights, 60, sample)
            
        if ("Synset" in next_word or nlp(next_word)[0].pos_ == "NOUN"): last_was_noun = True
        word_input = Variable(torch.tensor([[corpus.dictionary.word2idx[next_word]]]).long())
        try:
            doc = nlp(next_word)
        except:
            keep_going = False
            continue
        #print (next_word + ": " + doc[0].pos_)
        if doc[0].pos_ == "VERB" or '-' in next_word or (forwards == False and (last_pos == "DET" or last_pos == "ADP") and isNoun(next_word)): #check if verb
            keep_going = False
            continue
        if forwards:
            new_beam.append(next_word)
        else:
            new_beam.insert(0, next_word)

        ##stopping conditions for backwards
        if forwards==False and (doc[0].pos_ == "DET"):
            keep_going = False
        


        if last_was_noun: last_was_noun = False
        last_pos = doc[0].pos_
        if "Synset" in next_word or last_pos == "NOUN" or re.match('<.*?>.*', next_word):
            last_pos = "NOUN"
            last_was_noun = True
        last_word = next_word
        total_loss += loss
    if len(new_beam)==0: return ([''], 0)
    return new_beam, total_loss
        



def main(args):
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
    if args.cuda:
        model.cuda()
    else:
        model.cpu()


    # np_lm.cpu()
    print ('ARE WE SAMPLING: ' + str(args.sample))

    hidden = model.init_hidden(1)
    print (model)

    input_file = open(args.input_event_file, 'r')
    lines = input_file.readlines()
    input_file.close()

    lines = lines[:30]

    with open(args.outf, 'w') as outf:
        #outfile = open('to_grammar_corrector.txt', 'w')
        for line in tqdm(lines):
            # THIS PART IS ONLY FOR scifi-events-fullSent-nameChange.txt
            # parts = line.strip().split(';')
            # if len(parts) == 3:
            #     events, generalized_events, sentence = parts
            # else:
            #     continue
            # generalized_events = ast.literal_eval(generalized_events.replace("u'", "'"))

            # REGULAR EVENT FILES:
            generalized_events = [line.strip().split(' ')]


            for event in generalized_events:
                cur_state = ManipulateState(event, [])
                #print (event)
                FRAME = cur_state.getFramePOS(event)
                sent = []
                total_loss = 0.0
                size = 0.0
                if FRAME == None or FRAME == [] or args.sample == False:
                    #do vanilla lm stuff
                    if event[0] != "EmptyParameter":
                        if re.match('<.*?>.*', event[0]):
                            sent.append(event[0])
                        else:
                            pre, loss = predict_beam(event[0], model, forwards=False, maxlen=1, ending=False, sample=False)
                            sent.append(pre[0])
                            total_loss += loss
                            size += 1
                            sent.append(event[0])

                    if event[1] != "EmptyParameter":
                        sent.append(event[1])
                    
                    if event[2] != "EmptyParameter":
                        if re.match('<.*?>.*', event[2]):
                            sent.append(event[2])
                        else:
                            pre, loss = predict_beam(event[2], model, forwards=False, maxlen=1, ending=False, sample=False)
                            sent.append(pre[0])
                            total_loss += loss
                            size += 1
                            sent.append(event[2])

                    if event[4] != "EmptyParameter":
                        sent.append(event[4])

                    if event[3] != "EmptyParameter":
                        if re.match('<.*?>.*', event[3]):
                            sent.append(event[3])
                        else:
                            pre, loss = predict_beam(event[3], model, forwards=False, maxlen=1, ending=False, sample=False)
                            if pre[0] != event[4]:
                                sent.append(pre[0])
                                total_loss += loss
                                size += 1
                            sent.append(event[3])
                    if (size == 0):
                        outf.write(' '.join(sent) + ';' + str(event) + ';loss:' + str(1.0)+ '\n')
                        all_losses.append(1.0)
                    else:
                        outf.write(' '.join(sent) + ';' + str(event) + ';loss:' + str(total_loss.cpu().data.numpy()/size)+ '\n')
                        all_losses.append(total_loss.cpu().data.numpy()/size)
                    continue
                #print (FRAME)
                num_nps = 0
                for pos in FRAME:
                    beam = []
                    
                    
                    if pos == 'NP' or pos == 'S':
                        if num_nps == 0:
                            word = event[0]
                        elif num_nps != 0 or pos == 'S':
                            word = event[2]
                        if word == "EmptyParameter":
                            continue
                        num_nps += 1
                        beam.append(word)
                        pre, loss = predict_beam(word, model, forwards=False, maxlen=3)
                        beam = pre + beam
                        total_loss += loss
                        size += len(pre)
                        if num_nps > 0:
                            pre, loss = predict_beam(word, model, forwards=True, maxlen=3, ending=True)
                        else:
                            pre, loss = predict_beam(word, model, forwards=True, maxlen=3)
                        beam = beam + pre
                        total_loss += loss
                        size += len(pre)

                    elif pos == 'V':
                        word = event[1]
                        if word == "EmptyParameter":
                            continue
                        beam.append(word)

                    elif pos == 'PP':
                        if event[4] != "EmptyParameter":
                            beam.append(event[4])
                        if event[3] != "EmptyParameter":
                            beam.append(event[3])
                        
                    sent.extend(beam)


                #print (sent)
                #rint (' '.join(sent))
                #print ('*******************************************')
                if (size == 0):
                    outf.write(' '.join(sent) + ';' + str(event) + ';' + str(FRAME) + ';loss:' + str(1.0) + '\n')
                    all_losses.append(1.0)
                else:
                    outf.write(' '.join(sent) + ';' + str(event) + ';' + str(FRAME) + ';loss:' + str(total_loss.cpu().data.numpy()/size) + '\n')
                    all_losses.append(total_loss.cpu().data.numpy()/size)
                

        import matplotlib.pyplot as plt

        n, bins, patches = plt.hist(all_losses, facecolor='blue', alpha=0.5)
        plt.show()

if __name__ == '__main__':
    main(args)
        