import os
import torch
import hashlib
import spacy
from Templates.TahaManipulateState import ManipulateState
from Templates import data
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import re



class TemplateDecoder(object):

    def __init__(self, args):

        self.args = args
        print ("Initializing corpus...")

        self.fn = 'corpus.{}.data'.format(hashlib.md5(self.args.data.encode()).hexdigest())
        if os.path.exists(self.fn):
            print('Loading cached dataset...')
            self.corpus = torch.load(self.fn)
        else:
            print('Producing dataset...')
            self.corpus = data.Corpus(self.args.data)
            torch.save(self.corpus, self.fn)

        self.real_losses = []
        self.nlp = spacy.load('en_core_web_sm')

        self.dont_end_list = ['ADP', 'DET', 'CCONJ', 'PUNCT']

        self.ntokens = len(self.corpus.dictionary)
        self.criterion = nn.CrossEntropyLoss()
        print ("Corpus initialized!")
        self.template_sents = []
        self.template_scores = []
    
    def isNoun(self, word):
        return (("Synset" in word and word.split('.')[1]=='n') or self.nlp(unicode(word))[0].pos_ == "NOUN" or re.match('<.*?>.*', word) != None)

    def sample_for_word(self, word_weights, topn=40, sample=True):
        if sample:
            sample_weights, sample_idx = torch.topk(word_weights, topn)
            s_idx = torch.multinomial(sample_weights, 1)[0]
            word_idx = sample_idx[s_idx]
        else:
            _, word_idx = torch.max(word_weights, 0)
        next_word = self.corpus.dictionary.idx2word[word_idx]
        return next_word, word_idx

    #returns predicted beam in specified direction
    def predict_beam(self, next_word, model, forwards=True, maxlen=20, noun=True, ending=False, sample=True):
        new_beam = []
        hidden = model.init_hidden(1)

        total_loss = 0.0
        keep_going = True
        last_was_noun = noun
        last_word = next_word
        last_pos = "NOUN"
        got_conj = False
        for i in range(maxlen):
            if not keep_going:
                return new_beam, total_loss
            if next_word in self.corpus.dictionary.word2idx:
                word_input = Variable(torch.tensor([[self.corpus.dictionary.word2idx[next_word]]]).long())
            else:
                word_input = Variable(torch.rand(1, 1).mul(self.ntokens).long())

            if self.args.cuda:
                word_input.data = word_input.data.cuda()
            if forwards:
                model_output, _, hidden, rnn_hs, dropped_rnn_hs = model(word_input, hidden, return_h=True)
            else:
                _, model_output, hidden, rnn_hs, dropped_rnn_hs = model(word_input, hidden, return_h=True)
            
            
            loss = sum(self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])



            word_weights = model_output.squeeze().data.div(self.args.temperature).exp().cpu()
            next_word, word_idx = self.sample_for_word(word_weights, 60, sample)

                #or (self.nlp(next_word)[0].pos_ == last_pos) \
            while next_word == last_word \
                or (self.nlp(unicode(next_word))[0].pos_ == "PUNCT") \
                or (last_was_noun and self.isNoun(next_word)) \
                or (ending and i == maxlen - 1 and (self.nlp(unicode(next_word))[0].pos_ in self.dont_end_list)) \
                or (forwards == False and self.nlp(unicode(next_word))[0].pos_ == "DET" and not last_was_noun) \
                or (forwards and last_pos == "DET" and not self.isNoun(next_word)) \
                or (ending and i == maxlen - 1 and (self.nlp(unicode(next_word))[0].pos_ == "VERB" and (last_pos in self.dont_end_list or len(new_beam) == 0)))\
                or sample == False and next_word == 'to':

                # print ('******************************************')
                #print ('reject: ' + next_word)
                # print ('no "s" (punct): ' + str(self.nlp(next_word)[0].pos_ == "PUNCT"))
                # print ('no noun then noun: ' + str(last_was_noun and isNoun(next_word)))
                # #print (self.nlp(next_word)[0].pos_ == last_pos)
                # print ('no ending and end list: ' + str(ending and i == maxlen - 1 and (self.nlp(next_word)[0].pos_ in dont_end_list)))
                # print ('no noun then det: ' + str(forwards == False and self.nlp(next_word)[0].pos_ == "DET" and not last_was_noun))
                # print ('determiner and then not noun: ' +str(forwards and last_pos == "DET" and not isNoun(next_word)))
                # print ('dont end on bad word: ' + str(ending and i == maxlen - 1 and (self.nlp(next_word)[0].pos_ == "VERB" and (last_pos in dont_end_list or len(new_beam) == 0))))

                if next_word == '<eos>':
                    keep_going = False
                    break
                
                word_weights[word_idx] = 0
                next_word, word_idx = self.sample_for_word(word_weights, 60, sample)
                
            if ("Synset" in next_word or self.nlp(unicode(next_word))[0].pos_ == "NOUN"): last_was_noun = True
            word_input = Variable(torch.tensor([[self.corpus.dictionary.word2idx[next_word]]]).long())
            try:
                doc = self.nlp(unicode(next_word))
            except:
                keep_going = False
                continue
            #print (next_word + ": " + doc[0].pos_)
            if next_word == '<eos>' or (doc[0].pos_ == "CCONJ" and got_conj) or doc[0].pos_ == "VERB" or '-' in next_word or (forwards == False and (last_pos == "DET" or last_pos == "ADP") and self.isNoun(next_word)): #check if verb
                keep_going = False
                continue
            if doc[0].pos_ == "CCONJ":
                got_conj = True
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
            total_loss += loss.cpu().data.numpy()
        if len(new_beam)==0: return ([], 0)
        return new_beam, total_loss

    def template_main(self, input_event_file):

        max_loss = .11
        all_losses = []
        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(self.args.seed)

        if self.args.temperature < 1e-3:
            parser.error("--temperature has to be greater or equal 1e-3")

        print ("reloading model")
        with open(self.args.checkpoint, 'rb') as f:
            model = torch.load(f)[0]

        print ("model successfully loaded")
        model.eval()
        if self.args.cuda:
            model.cuda()
        else:
            model.cpu()


        # np_lm.cpu()
        print ('ARE WE SAMPLING: ' + str(self.args.sample))

        hidden = model.init_hidden(1)


        input_file = open(input_event_file, 'r')
        lines = input_file.readlines()
        input_file.close()

        with open(self.args.outf, 'w') as outf:
            #outfile = open('to_grammar_corrector.txt', 'w')
            for line in tqdm(lines):
                # THIS PART IS ONLY FOR scifi-events-fullSent-nameChange.txt
                # parts = line.strip().split(';')
                # if len(parts) == 3:
                #     events, generalized_events, sentence = parts
                # else:
                #     continue
                # generalized_events = ast.literal_eval(generalized_events.replace("u'", "'"))

                # # THIS PART IS ONLY FOR corrected-5tuple-full-genEvent-genSent.txt
                # parts = line.strip().split('|')
                # if len(parts) == 3:
                #     generalized_events, gen_sent, sentence = parts
                # else:
                #     continue
                # generalized_events = ast.literal_eval(generalized_events)

                #REGULAR EVENT FILES:
                generalized_events = [line.strip().split(' ')]


                for event in generalized_events:
                    cur_state = ManipulateState(event, [])
                    #print (event)
                    FRAME = cur_state.getFramePOS(event)
                    sent = []
                    total_loss = 0.0
                    size = 0.0
                    if FRAME == None or FRAME == [] or self.args.sample == False:
                        #do vanilla lm stuff
                        if event[0] != "EmptyParameter":
                            if re.match('<.*?>.*', event[0]):
                                sent.append(event[0])
                            else:
                                pre, loss = self.predict_beam(event[0], model, forwards=False, maxlen=1, ending=False, sample=False)
                                sent.extend(pre)
                                total_loss += loss
                                size += 1
                                sent.append(event[0])

                        if event[1] != "EmptyParameter":
                            sent.append(event[1])
                        
                        if event[2] != "EmptyParameter":
                            if re.match('<.*?>.*', event[2]):
                                sent.append(event[2])
                            else:
                                pre, loss = self.predict_beam(event[2], model, forwards=False, maxlen=1, ending=False, sample=False)
                                sent.extend(pre)
                                total_loss += loss
                                size += 1
                                sent.append(event[2])

                        if event[4] != "EmptyParameter":
                            sent.append(event[4])

                        if event[3] != "EmptyParameter":
                            if re.match('<.*?>.*', event[3]):
                                sent.append(event[3])
                            else:
                                pre, loss = self.predict_beam(event[3], model, forwards=False, maxlen=1, ending=False, sample=False)
                                if len(pre) != 0 and pre[0] != event[4]:
                                    sent.extend(pre)
                                    total_loss += loss
                                    size += 1
                                sent.append(event[3])
                        if (size == 0):
                            #outf.write(' '.join(sent) + ';' + str(' '.join(event)) + ';loss:' + str(1)+ '\n')
                            self.template_sents.append(' '.join(sent))
                            self.template_scores.append(1)
                            all_losses.append(1)
                        else:
                            #outf.write(' '.join(sent) + ';' + str(' '.join(event)) + ';loss:' + str((total_loss/size)/max_loss)+ '\n')
                            self.template_sents.append(' '.join(sent))
                            self.template_scores.append((total_loss/size)/max_loss)
                            all_losses.append(total_loss/size)
                            self.real_losses.append(total_loss/size)
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
                            pre, loss = self.predict_beam(word, model, forwards=False, maxlen=3)
                            beam = pre + beam
                            total_loss += loss
                            size += len(pre)
                            # if num_nps > 0:
                            #     pre, loss = predict_beam(word, model, forwards=True, maxlen=3, ending=True)
                            # else:
                            #     pre, loss = predict_beam(word, model, forwards=True, maxlen=3)
                            # beam = beam + pre
                            # total_loss += loss
                            # size += len(pre)

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
                        #outf.write(' '.join(sent) + ';' + str(' '.join(event)) + ';' + str(FRAME) + ';loss:' + str(1) + '\n')
                        self.template_sents.append(' '.join(sent))
                        self.template_scores.append(1)
                        all_losses.append(1)
                    else:
                        #outf.write(' '.join(sent) + ';' + str(' '.join(event)) + ';' + str(FRAME) + ';loss:' + str((total_loss/size)/max_loss) + '\n')
                        self.template_sents.append(' '.join(sent))
                        self.template_scores.append((total_loss/size)/max_loss)
                        all_losses.append(total_loss/size)
                        self.real_losses.append(total_loss/size)
                    

            # import matplotlib.pyplot as plt
            # max_loss = max(real_losses)
            # print (max_loss)
            # all_losses = [i / max_loss if i is not 1 else i for i in all_losses ]
            # n, bins, patches = plt.hist(all_losses, facecolor='blue', alpha=0.5)
            # plt.show()

        return
    def get_all_sents(self):
        return self.template_sents

    def get_all_scores(self):
        return self.template_scores
        