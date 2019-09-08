"Decode Seq2Seq model with beam search."""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
from model_lib import FSMBeamSearchDecoder, MCTSBeamSearchDecoder, BeamSearchDecoder
from TemplateDecoder import TemplateDecoder
import argparse
import os
import json


import ast
import time
from tqdm import tqdm
import re
import requests


import threading


from mcts.model import Seq2Seq, Seq2SeqAttention
from mcts.data_utils import read_nmt_data, get_minibatch, read_config
from mcts.monte_carlo import MonteCarlo
from mcts.evaluate import get_bleu
from FSM.beam_search_fsm import Beam
from FSM.constrained_beam_search import FSMBeamSearch



retedit_return_sents = []
retedit_edit_dist = []
retedit_beamprobs = []


mcts_sents = []
mcts_scores = []

fsm_sents = []
fsm_scores = []

template_sents = []
template_scores = []

retedit_sents = []
retedit_scores = []





parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')
parser.add_argument("--config", help="path to json config", required=True)
parser.add_argument('--data', type=str, default='./Templates/data/new_abstracted_scifi/',
                help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./Templates/abs_gen5.pt',
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
parser.add_argument('--input_event_file', type=str, default='./full_data/all-sci-fi-data-train_input.txt',
                help='input events')
# parser.add_argument('--input_event_file', type=str, default='event_data/fullGenE2S-singleEvent-NewVerb-Test_input.txt',
#                     help='input events')
parser.add_argument('--sample', action='store_true',
                help='sample instead of chosing top chose next/back word')
parser.add_argument('--alpha', type=float, default=2,
                help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
args = parser.parse_args()



config = read_config(args.config)
mcts_model_weights = os.path.join('mcts/' + config['data']['save_dir'], config['data']['preload_weights'])



src, trg = read_nmt_data(
    src=config['data']['src'],
    config=config,
    trg=config['data']['trg']
)

src_test, trg_test = read_nmt_data(
    src=config['data']['test_src'],
    config=config,
    trg=None #trg=config['data']['test_trg']
)

def mcts_thread(decoder):
    print ("RUNNING ON MCTS")
    decoder.translate()
    print ('DONE RUNNING ON MCTS')
    return

def fsm_thread(decoder):
    print ("RUNING ON FSM")
    decoder.translate()
    print('DONE RUNNING ON FSM')
    return

def vanilla_thread(decoder):
    print ("RUNING ON vaNILLA")
    decoder.translate()
    print('DONE RUNNING ON VANILLA')
    return

def templates_thread(input_event_file):
    print ("RUNING ON TEMPLATES")
    template_decoder.template_main(input_event_file)
    print('DONE RUNNING ON TEMPLATES')
    return

def retedit_thread(events):
    print ('GETTING RETEDIT')
    global retedit_return_sents
    global retedit_edit_dist
    global retedit_beamprobs
    retedit_return_sents, retedit_edit_dist, retedit_beamprobs = json.loads((requests.get('http://127.0.0.1:8080', data={'events':events}).text))
    #sent_fh = open('all-sci-fi-test-output.txt')
    #scores_fh = open('all-sci-fi-test-dist.txt')
    # retedit_return_sents = sent_fh.read().splitlines()
    # retedit_edit_dist = [float(x) for x in scores_fh.read().split(',')]
    
    
    print('DONE RUNNING ON RETEDIT')
    return



if trg_test is None: trg_test = {}

src_test['word2id'] = src['word2id']
src_test['id2word'] = src['id2word']

# print (src_test['data'])

trg_test['word2id'] = trg['word2id']
trg_test['id2word'] = trg['id2word']


if config['model']['decode'] == "beam_search":
    mcts_decoder = MCTSBeamSearchDecoder(config, mcts_model_weights, src_test, trg_test, config['model']['beam_size'])
    #mcts_decoder.translate()
    mcts_threadd = threading.Thread(target=mcts_thread, args=(mcts_decoder,))
    mcts_threadd.start()
    print ('thread mcts running on mcts')
else:
    print ("stop using greedy")
    mcts_decoder = MCTSGreedyDecoder(config, mcts_model_weights, src_test, trg_test)
    mcts_decoder.translate()

# print (allHyp)
# print (allScores)

# #****************************************************

fsm_model_weights = os.path.join('FSM/' + config['data']['save_dir'], config['data']['preload_weights'])

if config['model']['decode'] == "beam_search":
    fsm_decoder = FSMBeamSearchDecoder(config, fsm_model_weights, src_test, trg_test, config['model']['beam_size'])
    fsm_threadd = threading.Thread(target=fsm_thread, args=(fsm_decoder,))
    fsm_threadd.start()
    print ('thread fsm running on FSM')
else:
    print ("stop using greedy")
    fsm_decoder = FSMGreedyDecoder(config, fsm_model_weights, src_test, trg_test)
    fsm_decoder.translate()


#********************************************************
vanilla_model_weights = os.path.join('mcts/' + config['data']['save_dir'], config['data']['preload_weights'])

if config['model']['decode'] == "beam_search":
    vanilla_decoder = BeamSearchDecoder(config, vanilla_model_weights, src_test, trg_test, config['model']['beam_size'])
    vanilla_threadd = threading.Thread(target=vanilla_thread, args=(vanilla_decoder,))
    vanilla_threadd.start()
    print ('thread fsm running on FSM')
else:
    print ('literally stop')
    vanilla_decoder = GreedyDecoder(config, vanilla_model_weights, src_test, trg_test)
    vanilla_decoder.translate()



#****************************************************retedit
#print (src_test)
# input: events (string) = 'a1 b1 c1 d1 e1, a2 b2...'
# output: (list[list[string]]) = [output, editDistances, beamProbs] 
# def retedit(events):
#     return json.loads((requests.get('http://127.0.0.1:8080', data={'events':events}).text))
events = ', '.join([' '.join(i) for i in src_test['data']])
# #print (retedit(events))

# retedit_thread = threading.Thread(target=retedit_thread, args=(events,))
# retedit_return_sents, retedit_edit_dist, retedit_beamprobs = retedit(events)
# retedit_return_sents, retedit_edit_dist, retedit_beamprobs = json.loads((requests.get('http://127.0.0.1:8080', data={'events':events}).text))

retedit_threadd = threading.Thread(target=retedit_thread, args=(events,))
retedit_threadd.start()
print ('thread retedit running on retedit')

print ("RUNNING ON TEMPLATES")
template_decoder = TemplateDecoder(args)
template_threadd = threading.Thread(target=templates_thread, args=(config['data']['test_src'],))
template_threadd.start()

# print (template_sents)
# print (template_scores)


print ('waiting for templates to finish')
template_threadd.join()
print ('waiting for vanilla to finish')
vanilla_threadd.join()
print ('waiting for mcts to finish')
mcts_threadd.join()
print ('waiting for fsm to finish')
fsm_threadd.join()
print ('waiting for retedit to finish')
retedit_threadd.join()

retedit_sents = [' '.join(x) for x in retedit_return_sents]
# retedit_sents = retedit_return_sents

mcts_sents = mcts_decoder.get_all_sents()
mcts_scores = mcts_decoder.get_all_scores()
#print (mcts_sents)
#print (mcts_scores)

fsm_sents = fsm_decoder.get_all_sents()
fsm_score_output = fsm_decoder.get_all_scores()
#print (fsm_sents)

for s in fsm_sents:
    if s == '<pad>':
        fsm_scores.append(0)
    else:
        fsm_scores.append(1)
#print (fsm_scores)

vanilla_sents = vanilla_decoder.get_all_sents()

template_sents = template_decoder.get_all_sents()
template_scores = template_decoder.get_all_scores()
#print (vanilla_sents)


#waterfall
print ("*************************OUTPUT TIME***********************************")
verbose_outf = open(args.outf.split('.')[0] + '_verbose.' + args.outf.split('.')[1], 'w')
more_verbose_outf = open(args.outf.split('.')[0] + '_more_verbose.tsv', 'w')
sent_outf = open(args.outf, 'w')
more_verbose_outf.write('Events\tTemplates\tTemplate Score\tRetEdit\tRetEdit Score\tMonte Carlo\tMcts score\tFSM\tVanilla\n')


for i in range(len(template_sents)): 
    more_verbose_outf.write(' '.join(src_test['data'][i])+'\t' +\
        template_sents[i] + '\t' + str(template_scores[i]) + '\t'+ \
        retedit_sents[i] + '\t' + str(retedit_edit_dist[i]) + '\t'+ \
        mcts_sents[i] + '\t' + str(mcts_scores[i].data.cpu().numpy()[0]) + '\t'+ \
        fsm_sents[i] + '\t'+ \
        vanilla_sents[i] + '\n')



    if float(retedit_edit_dist[i]) < 0.2:
        #print ("RETEDIT: " + retedit_sents[i])
        verbose_outf.write("RETEDIT: " + retedit_sents[i] + '\n')
        sent_outf.write(retedit_sents[i] + '\n')
    else:
        if template_scores[i] < 0.2:
            #print ("TEMPLATES: " + template_sents[i])
            verbose_outf.write("TEMPLATES: " + template_sents[i] + '\n')
            sent_outf.write(template_sents[i] + '\n')
        else:
            if mcts_scores[i].data.cpu().numpy()[0] > 0.1:
                #print ("MCTS: " + mcts_sents[i])
                verbose_outf.write("MCTS: " + mcts_sents[i] + '\n')
                sent_outf.write(mcts_sents[i] + '\n')

            else:
                if fsm_scores[i] == 1:
                    #print ("FSM: " + fsm_sents[i])
                    verbose_outf.write("FSM: " + fsm_sents[i] + '\n')
                    sent_outf.write(fsm_sents[i] + '\n')
                else:
                    #print ("VANILLA: " + vanilla_sents[i])
                    verbose_outf.write("VANILLA: " + vanilla_sents[i] + '\n')
                    sent_outf.write(vanilla_sents[i] + '\n')
verbose_outf.close()
sent_outf.close()





        