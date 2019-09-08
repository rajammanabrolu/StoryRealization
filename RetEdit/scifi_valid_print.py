# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
import paths
import os

#os.environ['COPY_EDIT_DATA'] = paths.data_dir
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_output_encoding(encoding='utf-8'):
    import sys
    import codecs
    '''When piping to the terminal, python knows the encoding needed, and
       sets it automatically. But when piping to another program (for example,
       | less), python can not check the output encoding. In that case, it 
       is None. What I am doing here is to catch this situation for both 
       stdout and stderr and force the encoding'''
    current = sys.stdout.encoding
    if current is None :
        sys.stdout = codecs.getwriter(encoding)(sys.stdout)
    current = sys.stderr.encoding
    if current is None :
        sys.stderr = codecs.getwriter(encoding)(sys.stderr)

#Note - we need this or else the program crashes due to a utf-8 error when trying to pipe the outputs to a text file.
# set_output_encoding()

from gtd.utils import Config

from editor_code.copy_editor.edit_training_run import EditTrainingRun
from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRun
from editor_code.copy_editor.editor import EditExample
from editor_code.copy_editor.vocab import HardCopyDynamicVocab

from gtd.utils import bleu

print os.environ['COPY_EDIT_DATA']

# no-profile
profile = False

# config = Config.from_file('./editor_code/configs/editor/old/scifi_200d.txt')
# src_dir = os.environ['COPY_EDIT_DATA']+'/scifi_runs/200d' #for codalab

config = Config.from_file('./editor_code/configs/editor/all-sci-fi-data.100d.txt')
src_dir = os.environ['COPY_EDIT_DATA']+'/scifi_runs/final-all'  #for codalab

#config = Config.from_file('./editor_code/configs/editor/scifi_5tuple.txt')
#src_dir = os.environ['COPY_EDIT_DATA']+'/scifi_runs/5tuple' #for codalab

load_expt = RetrieveEditTrainingRun(config, src_dir)

###
# retedit model
import numpy as np

ret_model = load_expt.editor.ret_model
edit_model = load_expt.editor.edit_model
examples = load_expt._examples

from gtd.utils import chunks
from tqdm import tqdm

new_vecs = []
for batch in tqdm(chunks(examples.train,32), total=len(examples.train)/32):
    encin = ret_model.encode(batch, train_mode=False).data.cpu().numpy()
    for vec in encin:
        new_vecs.append(vec)
    del encin

new_lsh = ret_model.make_lsh(new_vecs)

# eval_num = len(examples.test)
eval_num = 10
# print(examples.test[0].target_words)
# print(dir(examples.test[0].target_words))

valid_eval = ret_model.ret_and_make_ex(examples.test[:eval_num], new_lsh, examples.train, 0, train_mode=False)
beam_list, edit_traces = edit_model.edit(valid_eval)

# base retriever.
import gtd.retrieval_func as rf
lsh, dict = rf.make_hash(examples.train)
output_index = rf.grab_nbs(examples.test[:eval_num], lsh, dict)
ret_pred = rf.generate_predictions(examples.train, output_index)

####
# eval code
gen_out = []
gen2_out = []
ret_trout = []
ret_fix_out = []
for i in range(len(edit_traces)):
    ret_fix_out.append(ret_pred[i])
    ret_trout.append(edit_traces[i].example.input_words[6])
    gen = beam_list[i][0]
    gen_out.append(gen)
    

f = open('scifi_5tuple.txt', 'w')
g = open('scifi_5tuple_truth.txt', 'w')
e = open('scifi_5tuple_examples.txt', 'w')

d = open('dist.txt', 'w')
p = open('prob.txt', 'w')
d_ex = open('dist_ex.txt', 'w')
p_ex = open('prob_ex.txt', 'w')

dist = []
prob = []

dist_ex = []
prob_ex = []

for i in range(len(edit_traces)):
    output = ""
    output += str(examples.test[i])+ '\n'
    output += "RET:\t" + ' '.join(ret_trout[i])  + '\n'
    output += "EDIT:\t" + ' '.join(gen_out[i]) + "\n"
    output += "RETDIST:\t" + str(valid_eval[i].dist)+ "\n"
    output += "BEAM PROB:\t" + str(edit_traces[i].decoder_trace.candidates[0].prob)+ "\n\n"

    g.write(' '.join(examples.test[i].target_words) + "\n")
    f.write(' '.join(gen_out[i]) + "\n")
    e.write(output)

    dist.append(str(valid_eval[i].dist))
    prob.append(str(edit_traces[i].decoder_trace.candidates[0].prob))

    dist_ex.append((valid_eval[i].dist,examples.test[i], gen_out[i]))
    prob_ex.append((edit_traces[i].decoder_trace.candidates[0].prob,examples.test[i], gen_out[i]))

dist_ex.sort()
prob_ex.sort()


d.write(",".join(dist))
p.write(",".join(prob))

d_ex.write(",".join([str(x) for x in dist_ex]))
p_ex.write(",".join([str(x) for x in prob_ex]))