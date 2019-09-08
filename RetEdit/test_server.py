def edit(events):
    
    x = "with enraged yells <PRP> repeatedly throws Synset('entity.n.01') at <ORGANIZATION>8 Synset('natural_phenomenon.n.01') that seals the Synset('action.n.01') startling <PERSON>14 the Synset('defender.n.01') on Synset('group_action.n.01')".split()
    processedEvents = [EditExample([[event[0]], [event[1]], event[2:]],x) for event in events]
    
    print(processedEvents[0])
    
    valid_eval = ret_model.ret_and_make_ex(processedEvents, new_lsh, examples.train, 0, train_mode=False)
    beam_list, edit_traces = edit_model.edit(valid_eval)

    # base retriever.
    import gtd.retrieval_func as rf
    lsh, dict = rf.make_hash(examples.train)
    output_index = rf.grab_nbs(processedEvents, lsh, dict)
    ret_pred = rf.generate_predictions(examples.train, output_index)

    ####
    # eval code
    gen_out = []
    for i in range(len(edit_traces)):
        gen = beam_list[i][0]
        gen_out.append(gen)
        
    dist = []
    prob = []

    for i in range(len(edit_traces)):
        dist.append(str(valid_eval[i].dist))
        prob.append(str(edit_traces[i].decoder_trace.candidates[0].prob))
    
    output = {'output':gen_out, 'distances':dist, 'beamProb': prob}
    print(output)
    return output
    
    
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

config = Config.from_file('./editor_code/configs/editor/scifi_5tuple_gen.txt')
src_dir = os.environ['COPY_EDIT_DATA']+'/scifi_runs/final'  #for codalab

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

edit([["<PRP>","throw-17.1-1-1","<PRP>","Synset('natural_phenomenon.n.01')","at"], ["<PERSON>0","run-51.3.2-2-1", "declare-29.4","Synset('social_group.n.01')","to"]])
