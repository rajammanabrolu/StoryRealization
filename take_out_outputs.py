import sys
fh = open(sys.argv[1])

mcts_outf_sents = open('MCTS' + sys.argv[2], 'w')
vanilla_outf_sents = open('VANILLA' + sys.argv[2], 'w')
templates_outf_sents = open('TEMPLATES' + sys.argv[2], 'w')
fsm_outf_sents = open('FSM' + sys.argv[2], 'w')
retedit_outf_sents = open('RETEDIT' + sys.argv[2], 'w')

retedit_sents = []
template_sents = []
mcts_sents = []
fsm_sents = []
vanilla_sents = []

lines = fh.read().splitlines()[1:]
for line in lines:
    parts = line.split('\t')
    retedit_sents.append(parts[3])
    template_sents.append(parts[1])
    mcts_sents.append(parts[5])
    fsm_sents.append(parts[7])
    vanilla_sents.append(parts[8])


mcts_outf_sents.close()
vanilla_outf_sents.close()
templates_outf_sents.close()
fsm_outf_sents.close()
retedit_outf_sents.close()
    #mcts_scores.append(float(parts[6]))
    #mcts_sents.append(parts[5])