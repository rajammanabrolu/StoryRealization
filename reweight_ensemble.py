

#more_verbose_outf.write('Events\tTemplates\tTemplate Score\tRetEdit\tRetEdit Score\tMonte Carlo\tMcts score\tFSM\tVanilla\n')

# fh = open('drl_even_more_verbose_output.txt')
# verbose_outf = open('verbose_output.txt', 'w')
# sent_outf = open('new_output.txt', 'w')
import sys
fh = open(sys.argv[1])
verbose_outf = open(sys.argv[2].split('.')[0] + '_verbose.' + sys.argv[2].split('.')[1], 'w')
sent_outf = open(sys.argv[2], 'w')

retedit_edit_dist = []
retedit_sents = []
template_scores = []
template_sents = []
mcts_scores = []
mcts_sents = []
fsm_scores = []
fsm_sents = []
vanilla_sents = []

lines = fh.read().splitlines()[1:]
for line in lines:
    parts = line.split('\t')
    retedit_edit_dist.append(float(parts[4]))
    retedit_sents.append(parts[3])
    template_scores.append(float(parts[2]))
    template_sents.append(parts[1])
    mcts_scores.append(float(parts[6]))
    mcts_sents.append(parts[5])
    if parts[7] == '<pad>':
        fsm_scores.append(0)
    else:
        fsm_scores.append(1)
    fsm_sents.append(parts[7])
    vanilla_sents.append(parts[8])
for i in range(len(lines)):
    if float(retedit_edit_dist[i]) < 0.1:
        #print ("RETEDIT: " + retedit_sents[i])
        verbose_outf.write("RETEDIT: " + retedit_sents[i] + '\n')
        sent_outf.write(retedit_sents[i] + '\n')
    else:
        if template_scores[i] < 0.3:
            #print ("TEMPLATES: " + template_sents[i])
            verbose_outf.write("TEMPLATES: " + template_sents[i] + '\n')
            sent_outf.write(template_sents[i] + '\n')
        else:
            if mcts_scores[i] > 0.2:
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