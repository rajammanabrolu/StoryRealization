"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, rev_vocab_src, rev_vocab, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.rev_vocab_src = rev_vocab_src
        self.rev_vocab = rev_vocab # idx to word
        self.rev_vocab[self.pad] = '<pad>'
        self.rev_vocab[self.bos] = '<s>'
        self.rev_vocab[self.eos] = '</s>'
        self.verb = False
        self.tt = torch.cuda if cuda else torch

	

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def reweight_beam(self, bestScores, bestScoresId, tempYs, evt_tokens, weights):
        #getting root word of verbnet category. i.e. own-100 -> own
        evt_tokens[2] = evt_tokens[2].split('-')[0]
	# loop through each beam, convert that beam's predicted word from an index to a word, then increment the best score
        # if word appears in events
        for i in range(self.size):
            curr_word = self.rev_vocab[tempYs[i].item()]
            if curr_word in evt_tokens: bestScores[i] += weights[evt_tokens.index(curr_word)]
	# return list of highest scoring words, sorted in descending order
	retScores, tempId = torch.sort(bestScores, 0, True)
        newIds = torch.LongTensor([bestScoresId[i] for i in tempId]).cuda()
        return retScores, newIds


    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk, inp_evt):
        """Advance the beam."""
        num_words = workd_lk.size(1)
        diversity_rate = 1.75
        # Convert event to corresponding event tokens using self.rev_vocab_src
        inp_evt = inp_evt.data.cpu().numpy().tolist()
        evt_tokens = [self.rev_vocab_src[inp_evt[i]] for i in range(len(inp_evt))]
	# Weighting of event components -> #TODO move away from this
        weights = [0, 0.125, 0.25, 0.125, 0.05, 0]

        # Sum the previous scores.
        if len(self.prevKs) > 0:
	    # sum word probabilities with scores (vectorized) to get beam_lk
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
 ########  change here  ###################################################
            for i in range(self.size):
                row = beam_lk[i] # gather all scores corresponding to beam i
                ranked_scores,rankings = row.topk(self.size,0,True,True) # get top K scores corresponding to this beam
                for j in range(self.size):
                    if j== 0:
                        continue
                    beam_lk[i][rankings[j]] -= diversity_rate*j 
        else:
            beam_lk = workd_lk[0]
        flat_beam_lk = beam_lk.view(-1) # flatten the beams

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True) # get top K scores, one for each beam
        bestScoresCache = bestScores
        tempK = bestScoresId / num_words # find which beam the top outputs came from
        tempYs = bestScoresId - tempK * num_words #  find outputs corresponding to the top K scores
        bestScores, bestScoresId = self.reweight_beam(bestScores, bestScoresId, tempYs, evt_tokens, weights) # 'reweight' the beam
        '''
        print(bestScores, bestScoresId / num_words)
        tmpk = bestScoresId / num_words
        tmpy = bestScoresId - tmpk * num_words
        for i in range(self.size):
            print(bestScores[i])
            print(self.rev_vocab[tmpy[i]])
            print('----')
        '''
        self.scores = bestScores # update top scores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k) # update backpointers, that is, which beam the word came from
        self.nextYs.append(bestScoresId - prev_k * num_words) # update current outputs
        #print(self.prevKs, self.nextYs, len(self.prevKs), len(self.nextYs))
        # End condition is when top-of-beam is EOS.
        '''
        for i in range(self.nextYs.size()[0]):
            for ys in self.nextYs[i]:
                if is_verb(ys[0]):
                    self.verb = True
        #'''
        '''
	
        _, ks = self.sort_best() 
        hyps = zip(*(self.get_hyp(k) for k in ks[:self.size]))
        hyp_inds = [x[0] for x in hyps]
        print(hyp_inds)
        pred = ' '.join([self.rev_vocab[x] for x in hyp_inds])
        print(pred)
        #'''
        if self.nextYs[-1][0] == self.eos:
            #print(self.scores[1])
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            #print(j, k)
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]
