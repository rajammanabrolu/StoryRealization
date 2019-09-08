from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np

class MonteCarloTreeNode(object):
    def __init__(self, tt, model, id, word, length, max_trg_length, parent, target, children = [], score = 1, prev_sentence = [], prev_id = [], done = False):
        self.id = id
        self.word = word
        self.length = length
        self.parent = parent
        self.children = children
        self.score = score
        self.sentence = prev_sentence + [word]
        self.sentence_id = prev_id + [id]
        self.reference_sentence = target
        self.max_trg_length = max_trg_length
        self.tt = tt
        self.model = model
    	self.done = done

    def playout(self, num_playouts, dec_states, context, rev_vocab, eos):
        cur_word = self.id
        cur_sentence = self.sentence[:]
        level = self.length
        if self.done:
            return self.score
        sum_score = 0
        for _ in range(num_playouts):
            cur_word = self.id
            cur_sentence = self.sentence[:]
            cur_level = self.length
            cur_dec_states = dec_states
            cur_context = context
            while cur_level < self.max_trg_length:
                input = self.tt.LongTensor(
                    [[cur_word]]
                ).t().contiguous().view(1, -1)

                trg_emb = self.model.trg_embedding(Variable(input).transpose(1, 0))
                #print trg_emb.size()
                #print dec_states[0].size(), dec_states[1].size()
                #print context.size()
                trg_h, (trg_h_t, trg_c_t) = self.model.decoder(
                trg_emb,
                (cur_dec_states[0].squeeze(0), cur_dec_states[1].squeeze(0)),
                cur_context
                )

                cur_dec_states = (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

                dec_out = trg_h_t.squeeze(1).view(-1, self.model.trg_hidden_dim)
                #print dec_out.size()
                out = F.softmax(self.model.decoder2vocab(dec_out)).unsqueeze(0)
                word_lk = out.view(
                -1
                )#.transpose(0, 1).contiguous()

                next_word = torch.multinomial(word_lk.data, 1)[0]


                cur_word = next_word.item()
                cur_sentence.append(rev_vocab[cur_word])

                if next_word == eos:
                    #self.score *= 10
                    break
                cur_level += 1

            sum_score += self.evaluate(self.reference_sentence, cur_sentence)

    	decay = 0.75
    	self.score = self.score * decay + (sum_score / num_playouts) * (1 - decay)
        # self.score *= self.evaluate(self.reference_sentence, cur_sentence)
        return self.score

    def evaluate(self, references, hypotheses, weights = [0.25,0.25,0.25,0.25,0.25]):
    	#weights = [0.15, 0.6, 0.15, 0.1]

        # the fixed weight after training
    	weights = np.array([0.2, 0.51, 0.19, 0.07, 0.03])#np.load("/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/ASTER/E2S-Ensemble/mcts/5tuple-weights.npy")
#np.array([0.2, 0.5, 0.15, 0.075, 0.075]) # np.load("/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/ASTER/Event2Sentence/5tuple-weights.npy")
        references = [r for r in references if r != "<s>"]
        hypotheses = [h for h in hypotheses if h != "<s>" and h!= "</s>"]
        #if len(references) == 6:
            #references = references[1:]
        references[1] = references[1].split('-')[0]
    	mask = [1.0 if ref != "EmptyParameter" and ref != "<s>" else 0.0 for ref in references]
        #print np.shape(references)
        #print references
        #print hypotheses
        #print np.shape(weights)
        #print np.shape(mask)
        weights *= np.array(mask)
        weights /= weights.sum()

        # third argument is how much 1 gram, 2 gram ... are weighted
        base_score = corpus_bleu([references], [hypotheses], weights)

        # calculate one gram score for each of the item in input event
        one_gram_score = 0
        count = 0
        for ref, weight in zip(references, weights):
            if weight != 0:
                one_gram_score += weight * corpus_bleu([ref], [hypotheses], [weight])
                count += 1

        score = (base_score + (one_gram_score / count)) / 2

        return score

class MonteCarlo(object):
    """
    ......
    add self.levels to __init__ to keep track of nodes at each level
    """

    def __init__(self, size, vocab, rev_vocab_src, rev_vocab, max_trg_length, model, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.rev_vocab_src = rev_vocab_src
        self.rev_vocab = rev_vocab
        self.rev_vocab[self.pad] = '<pad>'
        self.rev_vocab[self.bos] = '<s>'
        self.rev_vocab[self.eos] = '</s>'
        self.verb = False
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = [self.tt.FloatTensor(size).zero_()]

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

        self.model = model
        # beam at each level
        # change so that only one root
        self.levels = [[MonteCarloTreeNode(self.tt, self.model, self.bos, self.rev_vocab[self.bos], 0, max_trg_length, None, "", ) for _ in range(size)]]
        self.max_trg_length = max_trg_length
        # self.trg_sentence = trg_sentence


    def reweight_beam(self, bestScores, bestScoresId, tempYs, evt_tokens, weights):
        #print(bestScores)
        #getting root word of verbnet category, potential mod to check for everything in category
        evt_tokens[2] = evt_tokens[2].split('-')[0]
        for i in range(self.size):
            curr_word = self.rev_vocab[tempYs[i]]
            if curr_word in evt_tokens:
                bestScores[i] += weights[evt_tokens.index(curr_word)]
        retScores, tempId = torch.sort(bestScores, 0, True)
        newIds = torch.LongTensor([bestScoresId[i] for i in tempId]).cuda()
        return retScores, newIds

    def advance(self, workd_lk, inp_evt, dec_states, context):
        """Advance the beam.
        workd_lk is of size (beam_size, vocab_size)
                            (5, 60004)

        inp_evt example :
        ['<s>', "Synset('male.n.02')", 'wish-62', 'call', 'EmptyParameter']
        evt_tokens later:
        Variable containing:
          0
          6
         29
          4
          4

        """
        num_words = workd_lk.size(1)
        diversity_rate = 1.75
        #print(list(inp_evt.size())[0])
        inp_evt = inp_evt.data.cpu().numpy().tolist()
        evt_tokens = [self.rev_vocab_src[inp_evt[i]] for i in range(len(inp_evt))]
        # weights = [0, 0.125, 0.25, 0.125, 0.05, 0]
        #workd_lk: the likelihood of words

        # Sum the previous scores.
        # if len(self.prevKs) > 0:
        #     beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        # ########  change here  ###################################################
        #     for i in range(self.size):
        #         row = beam_lk[i]
        #         ranked_scores,rankings = row.topk(self.size,0,True,True)
        #         for j in range(self.size):
        #             if j== 0:
        #                 continue
        #             beam_lk[i][rankings[j]] -= diversity_rate*j
        # else:
        #     beam_lk = workd_lk[0]
        beam_lk = workd_lk
        flat_beam_lk = beam_lk.view(-1)

        # expand self.size * 2 nodes and only pick top self.size based on the playout score
        largeBeamSize = self.size + 3
        bestScores, bestScoresId = flat_beam_lk.topk(largeBeamSize, 0, True, True)
        bestScoresCache = bestScores
        tempK = bestScoresId / num_words
        tempYs = bestScoresId - tempK * num_words

        MCNodes = []
        for score, id, prev in zip(bestScores, tempYs, tempK):
            parent_node = self.levels[-1][prev]
    	    #if parent_node was already done
    	    id = id.item()
    	    if parent_node.done:
    	    	new_node = parent_node
    	    else:
                #change score
                new_node = MonteCarloTreeNode(self.tt, self.model, id, self.rev_vocab[id], parent_node.length + 1, self.max_trg_length, parent_node, evt_tokens, [], parent_node.score, parent_node.sentence, parent_node.sentence_id, parent_node.done)
    	        #boost up score for complete sentence
                if new_node.id == self.eos:
                    new_node.done = True
    		        #new_node.score *= 10
            # parent_node.children.append(new_node)
            MCNodes.append(new_node)

        num_playouts = 3

        bestScores = self.tt.FloatTensor([MCNode.playout(num_playouts, [dec_states[0][:,tempK[i]:tempK[i]+1,:],dec_states[1][:,tempK[i]:tempK[i]+1,:]], context[:,tempK[i]:tempK[i]+1,:], self.rev_vocab, self.eos) for i, MCNode in enumerate(MCNodes)])
        beamBestScores, beamBestScoresId = bestScores.topk(self.size, 0, True, True)
        self.levels.append([])
        for id in beamBestScoresId:
            self.levels[-1].append(MCNodes[id])
        # self.levels.append(MCNodes)

        # bestScores, bestScoresId = self.reweight_beam(bestScores, bestScoresId, tempYs, evt_tokens, weights)
        '''
        print(bestScores, bestScoresId / num_words)
        tmpk = bestScoresId / num_words
        tmpy = bestScoresId - tmpk * num_words
        for i in range(self.size):
            print(bestScores[i])
            print(self.rev_vocab[tmpy[i]])
            print('----')
        '''
        self.scores = beamBestScores
        # changed here
        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        #print(bestScoresId)
        bestScoresId = bestScoresId[beamBestScoresId]
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

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
    	maxScoreIndex = torch.sort(self.scores, 0, True)[1][0]
    	self.done = self.levels[-1][maxScoreIndex].done
        if self.done:
            #self.reweight(evt_tokens[:], self.levels[-1][maxScoreIndex].sentence)
            pass
    	#if self.done:
    	    #print(self.levels[-1][maxScoreIndex].sentence)
        #if self.nextYs[-1][0] == self.eos:
            #print(self.scores[1])
         #   self.done = True

        return self.done

    def reweight(self, event, generated):
        decay = 0.99
        old_weight = np.load("/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/ASTER/E2S-Ensemble/mcts/5tuple-weights.npy")
        boost = np.zeros(old_weight.shape)
        indices = []
        event[2] = event[2].split('-')[0]

        for i in range(1, 6):
            #skip the first one which is "<s>"
            if event[i] != "EmptyParameter" and event[i] not in generated:
                indices.append(i-1)

        if not indices:
            indices = list(range(5))

        boost[indices] = boost[indices] + ((1 - decay) * old_weight).sum() / len(indices)
        new_weight = decay * old_weight + boost
        np.save("/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/ASTER/E2S-Ensemble/mcts/5tuple-weights.npy", new_weight)

    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def sort_best(self):
        """Sort the beam."""
        scores = [node.score for node in self.levels[-1]]
        return torch.sort(self.tt.FloatTensor(scores), 0, True)

    def get_hyp(self, k):
        """Get hypotheses."""
	    #print(self.levels[-1][k].sentence)
        return self.levels[-1][k].sentence_id
