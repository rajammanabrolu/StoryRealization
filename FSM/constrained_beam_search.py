from beam_search_fsm import Beam
import torch

class FSMBeamSearch(object):
    def __init__(self, evt_indices):
        constraints = evt_indices
        self.constraints = constraints
        self.size = 2**len(constraints)
        self.accept = False

        self.dec_states = [None] * self.size
        self.context = [None] * self.size
        self.evt_indices = evt_indices
        self.evt_tokens = None

        self.beams = [None] * self.size
        self.num_states = 0
        self.done = False


    def advance(self, word_lk, inp_evt, dec_states, context):
        cur = 0
        update_list = []
        for i in range(self.size):
            if self.beams[i] and not self.beams[i].done:
                beam_size = self.beams[i].size
                self.dec_states[i] = (dec_states[0][:,cur*beam_size:(cur+1)*beam_size,:].clone(),dec_states[1][:,cur*beam_size:(cur+1)*beam_size,:].clone())
                self.context[i] = context[:,cur*beam_size:(cur+1)*beam_size,:].clone()
                self.beams[i].advance(word_lk[cur:cur+1].squeeze(), inp_evt)

                if self.beams[i].done:
                    self.num_states -= 1
                    if i == self.size - 1:
                        self.accept = True

                update_list.append(i)
                cur += 1

        for i in update_list:
            self.transition(i)

        #update number of states
        count = 0
        for i in range(self.size):
            if self.beams[i] and not self.beams[i].done:
                count += 1
        self.num_states = count

        # check whether all monte carlo objects are done:
        # if so this fsm is done too
        for i in range(self.size):
            if self.beams[i] and not self.beams[i].done:
                break
        else:
            self.done = True
        return self.done

    def transition(self, i):
        beam = self.beams[i]
        for j, evt in enumerate(self.evt_tokens):
            for k, word in enumerate(beam.nextYs[-1]):
                if evt != "EmptyParameter" and evt != "<unk>" and self.evt_indices[j].data.item() == word:
                    self.update_state(k, i, j)


    def update_state(self, index, state, update_bit):
        if state & (1 << update_bit) == 1:
            # no need to update
            return
        else:
            new_state = state | (1 << update_bit)
            if self.beams[new_state] is None:
                self.num_states += 1

                self.beams[new_state] = Beam(self.beams[state].size, self.beams[state].vocab, self.beams[state].rev_vocab_src, self.beams[state].rev_vocab, self.beams[state].cuda)

                self.beams[new_state].scores = self.beams[state].scores.clone()
                self.beams[new_state].prevKs = [k.clone() for k in self.beams[state].prevKs]
                self.beams[new_state].nextYs = [y.clone() for y in self.beams[state].nextYs]

                # clone decoder states and context
                self.dec_states[new_state] = (self.dec_states[state][0].clone(), self.dec_states[state][1].clone())
                self.context[new_state] = self.context[state].clone()


    def get_current_state(self):
        current_states = []
        for i in range(self.size):
            if self.beams[i] and not self.beams[i].done:
                current_states.append(self.beams[i].get_current_state())
        return current_states

    def get_dec_states(self):
        dec_states0 = None
        dec_states1 = None

        for i in range(self.size):
            if self.dec_states[i] is not None and not self.beams[i].done:
                if dec_states0 is None:
                    dec_states0 = self.dec_states[i][0].clone()
                else:
                    dec_states0 = torch.cat((dec_states0, self.dec_states[i][0].clone()), 1)

                if dec_states1 is None:
                    dec_states1 = self.dec_states[i][1].clone()
                else:
                    dec_states1 = torch.cat((dec_states1, self.dec_states[i][1].clone()), 1)
        return (dec_states0, dec_states1)

    def get_context(self):
        context = None

        for i in range(self.size):
            if self.context[i] is not None  and not self.beams[i].done:
                if context is None:
                    context = self.context[i].clone()
                else:
                    # print(context.size(), self.states[i].context.size())
                    context = torch.cat((context, self.context[i].clone()), 1)

        return context

    def get_hyp(self):
        """Get hypotheses."""
        #order = ["11111"]
        order = ["11111", "01111", "10111", "11011", "11101", "11110", "00111", "10011", "11001", "11100", "01011", "10101", "11010", "01101", "10110", "01110"]
        #order = ["1111","0111","1011","0011","1101","1110"]
        #order = ["1111"]
        #order = ["1111","0111","1011","0011","1101","1110","1001","1010","0110","0101","0001","0010","1100","1000","0100","0000"]
        for b in order:
            state = int(b, 2)
            if self.beams[state] and self.beams[state].done:
                n_best = 1
                scores, ks = self.beams[state].sort_best()
                scores = scores[:n_best]
                hyps = zip(*[self.beams[state].get_hyp(k) for k in ks[:n_best]])
                break
        else:
            # for b in order:
            #     state = int(b, 2)
            #     if state in self.states:
            #         n_best = 1
            #         scores, ks = self.states[state].sort_best()
            #         scores = scores[:n_best]
            #         hyps = zip(*[self.states[state].get_hyp(k) for k in ks[:n_best]])
            #         break
            hyps = [tuple(torch.ones(1).type(torch.IntTensor))]
            #hyps = [[1,1,1,1,1,1,1,1,1,1]]
            scores = [1]
        return hyps, scores



# s = FSMBeamSearch(['a', 'b', 'c', 'd'])
# for state in s.states:
#     print(s.states[state].successors)
