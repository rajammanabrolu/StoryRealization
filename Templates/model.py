import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        print ('ntoken: ' + str(ntoken))
        print ('ninp: ' + str(ninp))
        print ('nhid: ' + str(nhid))
        print ('nlayers: ' + str(nlayers))
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        print ('making model')
        #assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'BiLSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid//2 if l != nlayers - 1 else (ninp//2 if tie_weights else nhid//2), 1, dropout=0, bidirectional=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'BiGRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0, bidirectional=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)


        self.decoder_fw = nn.Linear(nhid, ntoken)
        self.decoder_bw = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_fw.weight = self.encoder.weight
            self.decoder_bw.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder_fw.bias.data.fill_(0)
        self.decoder_fw.weight.data.uniform_(-initrange, initrange)
        self.decoder_bw.bias.data.fill_(0)
        self.decoder_bw.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        #print (output.view(output.size(0), output.size(1), output.size(2)).shape)

        output_fw = output.view(output.size(0), output.size(1), output.size(2))[:,:,:400]
        output_bw = output.view(output.size(0), output.size(1), output.size(2))[:,:,400:]


        #print (output_fw.view(output_fw.size(0)*output_fw.size(1), output_fw.size(2)).shape)
        #print (output_bw.view(output_bw.size(0)*output_bw.size(1), output_bw.size(2)).shape)

        decoded_fw = self.decoder_fw(output_fw.view(output_fw.size(0)*output_fw.size(1), output_fw.size(2)))
        decoded_bw = self.decoder_bw(output_bw.view(output_bw.size(0)*output_bw.size(1), output_bw.size(2)))

        result_fw = decoded_fw.view(output_fw.size(0)*output_fw.size(1), decoded_fw.size(1))
        result_bw = decoded_bw.view(output_bw.size(0)*output_bw.size(1), decoded_bw.size(1))
        #result = decoded.view(output.size(0)*output.size(1), decoded.size(1))

        #result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            #return result, hidden, raw_outputs, outputs
            #return result, hidden, decode_fw, decode_bw
            return result_fw, result_bw, hidden, raw_outputs, outputs
        return result_fw, result_bw, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        #print('==rnn type {}'.format(self.rnn_type))
        #print(weight)
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'BiGRU':
            return [weight.new(2, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
        else: #BiLSTM
            return [(weight.new(1, bsz, self.nhid//2 if l != self.nlayers - 1 else (self.ninp//2 if self.tie_weights else self.nhid//2)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp//2 if self.tie_weights else self.nhid//2)).zero_())
                    for l in range(self.nlayers)]
