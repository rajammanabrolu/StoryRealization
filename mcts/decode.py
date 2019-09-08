"Decode Seq2Seq model with beam search."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import Seq2Seq, Seq2SeqAttention
from data_utils import read_nmt_data, get_minibatch, read_config
from new_monte_carlo import MonteCarlo
from evaluate import get_bleu
import argparse
import os

class BeamSearchDecoder(object):
    """Beam Search decoder."""

    def __init__(
        self,
        config,
        model_weights,
        src,
        trg,
        beam_size=5
    ):
        """Initialize model."""
        self.config = config
        self.model_weights = model_weights
        self.beam_size = beam_size

        self.src = src
        self.trg = trg
        self.src_dict = src['word2id']
        self.id2word_src = src['id2word']
        self.tgt_dict = trg['word2id']
        self._load_model()

    def _load_model(self):
        print 'Loading pretrained model'
        if self.config['model']['seq2seq'] == 'vanilla':
            print 'Loading Seq2Seq Vanilla model'

            self.model = Seq2Seq(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.,
            ).cuda()

        elif self.config['model']['seq2seq'] == 'attention':
            print 'Loading Seq2Seq Attention model'

            self.model = Seq2SeqAttention(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                ctx_hidden_dim=self.config['model']['dim'],
                attention_mode='dot',
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.,
            ).cuda()

        self.model.load_state_dict(torch.load(
            open(self.model_weights)
        ))
        print "Model loaded"

    def get_hidden_representation(self, input):
        """Get hidden representation for a sentence."""
        src_emb = self.model.src_embedding(input)
        h0_encoder, c0_encoder = self.model.get_state(src_emb)
        src_h, (src_h_t, src_c_t) = self.model.encoder(
            src_emb, (h0_encoder, c0_encoder)
        )

        if self.model.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        return src_h, (h_t, c_t)

    def get_init_state_decoder(self, input):
        """Get init state for decoder."""
        decoder_init_state = nn.Tanh()(self.model.encoder2decoder(input))
        return decoder_init_state

    def decode_batch(self, idx):
        """Decode a minibatch."""
        # Get source minibatch
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            self.src['data'], self.src_dict, idx,
            self.config['data']['batch_size'],
            self.config['data']['max_src_length'], add_start=True, add_end=True
        )
        #print(self.src_dict)
        '''
        lines = [
                ['<s>'] + line + ['</s>']
                for line in self.src['data'][idx:idx + self.config['data']['max_src_length']]
                ]
        lines = [line[:self.config['data']['max_src_length']] for line in lines]
        lens = [len(line) for line in lines]
        max_len = max(lens)
        word2ind = self.src_dict
        input_lines = [
                [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
                [word2ind['<pad>']] * (max_len - len(line))
                for line in lines
                ]
        #print(len(input_lines))
        #print(input_lines_src[0])
        '''
        #id2word_src = {v: k for k, v in self.src_dict.iteritems()}
        #inp = input_lines_src[0].data.cpu().numpy().tolist()
        #print([inv_dict[a] for a in inp])
        beam_size = self.beam_size

        #  (1) run the encoder on the src

        context_h, (context_h_t, context_c_t) = self.get_hidden_representation(
            input_lines_src
        )

        context_h = context_h.transpose(0, 1)  # Make things sequence first.

        #  (3) run the decoder to generate sentences, using beam search

        batch_size = context_h.size(1)

        # Expand tensors for each beam.
        context = Variable(context_h.data.repeat(1, beam_size, 1))
        #print context.size()
        dec_states = [
            Variable(context_h_t.data.repeat(1, beam_size, 1)),
            Variable(context_c_t.data.repeat(1, beam_size, 1))
        ]

        beam = [
            MonteCarlo(beam_size, self.tgt_dict, self.id2word_src, trg['id2word'],self.config['data']['max_trg_length'], self.model, cuda=True)
            for k in range(batch_size)
        ]

        dec_out = self.get_init_state_decoder(dec_states[0].squeeze(0))
        dec_states[0] = dec_out
        #print(dec_states[0].size())

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.config['data']['max_trg_length']):
            #print(i)
            input = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)

            trg_emb = self.model.trg_embedding(Variable(input).transpose(1, 0))
            #print trg_emb.size()
            #print dec_states[0].size(), dec_states[1].size()
            #print context.size()
            trg_h, (trg_h_t, trg_c_t) = self.model.decoder(
                trg_emb,
                (dec_states[0].squeeze(0), dec_states[1].squeeze(0)),
                context
            )

            dec_states = (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = trg_h_t.squeeze(1).view(-1, self.model.trg_hidden_dim)
            #print dec_out.size()
            out = F.softmax(self.model.decoder2vocab(dec_out)).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            cur = 0
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                #print(idx, len(lines), input_lines_src.size())
                if not beam[b].advance(word_lk.data[idx], input_lines_src[idx], [dec_states[0][:,cur*self.beam_size:(cur+1)*self.beam_size,:],dec_states[1][:,cur*self.beam_size:(cur+1)*self.beam_size,:]], context[:,cur*self.beam_size:(cur+1)*self.beam_size,:]):
                    active += [b]
                    cur += 1

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    #print dec_state.size(1), dec_state.size(2), dec_state.size(3)
                    state_size = dec_state.size(1) * dec_state.size(3) if self.model.nlayers_trg > 1 else dec_state.size(2)
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, state_size
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.model.decoder.hidden_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            context = update_active(context)

            remaining_sents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            #print(ks)

            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            #print "Element in batch " + str(hyps)
            #print(hyps)
            allHyp += [hyps]

        return allHyp, allScores

    def translate(self):
        """Translate the whole dataset."""
        trg_preds = []
        trg_gold = []
        for j in xrange(
            0, len(self.src['data']),
            self.config['data']['batch_size']
        ):
            """Decode a single minibatch."""
            #print 'Decoding %d out of %d ' % (j, len(self.src['data']))
            hypotheses, scores = decoder.decode_batch(j)
            #print "Here's what I get back from decode_batch " + str(hypotheses)

            all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
            #print "All hyps is " + str(all_hyp_inds)
            all_preds = [
                ' '.join([trg['id2word'][x] for x in hyp])
                for hyp in all_hyp_inds
            ]

            # Get target minibatch
            # input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
            #    get_minibatch(
            #        self.trg['data'], self.tgt_dict, j,
            #        self.config['data']['batch_size'],
            #        self.config['data']['max_trg_length'],
            #        add_start=True, add_end=True
            #    )
            #)

            #output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
            #all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
            #all_gold = [
            #    ' '.join([trg['id2word'][x] for x in hyp])
            #    for hyp in all_gold_inds
            #]

            trg_preds += all_preds
            #trg_gold += all_gold
	    #with open("e2s-monte-carlo-output.txt", "a") as f:
	    for p, score in zip(all_preds, scores):
                print(p.replace('<s>', '').replace('</s>', '').strip())
		print(score[0].item())
	    #	    f.write('\n')
	    #with open("e2s-monte-carlo-score.txt", "a") as f:
            #for score in scores:
	    #	    f.write(str(score[0]))
	    #	    f.write('\n')
            #all_preds = ' '.join(all_preds)#.replace('</s>', '|')
            #all_gold = ' '.join(all_gold)
            #print len(all_preds.split('</s>'))
            #print all_preds
            #print 'BLEU : %.5f ' % (get_bleu(all_preds, all_gold))
            #for pred, truth in zip(all_preds.split('</s>'), all_gold.split('</s>')):
            #    print pred#' '.join(pred)
                #print "Truth: ", truth
        #trg_preds = ' '.join(trg_preds).split('</s>')
        #print len(
        #for pred in trg_preds:
        #    print pred
        #bleu_score = get_bleu(trg_preds, trg_gold)
        #print 'BLEU : %.5f ' % (bleu_score)


class GreedyDecoder(object):
    """Beam Search decoder."""

    def __init__(
        self,
        config,
        model_weights,
        src,
        trg,
        beam_size=1
    ):
        """Initialize model."""
        self.config = config
        self.model_weights = model_weights
        self.beam_size = beam_size

        self.src = src
        self.trg = trg
        self.src_dict = src['word2id']
        self.tgt_dict = trg['word2id']
        self._load_model()

    def _load_model(self):
        print 'Loading pretrained model'
        if self.config['model']['seq2seq'] == 'vanilla':
            print 'Loading Seq2Seq Vanilla model'

            self.model = Seq2Seq(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.,
            ).cuda()

        elif self.config['model']['seq2seq'] == 'attention':
            print 'Loading Seq2Seq Attention model'

            self.model = Seq2SeqAttention(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                ctx_hidden_dim=self.config['model']['dim'],
                attention_mode='dot',
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.,
            ).cuda()

        self.model.load_state_dict(torch.load(
            open(self.model_weights)
        ))

    def decode_minibatch(
        self,
        input_lines_src,
        input_lines_trg,
        output_lines_trg_gold
    ):
        """Decode a minibatch."""
        for i in xrange(self.config['data']['max_trg_length']):

            decoder_logit = self.model(input_lines_src, input_lines_trg)
            word_probs = self.model.decode(decoder_logit)
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[:, -1])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(1)),
                1
            )

        return input_lines_trg

    def translate(self):
        """Evaluate model."""
        preds = []
        ground_truths = []
        for j in xrange(
            0, len(self.src['data']),
            self.config['data']['batch_size']
        ):

            #print 'Decoding : %d out of %d ' % (j, len(self.src['data']))
            # Get source minibatch
            input_lines_src, output_lines_src, lens_src, mask_src = (
                get_minibatch(
                    self.src['data'], self.src['word2id'], j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_src_length'],
                    add_start=True, add_end=True
                )
            )

            input_lines_src = Variable(input_lines_src.data, volatile=True)
            output_lines_src = Variable(output_lines_src.data, volatile=True)
            mask_src = Variable(mask_src.data, volatile=True)

            # Get target minibatch
            input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
                get_minibatch(
                    self.trg['data'], self.trg['word2id'], j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_trg_length'],
                    add_start=True, add_end=True
                )
            )

            input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
            output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)
            mask_src = Variable(mask_src.data, volatile=True)

            # Initialize target with <s> for every sentence
            input_lines_trg = Variable(torch.LongTensor(
                [
                    [trg['word2id']['<s>']]
                    for i in xrange(input_lines_src.size(0))
                ]
            ), volatile=True).cuda()

            # Decode a minibatch greedily add beam search decoding
            input_lines_trg = self.decode_minibatch(
                input_lines_src, input_lines_trg,
                output_lines_trg_gold
            )

            # Copy minibatch outputs to cpu and convert ids to words
            input_lines_trg = input_lines_trg.data.cpu().numpy()
            input_lines_trg = [
                [self.trg['id2word'][x] for x in line]
                for line in input_lines_trg
            ]

            # Do the same for gold sentences
            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
            output_lines_trg_gold = [
                [self.trg['id2word'][x] for x in line]
                for line in output_lines_trg_gold
            ]

            # Process outputs
            for sentence_pred, sentence_real, sentence_real_src in zip(
                input_lines_trg,
                output_lines_trg_gold,
                output_lines_src
            ):
                if '</s>' in sentence_pred:
                    index = sentence_pred.index('</s>')
                else:
                    index = len(sentence_pred)
                preds.append(['<s>'] + sentence_pred[:index + 1])

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                else:
                    index = len(sentence_real)

                ground_truths.append(['<s>'] + sentence_real[:index + 1])
                #trg_preds += preds
                #trg_gold += all_gold
                for p in preds:
                    print " ".join(p).replace('</s>', '').strip()
                #TODO fix sentence markers
                #all_preds = ' '.join(preds)#.replace('</s>', '|')
                #all_gold = ' '.join(ground_truths)
                #print 'BLEU : %.5f ' % (get_bleu(all_preds, all_gold))
                #for pred, truth in zip(all_preds.split('</s>'), all_gold.split('</s>')):
                #    print pred#' '.join(pred)
                    #print "Truth: ", truth

        #bleu_score = get_bleu(preds, ground_truths)

        #for pred, truth in zip(preds, ground_truths):
        #    print 'Pred:' + ' '.join([a for a in pred])
        #    print 'Truth:' + ' '.join([a for a in ground_truths])

        #print 'BLEU score : %.5f ' % (bleu_score)

def pipeline_predict(src, config, trg=None):
    config = read_config(config)
    model_weights = os.path.join(config['data']['save_dir'], config['data']['preload_weights'])

    src, trg = read_data_pipeline(src, config)
    src_test = {}
    trg_test = {}

    src_test['word2id'] = src['word2id']
    src_test['id2word'] = src['id2word']

    trg_test['word2id'] = trg['word2id']
    trg_test['id2word'] = trg['id2word']

    if config['model']['decode'] == "beam_search":
        decoder = BeamSearchDecoder(config, model_weights, src_test, trg_test, config['model']['beam_size'])
        decoder.translate()
    else:
        decoder = GreedyDecoder(config, model_weights, src_test, trg_test)
        decoder.translate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to json config",
        required=True
    )
    args = parser.parse_args()
    config = read_config(args.config)
    model_weights = os.path.join(config['data']['save_dir'], config['data']['preload_weights'])

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

    if trg_test is None:
	trg_test = {}

    src_test['word2id'] = src['word2id']
    src_test['id2word'] = src['id2word']

    trg_test['word2id'] = trg['word2id']
    trg_test['id2word'] = trg['id2word']

    if config['model']['decode'] == "beam_search":
        decoder = BeamSearchDecoder(config, model_weights, src_test, trg_test, config['model']['beam_size'])
        decoder.translate()
    else:
        decoder = GreedyDecoder(config, model_weights, src_test, trg_test)
        decoder.translate()
    '''
    allHyp, allScores = decoder.decode_batch(0)
    all_hyp_inds = [[x[0] for x in hyp] for hyp in allHyp]
    all_preds = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_hyp_inds]

    input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
        get_minibatch(
            trg['data'], trg['word2id'], 0,
            80,
            50,
            add_start=True, add_end=True
        )
    )

    output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
    all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
    all_gold = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_gold_inds]

    for hyp, gt in zip(all_preds, all_gold):
        print hyp, len(hyp.split())
        print '-------------------------------------------------'
        print gt
        print '================================================='
    '''
