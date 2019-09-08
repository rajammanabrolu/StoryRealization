# Event-to-Sentence Ensemble

Code for the paper "Story Realization: Expanding Plot Events into Sentences" Prithviraj Ammanabrolu, Ethan Tien, Wesley Cheung, Zhaochen Luo, William Ma, Lara J. Martin, and Mark O. Riedl https://arxiv.org/abs/


BibTex:

    {
          @article{ammanabrolu-storyrealize,
            title = "Story Realization: Expanding Plot Events into Sentences",
            author = "Ammanabrolu, Prithviraj and Tien, Ethan and Cheung, Wesley and Luo, Zhaochen and Ma, William and Martin, Lara J. and
              Riedl, Mark O.",
            archivePrefix = {arXiv},
            journal={CoRR},
            volume={},
            arxivId = {}
    }

Disclaimer: Code is not upkept

## TL;DR

**Dataset:**
The full generalized sci-fi dataset can be found [here](https://drive.google.com/open?id=1A5RYjrj9FZsrBtyTr45-fnYWKZX1e7KA), as all-sci-fi-data.txt.

Data columns are split by '|||' and the columns on each line are:
- List of original word events
- List of generalized events
- Original split and pruned sentence
- Generalized split and pruned sentence

Each story ends with an `<EOS>` tag and is followed by a dictionary that contains all the named entities that were generalized by category for that particular story.

For convenience, the data is preprocessed into bitext for the purpose of training our models is also included as all-sci-fi-data-{train, val, test}_{input, output}.txt, with input/output representing the bitext aligned by line number.

**Start RetEdit server:**
```bash
cd E2S-Ensemble/RetEdit/
sudo bash run_server.sh
```
Takes a few minutes to set up, once you see flask output (says something like 'Running on 127.0.0.1:8080') then proceed in a separate terminal.

**Run Ensemble:**

Edit or create a `<config_file>.json` (most recently used is `config_drl_sents.json`). Most likely the only thing that will need to be changed is the "test_src" entry under "data" to reflect a new input file. _(Note: Gold standard dependencies have been removed)_

Then:
```bash
cd E2S-Ensemble/
source activate e2s_ensemble
python ensemble_main.py --config <config_file>.json --sample --cuda --outf <outputfile>.txt
```
This will generate 3 output files:

`<outputfile>.txt`: Pure e2s output with ensemble thresholds

`<outputfile>_verbose.txt`: Each sentence is preceded by the model that generated the chosen sentence.

`<outputfile>_more_verbose.tsv`: Output of ALL 5 models, with their respective confidence scores (useful for quick threshold tuning)

**After Ensemble**

If no longer in use, ctrl+c out of the RetEdit flask instance, and run 
```bash
sudo bash kill_servers.sh
```
to clean up the RetEdit docker instances in order to prevent excess memory consumption.

**Slotfilling**
Code is in `Slotfilling` - edit fillSentsFromFile_comparison.py to read in your data. Pass the event and the memory graph (responsible for keeping track of entities) through ``getAgentTurn``. 

## Other potentially useful files:
```bash
python reweight_ensemble.py <outputfile>_more_verbose.tsv <reweighted_outf>.txt
```
`reweight_ensemble.py` takes in a .tsv generated from _ensemble\_main.py_ and creates new output files based on the thresholds defined in `reweight_ensemble.py`. 
(Generates `<reweighted_outf>.txt` and `<reweighted_outf>_verbose.txt`)

```bash
python take_out_outputs.py <outputfile>_more_verbose.tsv <individual_outf>.txt
```
`take_out_outputs.py` takes in a .tsv generated from `ensemble_main.py` and creates 5 separate output files, one for each model and their respective outputs. 
(Generates `<reweighted_outf>.txt` and `<reweighted_outf>_verbose.txt`)

```bash
python avg_sent_length.py <file>.txt
```
`avg_sent_length.py` calculates average number of words per sentence in the entire file.

```bash
python percent.py <outputfile>_verbose.txt
```
`percent.py` takes in a `<outputfile>_verbose.txt` generated from `ensemble_main.py` and prints out the % utilization of each model in the output file.

## Details:
Main code is under `ensemble_main.py`, which calls the seq2seq-based decoders (Vanilla, mcts, and FSM) that have been reimplemented under `model_lib.py` and the template decoder under `TemplateDecoder.py`, and also queries the Retreival-and-Edit flask instance, which must be previously instantiated.

It takes in an input event file, and runs 5 different threads, one for each model. There are tqdm progress bars for each model to see approxmiately how long each model will take (mcts will take the longest, followed by RetEdit).

Model files and data files have been (for the most part) omitted from this repo due to space issues, but to rebuild a working version of the ensemble from scratch, these are the steps that need to be followed:

### Environment

Use `e2s_ensemble.yml` to create the correct conda environment by doing
```bash
conda env create -f e2s_ensemble.yml
source activate e2s_ensemble
```

### Preparing the Retrival-and-Edit (RetEdit) model (based on this [paper](https://worksheets.codalab.org/worksheets/0x1ad3f387005c492ea913cf0f20c9bb89/))

**Requirements**

docker / nvidia-docker

**How to train**

Work in folder E2S-Ensemble/RetEdit/

1. Prepare data
⋅⋅* create a dataset folder within `/datasets` 
..* create tab-separated dataset: `train.tsv`, `valid.tsv`, and `test.tsv`
..* initialize word vectors within `/word_vectors` (using helper script `create_vectors.py`)
..* set up config file in `/editor_code/configs/editor`
2. Run `sudo bash docker.sh`
3. Run `export COPY_EDIT_DATA=$(pwd); export PYTHONIOENCODING=utf8; cd cond-editor-codalab; python train_seq2seq.py {CONFIG-FILE.TXT}`

**How to use**
Either modify and run `scifi_valid_print.py` or `run_server.sh` (which calls `run_server.py` to handle requests outside of the docker container)

### Preparing the Sequence-to-Sequence based models (Vanilla, FSM, mcts)

Work in the folder E2S-Ensemble/mcts
Modify or create a `<config_file>.json` to incorporate the intended training and testing sets for training. To run training, under the `e2s_ensemble` conda environment, run `python nmt.py --config config.json`. To get output for the vanilla seq2seq model, run `python decode.py --config config.json`.

The FSM and mcts model needs to point to this seq2seq trained model, but does not require any further training.

### Preparing the Templates model

This code adapted from the [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm) codebase.
Work in the folder E2S-Ensemble/Templates

Although the dataset, is generalized, we generalize it a bit further in an attempt to yield better training by removing the numbers from the named entities. To do this, run `python abstract_dataset.py input.txt output.txt` to create a file that has the numbers on the named entities removed. 

Then, prepare a training/validation/test split, each named `train.txt`, `valid.txt`, `test.txt` respectively. To train the model, 
`python main.py --data data/scifi/ --model BiGRU --batch_size 20 --nlayers 1 --epochs 60 --save templates_model.pt --lr 1`

To incorpate for use in `ensemble_main.py`, update the parameters in the first few argument statements.


