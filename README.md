# tb-amr

Unrestricted Transition-based AMR parsing

## Description

tb-amr is a system for Abstract Meaning Representation based on transition-based algorithms.

Supported algorithms:

- AMR-Covington (Vilares and Gómez-Rodríguez, 2018)

The system was tested on the LDC2015E86 corpus, an AMR dataset from the Linguistic Data Consortium. The license does not allow to redistribute such dataset together with our code.


## Requisites

**Recommendation:** Create a virtual environment to avoid problems with existent versions in your computer:

	mkdir $HOME/env
	virtualenv $HOME/env/tb-amr
	source $HOME/env/tb-amr/bin/activate

The repository includes a script **`install.sh`**. Try it for an automatic installation (tested on Ubuntu 16.04).

tb-amr relies on:

**Python 2.7**

**Python packages**:

- [Keras 2.0.9](https://keras.io/) and [Tensorflow 1.4.0 (with support for GPU)](https://www.tensorflow.org/install/)
- [pattern](https://www.clips.uantwerpen.be/pattern)
- stop-words
- numpy
- scipy
- prettytable
- sklearn
- python-dateutil
- Matplotlib
- penman
- h5py

**Additional software**:

- [JAMR](https://github.com/jflanigan/jamr) (for tokenizing and aligning)
- [amr-eager](https://github.com/mdtux89/amr-eager) (we use their text resources and the evaluation scripts for comparison)
- [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/) (they need to be formatted in word2vec format. This basically means adding a first line with two tokens indicating the number of embeddings and the size of the embeddings).


## Pretrained model

It can be downloaded [here](https://drive.google.com/drive/folders/1nrm_9-uehecdxbEZ8MK4yVJbos8KRhXM?usp=sharing). It contains four files:

- The transition classifier: `T.params.hdf5`
- The relation classifier  : `R.params.hdf5`
- The concept classifier   : `C.params.hdf5`
- Parameters               : `params.pickle`


## Run the model

Execute `preprocess.py` on your test set. It applies part-of-speech tagging, dependency parsing and NER and it also runs the JAMR tokenizer and aligner (the latter is only used for `dataset.py`, and is only needed if you want to train your own model from the scratch). The input is a file in AMR format (e.g. `amr-test.txt`)

	python preprocess.py  --amrs /data/amr-test.txt
    
`preprocess.py` generates:

- `*.input`: Raw sentences preprocessed and ready to use in the parsing phase.
- `*.dependencies`: A pickle file containing the part-of-speech/parsing/NER info.
- `*.aligned`: The alignments established by JAMR. (only needed/used for `dataset.py` to create the training and development sets).

>NOTE: `preprocess.py` only supports at the time input files in AMR format, but if you are interested in preprocessing raw file it will be easy to adapt the script to do it.

> NOTE: '*' will be the name of the processed dataset (e.g. amr-test.txt.input)


Execute `main-py` to run the model.

	python main.py --test $PATH/amr-test.txt --outdir $OUTPUT --params $PATH/params.pickle --params_model $PATH/params.hdf5 --predict
    
> NOTE: `main.py` reads the file `configuration.conf` to find the paths where external resources are located. 
> 
> NOTE: tb-amr assumes the classifiers are named by a T, R and C, plus a common surname (e.g. params.hdf5). To run the classifiers you only need to indicate this common surname in the parameter `--params_model` (as showed in the example). 

## Train your own model

Let `amr-training.txt` and `amr-dev.txt` be the training and development AMR files.

Preprocess them:

	python preprocess.py  --amrs /data/amr-training.txt
    python preprocess.py  --amrs /data/amr-dev.txt
  
Execute `dataset.py`. It creates an AMR dataset (the `*.graphs` file) ready to use for training a model using tb-amr 

	python dataset.py  --amrs /data/amr-training.txt
    python dataset.py  --amrs /data/amr-dev.txt
    
`dataset.py` also stores subgraphs that are created from named-entities, phrases and words into:

- `*.subgraphs`
- `*.phrase_subgraphs` 
- `*.word_concepts`

Execute also the `main.py` file to train your model by typing the following command:

	python main.py --train $PATH/amr-training.txt --dev $PATH/amr-dev.txt --outdir $OUTPUT --params $OUTPUT/params.pickle --params_model $OUTPUT/params.hdf5
	
`main.py` will train three classifiers:

- The transition classifier: Stored in `$OUTPUT/T.params.hdf5`
- The relation classifier  : Stored in `$OUTPUT/R.params.hdf5`
- The concept classifier   : Stored in `$OUTPUT/C.params.hdf5`
- Parameters             : Stored in `$OUTPUT/params.pickle`


## References

David Vilares and Carlos Gómez-Rodríguez. A Transition-based Algorithm for Unrestricted AMR Parsing,
NAACL HLT 2018 - The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, accepted, New Orleans, USA, 2018


## Additional notes

tb-amr is in a early stage and under development. Interfaces are subject to changes.

## Contact

If you have any suggestions, inquiry or bug to report, please contact david.vilares@udc.es
