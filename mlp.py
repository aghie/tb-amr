# # -*- coding: utf-8 -*-

from keras.models import Sequential, Model, save_model
from keras.layers import Dense, Dropout, Embedding, Activation, Flatten, Input, Masking, Reshape, RepeatVector
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional, merge, Merge, Lambda, Permute
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras import backend as K
from keras.backend.tensorflow_backend import print_tensor
from keras.metrics import categorical_accuracy

from prettytable import PrettyTable
from operator import itemgetter
from itertools import chain
from numpy import argmax
from collections import Counter, OrderedDict
from algorithm import AMRCovington
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from tempfile import NamedTemporaryFile
from pattern.en import verbs, conjugate, INFINITIVE, singularize
from datetime import datetime
from dateutil.parser import parse


from algorithm import CovingtonConfiguration
from concept import RuleBasedConceptModel

import dateutil.parser as dtparser
import matplotlib.pyplot as plt
import time, random
import numpy as np
import warnings
import pickle
import cPickle
import copy
import time
import sys
import os
import codecs
import numpy as np
import keras
import utils
import itertools
import mlp_utils
import string
from stop_words import get_stop_words

class PerceptronAMR(object):

    ROOT_INDEX = 0
    UNK_INDEX = 1
    EMPTY_INDEX = 2
    SPECIAL_INDEXES = [ROOT_INDEX, UNK_INDEX, EMPTY_INDEX]    
    INIT_REAL_INDEX = len(SPECIAL_INDEXES)
    SENTENCE_SEPARATOR_TOKENS = [".", "?", "!", "...", ";"]
    
    #POTENTIAL FEATURES TO BE USED
    BUFFER = "B"
    L1 = "L1" 
    DEP = "DEP" #Extracted externally from a dependency tree
    
    INTERNAL_WORD = "INTERNAL-WORD"
    EXTERNAL_WORD = "EXTERNAL-WORD"
    POSTAG = "POSTAG"
    CONCEPT = "CONCEPT"
    ENTITY = "ENTITYW"
    LEFTMOST_HEAD_CONCEPT = "LEFTMOST-HEAD-CONCEPT"
    LEFTMOST_CHILD_CONCEPT = "LEFTMOST-CHILD-CONCEPT"
    LEFTMOST_GCHILD_CONCEPT = "LEFTMOST-GCHILD-CONCEPT"
    RIGHTMOST_HEAD_CONCEPT = "RIGHTMOST-HEAD-CONCEPT"
    RIGHTMOST_CHILD_CONCEPT = "RIGHTMOST-CHILD-CONCEPT"
    RIGHTMOST_GCHILD_CONCEPT = "RIGHTMOST-GCHILD-CONCEPT"
    N_HEADS = "N-HEADS"
    N_CHILDREN = "N-CHILDREN"
    DEPTH = "DEPTH"
    CONCEPT_TYPE = "CONCEPT-TYPE"
    DEP_NAME = "DEP-NAME"
    LAST_HEAD_EDGE = "LAST-HEAD-EDGE"
    GENERATED_BY = "GENERATED"
    N_PREVIOUS_SENTENCE_TOKENS = "N-SEPARATOR-TOKENS"     

    #These are not being used at the moment
    LM_HEAD_INTERNAL_WORD = "LMHIW"
    LM_CHILD_INTERNAL_WORD = "LMCIW"
    LEFTMOST_GCHILD_INTERNAL_WORD = "LMCCIW"   
    
    #ALL POSSIBLE FEATURES
    FEATURES = [INTERNAL_WORD, EXTERNAL_WORD, POSTAG, CONCEPT, ENTITY, LEFTMOST_HEAD_CONCEPT, 
                LEFTMOST_CHILD_CONCEPT, LEFTMOST_GCHILD_CONCEPT, N_HEADS, N_CHILDREN, GENERATED_BY, 
                LAST_HEAD_EDGE, DEPTH, CONCEPT_TYPE, RIGHTMOST_HEAD_CONCEPT, RIGHTMOST_CHILD_CONCEPT, 
                RIGHTMOST_GCHILD_CONCEPT, LM_HEAD_INTERNAL_WORD, LM_CHILD_INTERNAL_WORD, 
                LEFTMOST_GCHILD_INTERNAL_WORD, N_PREVIOUS_SENTENCE_TOKENS]
    
    #Feature indexes that can be eventually randomized
    INDEXES_TO_RANDOMIZE = [3, 5, 6, 7, 14, 15, 16]
      
    #FEATURES FROM B and L1 USED BY THE CONCEPT CLASSIFIER
    FC_B = [INTERNAL_WORD, EXTERNAL_WORD, POSTAG, CONCEPT, ENTITY, GENERATED_BY, DEPTH]
    FC_L1 = [INTERNAL_WORD, EXTERNAL_WORD, POSTAG, CONCEPT, ENTITY, GENERATED_BY, DEPTH]
    
    #FEATURES FROM B and L1 USED BY THE TRANSITIONS CLASSIFIER
    FT_B = [INTERNAL_WORD, EXTERNAL_WORD, POSTAG, CONCEPT, ENTITY, LEFTMOST_HEAD_CONCEPT, 
            LEFTMOST_CHILD_CONCEPT, LEFTMOST_GCHILD_CONCEPT, N_HEADS, N_CHILDREN, DEPTH, 
            CONCEPT_TYPE, RIGHTMOST_HEAD_CONCEPT, RIGHTMOST_CHILD_CONCEPT, 
            RIGHTMOST_GCHILD_CONCEPT,N_PREVIOUS_SENTENCE_TOKENS]  
    FT_L1 = [INTERNAL_WORD, EXTERNAL_WORD, POSTAG, CONCEPT, ENTITY, LEFTMOST_HEAD_CONCEPT,
              LEFTMOST_CHILD_CONCEPT, LEFTMOST_GCHILD_CONCEPT, N_HEADS, N_CHILDREN, DEPTH, 
              CONCEPT_TYPE, RIGHTMOST_HEAD_CONCEPT, RIGHTMOST_CHILD_CONCEPT, 
              RIGHTMOST_GCHILD_CONCEPT, N_PREVIOUS_SENTENCE_TOKENS]
    
    #FEATURES FROM B and L1 USED BT THE RELATIONS CLASSIFIER
    FR_B = [INTERNAL_WORD, EXTERNAL_WORD, POSTAG, CONCEPT, ENTITY, LEFTMOST_HEAD_CONCEPT, 
            LEFTMOST_CHILD_CONCEPT, LEFTMOST_GCHILD_CONCEPT, N_HEADS, N_CHILDREN, LAST_HEAD_EDGE,
            DEPTH, CONCEPT_TYPE, RIGHTMOST_HEAD_CONCEPT, RIGHTMOST_CHILD_CONCEPT, 
            RIGHTMOST_GCHILD_CONCEPT,N_PREVIOUS_SENTENCE_TOKENS]  
    FR_L1 = [INTERNAL_WORD, EXTERNAL_WORD, POSTAG, CONCEPT, ENTITY, LEFTMOST_HEAD_CONCEPT, 
             LEFTMOST_CHILD_CONCEPT, LEFTMOST_GCHILD_CONCEPT, N_HEADS, N_CHILDREN, LAST_HEAD_EDGE, 
             DEPTH, CONCEPT_TYPE, RIGHTMOST_HEAD_CONCEPT, RIGHTMOST_CHILD_CONCEPT, 
             RIGHTMOST_GCHILD_CONCEPT,N_PREVIOUS_SENTENCE_TOKENS] 


    #TODO: Integrate this info as a part of the information given by an AMR_ENTRY. At the moment
    #it is done externally.
    #Dependency features used by the concepts, transition and relations classifiers
    #Similar to Damonte et al (2017): An Incremental Parser for Abstract Meaning Representation
    B_DEPS = [0, 1] 
    L1_DEPS = [0, 1, 2, 3]
    
    FEATURE_DEPS = [("B2L1", bi, l1i) for bi, l1i in list(itertools.product(B_DEPS, L1_DEPS))]
    FEATURE_DEPS.extend([("L12B", l1i, bi) for l1i, bi in list(itertools.product(L1_DEPS, B_DEPS))])
    
    FC_DEP = [("B2L1", bi, l1i) for bi, l1i in list(itertools.product(B_DEPS, L1_DEPS))]
    FC_DEP.extend([("L12B", l1i, bi) for l1i, bi in list(itertools.product(L1_DEPS, B_DEPS))])
    
    FT_DEP = [("B2L1", bi, l1i) for bi, l1i in list(itertools.product(B_DEPS, L1_DEPS))]
    FT_DEP.extend([("L12B", l1i, bi) for l1i, bi in list(itertools.product(L1_DEPS, B_DEPS))])
    

    FR_DEP = [("B2L1", bi, l1i) for bi, l1i in list(itertools.product(B_DEPS, L1_DEPS))]
    FR_DEP.extend([("L12B", l1i, bi) for l1i, bi in list(itertools.product(L1_DEPS, B_DEPS))])


    FEATURES.extend(FEATURE_DEPS)


    
    def __init__(self, vocab, pos, rels, nodes, entities, deps, eew, eep, ee_edge, ee_dep, eee, graph_templates,
                 multiword_graph_templates,
                path_lookup_table, 
                options, trained=False):

        self.vocab = vocab
        self.algorithm = AMRCovington(use_shift=True)   
        self.r_concept_model = RuleBasedConceptModel(self.vocab,self.algorithm, 
                                                     graph_templates,
                                                     options.nationalities,
                                                     options.nationalities2,
                                                     options.cities, 
                                                     options.countries,
                                                     options.states,
                                                     options.negations,
                                                     options.verbalize,
                                                     multiword_graph_templates)    
        
        if path_lookup_table is not None:
            self.r_concept_model.set_lookup_concepts(path_lookup_table)
        
        self.switch_to_unk_c_prob = 0.002
        self.threshold_c_occ = 1
        self.cdims = 50
        self.wdims = 100
        self.pdims = 10
        self.edge_dims = 10
        self.dep_dims = 10
        self.entity_dims = 10
        self.dropout = 0.40
        self.batch_size = 32
        self.epochs = options.epochs
        self.options = options
        self.C_unk = set([])
        self.C = set([])
        
        for node in nodes:
            if nodes[node] <= self.threshold_c_occ: 
                self.C_unk.add(node)
            else:
                self.C.add(node)
                
        with codecs.open(options.arg_rules) as f:         
            self.args_rules = {l.split(" , ")[0]:map(int, l.split(" , ")[1:]) 
                                for l in f.readlines()} 
 
        ########################################################################
        # Output classes for the (T)ransition, (R)elation and (C) NNs
        ########################################################################
        if options.expanded_rels:
            self.R = {rel: ind for ind, rel in enumerate(sorted(self._get_labels(rels)))}
        else:
            self.R = {rel: ind for ind, rel in enumerate(sorted(rels.keys()))}    
        self.C = set(self.C)
        self.C_unk = set(self.C_unk)
        self.T = {a: ind for ind, a in enumerate(AMRCovington.TRANSITIONS)}

        self.entities = {ent: ind for ind, ent in enumerate(sorted(entities.keys()))}
        self.nodes = {node:ind for ind, node in enumerate(sorted(self.C))}    
        self.nodes_i = {self.nodes[n]: n for n in self.C}  

        #Used just for evaluation
        self.rels_all = {}
        self.rels_all.update(self.R)
        #Used just for evaluation
        self.nodes_all = {}
        self.nodes_all.update(self.nodes)
        self.nodes_all.update({node:ind + len(self.nodes) for ind, node in enumerate(sorted(self.C_unk))})    
        
        ########################################################################
        #       Components to define the different embedding matrices          #
        ########################################################################
        
        # Internal word embedding
        self.i_iforms, self.wi_lookup, self.widims = self._get_embeddings(None, dims=self.wdims, 
                                                                          keys=sorted(vocab.keys()))
        self.i_iforms_reverse = {self.i_iforms[w]:w for w in self.i_iforms}
        
        #External word embeddings
        self.i_eforms, self.w_lookup, self.wdims =  self._get_embeddings(eew, dims=self.wdims, 
                                                             keys=sorted(vocab.keys()))
        self.i_eforms_reverse = {self.i_eforms[e]:e for e in self.i_eforms}
        
        #Postag embeddings
        self.ipos, self.p_lookup, self.pdims = self._get_embeddings(eep, dims=self.pdims, 
                                                                    keys=sorted(pos.keys()))
        self.ipos_reverse = {self.ipos[e]:e for e in self.ipos}        
    
        #Label embeddings
        self.iedge, self.edge_lookup ,self.edge_dims = self._get_embeddings(ee_edge, dims=self.edge_dims, 
                                                                            keys=sorted(rels.keys()))
        self.iedge_reverse = {self.iedge[e]:e for e in self.iedge} 
    
        #Dependency embeddings
        self.idep, self.d_lookup, self.ddims = self._get_embeddings(ee_dep, dims=self.dep_dims, 
                                                                    keys = sorted(deps.keys()))
        self.idep_reverse = {self.idep[e]:e for e in self.idep} 

        #Entity embeddings
        self.ientity, self.e_lookup, self.entdims = self._get_embeddings(eee, dims=self.entity_dims, 
                                                                         keys=sorted(entities.keys()))
        self.ientity_reverse = {self.ientity[e]:e for e in self.ientity}
        # Concept embedding           
        #TODO: Minor bug, there is no need to sum self.nodes_all, I think,
        #but it was used for the pretrained model
        self.iconcepts = {n:self.INIT_REAL_INDEX + self.nodes_all[n] for n in sorted(self.nodes_all)}
        self.c_lookup = np.zeros(shape=(len(self.iconcepts) + len(self.SPECIAL_INDEXES), self.cdims))       
        self.iconcepts_reverse = {self.iconcepts[c]: c for c in self.iconcepts}
        
        
        self.t_classes = len(self.T)
        self.c_classes = len(self.C) 
        self.r_classes = len(self.R)
        

        # Generate the inputs:
        dict_input = OrderedDict({})
        inputs = []
        for j, f in enumerate(range(0, self.algorithm.wB)):
                          
            dict_input["_".join((self.BUFFER, str(j), self.INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.INTERNAL_WORD)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.EXTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.EXTERNAL_WORD)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.POSTAG))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.POSTAG)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.CONCEPT))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.CONCEPT)), dtype='float32')  
            dict_input["_".join((self.BUFFER, str(j), self.ENTITY))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.ENTITY)), dtype='float32')                
            dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_HEAD_CONCEPT))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.LEFTMOST_HEAD_CONCEPT)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_CHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.LEFTMOST_CHILD_CONCEPT)), dtype='float32')          
            dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_CONCEPT)), dtype='float32')              
            dict_input["_".join((self.BUFFER, str(j), self.N_HEADS))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.N_HEADS)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.N_CHILDREN))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.N_CHILDREN)), dtype='float32') 
            dict_input["_".join((self.BUFFER, str(j), self.GENERATED_BY))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.GENERATED_BY)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.LAST_HEAD_EDGE))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.LAST_HEAD_EDGE)), dtype='float32')  
            dict_input["_".join((self.BUFFER, str(j), self.DEPTH))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.DEPTH)), dtype='float32')    
            dict_input["_".join((self.BUFFER, str(j), self.CONCEPT_TYPE))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.CONCEPT_TYPE)), dtype='float32')            
            dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_HEAD_CONCEPT))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.RIGHTMOST_HEAD_CONCEPT)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_CHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.RIGHTMOST_CHILD_CONCEPT)), dtype='float32')          
            dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_GCHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.RIGHTMOST_GCHILD_CONCEPT)), dtype='float32') 
            dict_input["_".join((self.BUFFER, str(j), self.LM_HEAD_INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.LM_HEAD_INTERNAL_WORD)), dtype='float32')
            dict_input["_".join((self.BUFFER, str(j), self.LM_CHILD_INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.LM_CHILD_INTERNAL_WORD)), dtype='float32')          
            dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_INTERNAL_WORD)), dtype='float32')             
            dict_input["_".join((self.BUFFER, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))] = Input(shape=(1,), name="_".join((self.BUFFER, str(j), self.N_PREVIOUS_SENTENCE_TOKENS)), dtype='float32') 
        
        for j, f in enumerate(range(0, self.algorithm.wL1)):
            
            dict_input["_".join((self.L1, str(j), self.INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.INTERNAL_WORD)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.EXTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.EXTERNAL_WORD)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.POSTAG))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.POSTAG)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.CONCEPT))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.CONCEPT)), dtype='float32')  
            dict_input["_".join((self.L1, str(j), self.ENTITY))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.ENTITY)), dtype='float32')   
            dict_input["_".join((self.L1, str(j), self.LEFTMOST_HEAD_CONCEPT))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.LEFTMOST_HEAD_CONCEPT)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.LEFTMOST_CHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.LEFTMOST_CHILD_CONCEPT)), dtype='float32')          
            dict_input["_".join((self.L1, str(j), self.LEFTMOST_GCHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.LEFTMOST_GCHILD_CONCEPT)), dtype='float32')  
            dict_input["_".join((self.L1, str(j), self.N_HEADS))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.N_HEADS)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.N_CHILDREN))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.N_CHILDREN)), dtype='float32') 
            dict_input["_".join((self.L1, str(j), self.GENERATED_BY))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.GENERATED_BY)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.LAST_HEAD_EDGE))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.LAST_HEAD_EDGE)), dtype='float32')            
            dict_input["_".join((self.L1, str(j), self.DEPTH))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.DEPTH)), dtype='float32')             
            dict_input["_".join((self.L1, str(j), self.CONCEPT_TYPE))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.CONCEPT_TYPE)), dtype='float32') 
            dict_input["_".join((self.L1, str(j), self.RIGHTMOST_HEAD_CONCEPT))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.RIGHTMOST_HEAD_CONCEPT)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.RIGHTMOST_CHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.RIGHTMOST_CHILD_CONCEPT)), dtype='float32')          
            dict_input["_".join((self.L1, str(j), self.RIGHTMOST_GCHILD_CONCEPT))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.RIGHTMOST_GCHILD_CONCEPT)), dtype='float32')  
            dict_input["_".join((self.L1, str(j), self.LM_HEAD_INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.LM_HEAD_INTERNAL_WORD)), dtype='float32')
            dict_input["_".join((self.L1, str(j), self.LM_CHILD_INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.LM_CHILD_INTERNAL_WORD)), dtype='float32')          
            dict_input["_".join((self.L1, str(j), self.LEFTMOST_GCHILD_INTERNAL_WORD))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.LEFTMOST_GCHILD_INTERNAL_WORD)), dtype='float32')           
            dict_input["_".join((self.L1, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))] = Input(shape=(1,), name="_".join((self.L1, str(j), self.N_PREVIOUS_SENTENCE_TOKENS)), dtype='float32') 
    
        for j in range(0, len(self.FEATURE_DEPS)):
            dict_input["_".join((self.DEP, str(j), self.DEP_NAME))] = Input(shape=(1,), name="_".join((self.DEP, str(j),
                                                                                                      self.DEP_NAME)), dtype='float32')

        for i in dict_input:            
             inputs.append(dict_input[i])

        self.N_inputs = len(inputs)
        #Computing the inverted indexes
        self.nodes_i = {self.nodes[n]: n for n in self.nodes}    
        self.actions_i = {self.T[n]: n for n in self.T}     
        self.rels_i = {self.R[n]:n for n in self.R}
        
        ########################################################################
        #      EMBEDDINGS MATRICES FOR THE AMR-COVINGTON TRANSTIONS NN
        ########################################################################
        
        e_IWT = Embedding(self.wi_lookup.shape[0],self.widims, 
                          #embeddings_initializer='glorot_uniform',
                          weights=[self.wi_lookup],
                          input_length=1,name="e_IWT", trainable=True)

        e_EWT = Embedding(self.w_lookup.shape[0], self.wdims,
                          #embeddings_initializer='glorot_uniform', 
                          weights=[self.w_lookup],
                          input_length=1,name="e_EWT",trainable=True)
          
        e_PT = Embedding(self.p_lookup.shape[0],self.pdims,
                       #  embeddings_initializer='glorot_uniform',
                         weights=[self.p_lookup],
                         input_length=1,name="e_PT",trainable=True)
                                          
        e_CT = Embedding(self.c_lookup.shape[0],self.cdims,
                      #   embeddings_initializer='glorot_uniform',
                         weights=[self.c_lookup],
                         input_length=1,name="e_CT",trainable=True) 
          
        e_ET = Embedding(self.e_lookup.shape[0],self.entdims,
                        # embeddings_initializer='glorot_uniform',
                         weights=[self.e_lookup],
                         input_length=1,name="e_ET", trainable=True) 
        
        e_EDGET = Embedding(self.edge_lookup.shape[0], self.edge_dims, 
                         #   embeddings_initializer='glorot_uniform',
                            weights=[self.edge_lookup],
                        input_length=1, name="e_EDGET", trainable=True)

        e_DEPT = Embedding(self.d_lookup.shape[0],self.ddims, 
                          # embeddings_initializer='glorot_uniform',
                           weights=[self.d_lookup],
                        input_length=1,name="e_DEPT",trainable=True)

        ########################################################################
        #       EMBEDDINGS MATRICES FOR THE RELATIONS NN
        ########################################################################

        e_IWR = Embedding(self.wi_lookup.shape[0], self.widims, 
                          #embeddings_initializer='glorot_uniform',
                          weights=[self.wi_lookup],
                          input_length=1, name="e_IWR", trainable=True)
          
        e_EWR = Embedding(self.w_lookup.shape[0], self.wdims,
                          #embeddings_initializer='glorot_uniform',
                          weights=[self.w_lookup],
                          input_length=1,name="e_EWR",trainable=True)
          
        e_PR = Embedding(self.p_lookup.shape[0], self.pdims,
                         #embeddings_initializer='glorot_uniform',
                         weights=[self.p_lookup],
                         input_length=1,name="e_PR",trainable=True)
                                          
        e_CR = Embedding(self.c_lookup.shape[0],self.cdims,
                         #embeddings_initializer='glorot_uniform',
                         weights=[self.c_lookup],
                         input_length=1,name="e_CR", trainable=True) 
          
        e_ER = Embedding(self.e_lookup.shape[0],self.entdims,
                         #embeddings_initializer='glorot_uniform',
                         weights=[self.e_lookup],
                         input_length=1,name="e_ER",trainable=True) 
        
        e_EDGER = Embedding(self.edge_lookup.shape[0], self.edge_dims,
                            #embeddings_initializer='glorot_uniform',
                            weights=[self.edge_lookup],
                        input_length=1,name="e_EDGER",trainable=True)

        e_DEPR = Embedding(self.d_lookup.shape[0],self.ddims,
                           #embeddings_initializer='glorot_uniform',
                           weights=[self.d_lookup],
                        input_length=1,name="e_DEPR",trainable=True)        

        ########################################################################
        #       EMBEDDINGS MATRICES FOR THE CONCEPT NN
        ########################################################################        
         
        e_IWC = Embedding(self.wi_lookup.shape[0],self.widims, 
                          #embeddings_initializer='glorot_uniform',
                          weights=[self.wi_lookup],
                          input_length=1,name="e_IWC",trainable=True)

        e_EWC = Embedding(self.w_lookup.shape[0],self.wdims, 
                          #embeddings_initializer='glorot_uniform',
                          weights=[self.w_lookup],
                          input_length=1,name="e_EWC",trainable=True)
          
        e_PC = Embedding(self.p_lookup.shape[0],self.pdims, 
                        # embeddings_initializer='glorot_uniform',
                        weights=[self.p_lookup],
                        input_length=1,name="e_PC",trainable=True)
                                          
        e_CC = Embedding(self.c_lookup.shape[0],self.cdims,
                         #embeddings_initializer='glorot_uniform',
                         weights=[self.c_lookup],
                         input_length=1,name="e_CC",trainable=True) 
          
        e_EC = Embedding(self.e_lookup.shape[0],self.entdims,
                         #embeddings_initializer='glorot_uniform',
                         weights=[self.e_lookup],
                         input_length=1,name="e_EC",trainable=True) 
         
        e_EDGEC = Embedding(self.edge_lookup.shape[0],self.edge_dims,
                        # embeddings_initializer='glorot_uniform',   
                        weights=[self.edge_lookup],
                        input_length=1,name="e_EDGEC",trainable=True)
        
        e_DEPC = Embedding(self.d_lookup.shape[0],self.ddims,
                           #embeddings_initializer='glorot_uniform',
                           weights=[self.d_lookup],
                           input_length=1,name="e_DEPC",trainable=True)
    
    
        d_eC = {}
        d_eT = {}
        d_eR = {}
        
        ################################################
        # EMBEDDINGS FOR THE TRANSITION MODEL MODEL
        ################################################        
        
        for j, f in enumerate(range(0, self.algorithm.wB)):
            
            d_eT["_".join((self.BUFFER, str(j), self.INTERNAL_WORD))] = e_IWT(dict_input["_".join((self.BUFFER, str(j), self.INTERNAL_WORD))])
            d_eT["_".join((self.BUFFER, str(j), self.EXTERNAL_WORD))] = e_EWT(dict_input["_".join((self.BUFFER, str(j), self.EXTERNAL_WORD))])
            d_eT["_".join((self.BUFFER, str(j), self.POSTAG))] = e_PT(dict_input["_".join((self.BUFFER, str(j), self.POSTAG))])
            d_eT["_".join((self.BUFFER, str(j), self.CONCEPT))] = e_CT(dict_input["_".join((self.BUFFER, str(j), self.CONCEPT))]) 
            d_eT["_".join((self.BUFFER, str(j), self.ENTITY))] = e_ET(dict_input["_".join((self.BUFFER, str(j), self.ENTITY))])          
            d_eT["_".join((self.BUFFER, str(j), self.LEFTMOST_HEAD_CONCEPT))] = e_CT(dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_HEAD_CONCEPT))])
            d_eT["_".join((self.BUFFER, str(j), self.LEFTMOST_CHILD_CONCEPT))] = e_CT(dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_CHILD_CONCEPT))])
            d_eT["_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_CONCEPT))] = e_CT(dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_CONCEPT))])      
            d_eT["_".join((self.BUFFER, str(j), self.N_HEADS))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.N_HEADS))]) 
            d_eT["_".join((self.BUFFER, str(j), self.N_CHILDREN))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.N_CHILDREN))])
            d_eT["_".join((self.BUFFER, str(j), self.DEPTH))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.DEPTH))])
            d_eT["_".join((self.BUFFER, str(j), self.CONCEPT_TYPE))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.CONCEPT_TYPE))])
            d_eT["_".join((self.BUFFER, str(j), self.RIGHTMOST_HEAD_CONCEPT))] = e_CT(dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_HEAD_CONCEPT))])
            d_eT["_".join((self.BUFFER, str(j), self.RIGHTMOST_CHILD_CONCEPT))] = e_CT(dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_CHILD_CONCEPT))])
            d_eT["_".join((self.BUFFER, str(j), self.RIGHTMOST_GCHILD_CONCEPT))] = e_CT(dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_GCHILD_CONCEPT))])    
            d_eT["_".join((self.BUFFER, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))]) 
            d_eT["_".join((self.BUFFER,str(j),self.LAST_HEAD_EDGE))] = e_EDGET(dict_input["_".join((self.BUFFER,str(j),self.LAST_HEAD_EDGE))]) 
       
        for j, f in enumerate(range(0, self.algorithm.wL1)):
    
            d_eT["_".join((self.L1, str(j), self.INTERNAL_WORD))] = e_IWT(dict_input["_".join((self.L1, str(j), self.INTERNAL_WORD))]) 
            d_eT["_".join((self.L1, str(j), self.EXTERNAL_WORD))] = e_EWT(dict_input["_".join((self.L1, str(j), self.EXTERNAL_WORD))]) 
            d_eT["_".join((self.L1, str(j), self.POSTAG))] = e_PT(dict_input["_".join((self.L1, str(j), self.POSTAG))]) 
            d_eT["_".join((self.L1, str(j), self.CONCEPT))] = e_CT(dict_input["_".join((self.L1, str(j), self.CONCEPT))])  
            d_eT["_".join((self.L1, str(j), self.ENTITY))] = e_ET(dict_input["_".join((self.L1, str(j), self.ENTITY))])          
            d_eT["_".join((self.L1, str(j), self.LEFTMOST_HEAD_CONCEPT))] = e_CT(dict_input["_".join((self.L1, str(j), self.LEFTMOST_HEAD_CONCEPT))])
            d_eT["_".join((self.L1, str(j), self.LEFTMOST_CHILD_CONCEPT))] = e_CT(dict_input["_".join((self.L1, str(j), self.LEFTMOST_CHILD_CONCEPT))])
            d_eT["_".join((self.L1, str(j), self.LEFTMOST_GCHILD_CONCEPT))] = e_CT(dict_input["_".join((self.L1, str(j), self.LEFTMOST_GCHILD_CONCEPT))])    
            d_eT["_".join((self.L1, str(j), self.N_HEADS))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.N_HEADS))]) 
            d_eT["_".join((self.L1, str(j), self.N_CHILDREN))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.N_CHILDREN))])   
            d_eT["_".join((self.L1, str(j), self.DEPTH))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.DEPTH))])    
            d_eT["_".join((self.L1, str(j), self.CONCEPT_TYPE))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.CONCEPT_TYPE))]) 
            d_eT["_".join((self.L1, str(j), self.RIGHTMOST_HEAD_CONCEPT))] = e_CT(dict_input["_".join((self.L1, str(j), self.RIGHTMOST_HEAD_CONCEPT))])
            d_eT["_".join((self.L1, str(j), self.RIGHTMOST_CHILD_CONCEPT))] = e_CT(dict_input["_".join((self.L1, str(j), self.RIGHTMOST_CHILD_CONCEPT))])
            d_eT["_".join((self.L1, str(j), self.RIGHTMOST_GCHILD_CONCEPT))] = e_CT(dict_input["_".join((self.L1, str(j), self.RIGHTMOST_GCHILD_CONCEPT))])  
            d_eT["_".join((self.L1, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))]) 
            d_eT["_".join((self.L1,str(j),self.LAST_HEAD_EDGE))] = e_EDGET(dict_input["_".join((self.L1,str(j),self.LAST_HEAD_EDGE))])

        for j in range(0, len(self.FT_DEP)):
            d_eT["_".join((self.DEP, str(j), self.DEP_NAME))] = e_DEPT(dict_input["_".join((self.DEP, str(j), self.DEP_NAME))])

        ################################################
        # EMBEDDINGS FOR THE RELATION MODEL
        ################################################

        for j, f in enumerate(range(0, self.algorithm.wB)):
            
            d_eR["_".join((self.BUFFER, str(j), self.INTERNAL_WORD))] = e_IWR(dict_input["_".join((self.BUFFER, str(j), self.INTERNAL_WORD))])
            d_eR["_".join((self.BUFFER, str(j), self.EXTERNAL_WORD))] = e_EWR(dict_input["_".join((self.BUFFER, str(j), self.EXTERNAL_WORD))])
            d_eR["_".join((self.BUFFER, str(j), self.POSTAG))] = e_PR(dict_input["_".join((self.BUFFER, str(j), self.POSTAG))])
            d_eR["_".join((self.BUFFER, str(j), self.CONCEPT))] = e_CR(dict_input["_".join((self.BUFFER, str(j), self.CONCEPT))]) 
            d_eR["_".join((self.BUFFER, str(j), self.ENTITY))] = e_ER(dict_input["_".join((self.BUFFER, str(j), self.ENTITY))])       
            d_eR["_".join((self.BUFFER, str(j), self.LEFTMOST_HEAD_CONCEPT))] = e_CR(dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_HEAD_CONCEPT))])
            d_eR["_".join((self.BUFFER, str(j), self.LEFTMOST_CHILD_CONCEPT))] = e_CR(dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_CHILD_CONCEPT))])
            d_eR["_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_CONCEPT))] = e_CR(dict_input["_".join((self.BUFFER, str(j), self.LEFTMOST_GCHILD_CONCEPT))])    
            d_eR["_".join((self.BUFFER, str(j), self.N_HEADS))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.N_HEADS))]) 
            d_eR["_".join((self.BUFFER, str(j), self.N_CHILDREN))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.N_CHILDREN))])
            d_eR["_".join((self.BUFFER, str(j), self.LAST_HEAD_EDGE))] = e_EDGER(dict_input["_".join((self.BUFFER, str(j), self.LAST_HEAD_EDGE))]) 
            d_eR["_".join((self.BUFFER, str(j), self.DEPTH))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.DEPTH))])
            d_eR["_".join((self.BUFFER, str(j), self.CONCEPT_TYPE))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.CONCEPT_TYPE))])
            d_eR["_".join((self.BUFFER, str(j), self.RIGHTMOST_HEAD_CONCEPT))] = e_CR(dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_HEAD_CONCEPT))])
            d_eR["_".join((self.BUFFER, str(j), self.RIGHTMOST_CHILD_CONCEPT))] = e_CR(dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_CHILD_CONCEPT))])
            d_eR["_".join((self.BUFFER, str(j), self.RIGHTMOST_GCHILD_CONCEPT))] = e_CR(dict_input["_".join((self.BUFFER, str(j), self.RIGHTMOST_GCHILD_CONCEPT))])     
            d_eR["_".join((self.BUFFER, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))]) 
                                                                 
        for j, f in enumerate(range(0, self.algorithm.wL1)):
            
            d_eR["_".join((self.L1, str(j), self.INTERNAL_WORD))] = e_IWR(dict_input["_".join((self.L1, str(j), self.INTERNAL_WORD))]) 
            d_eR["_".join((self.L1, str(j), self.EXTERNAL_WORD))] = e_EWR(dict_input["_".join((self.L1, str(j), self.EXTERNAL_WORD))]) 
            d_eR["_".join((self.L1, str(j), self.POSTAG))] = e_PR(dict_input["_".join((self.L1, str(j), self.POSTAG))]) 
            d_eR["_".join((self.L1, str(j), self.CONCEPT))] = e_CR(dict_input["_".join((self.L1, str(j), self.CONCEPT))])  
            d_eR["_".join((self.L1, str(j), self.ENTITY))] = e_ER(dict_input["_".join((self.L1, str(j), self.ENTITY))]) 
            d_eR["_".join((self.L1, str(j), self.LEFTMOST_HEAD_CONCEPT))] = e_CR(dict_input["_".join((self.L1, str(j), self.LEFTMOST_HEAD_CONCEPT))])
            d_eR["_".join((self.L1, str(j), self.LEFTMOST_CHILD_CONCEPT))] = e_CR(dict_input["_".join((self.L1, str(j), self.LEFTMOST_CHILD_CONCEPT))])
            d_eR["_".join((self.L1, str(j), self.LEFTMOST_GCHILD_CONCEPT))] = e_CR(dict_input["_".join((self.L1, str(j), self.LEFTMOST_GCHILD_CONCEPT))])
            d_eR["_".join((self.L1, str(j), self.N_HEADS))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.N_HEADS))])
            d_eR["_".join((self.L1, str(j), self.N_CHILDREN))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.N_CHILDREN))])            
            d_eR["_".join((self.L1, str(j), self.LAST_HEAD_EDGE))] = e_EDGER(dict_input["_".join((self.L1, str(j), self.LAST_HEAD_EDGE))])          
            d_eR["_".join((self.L1, str(j), self.DEPTH))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.DEPTH))])   
            d_eR["_".join((self.L1, str(j), self.CONCEPT_TYPE))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.CONCEPT_TYPE))])
            d_eR["_".join((self.L1, str(j), self.RIGHTMOST_HEAD_CONCEPT))] = e_CR(dict_input["_".join((self.L1, str(j), self.RIGHTMOST_HEAD_CONCEPT))])
            d_eR["_".join((self.L1, str(j), self.RIGHTMOST_CHILD_CONCEPT))] = e_CR(dict_input["_".join((self.L1, str(j), self.RIGHTMOST_CHILD_CONCEPT))])
            d_eR["_".join((self.L1, str(j), self.RIGHTMOST_GCHILD_CONCEPT))] = e_CR(dict_input["_".join((self.L1, str(j), self.RIGHTMOST_GCHILD_CONCEPT))])     
            d_eR["_".join((self.L1, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.N_PREVIOUS_SENTENCE_TOKENS))])

        for j in range(0, len(self.FR_DEP)):
            d_eR["_".join((self.DEP, str(j), self.DEP_NAME))] = e_DEPR(dict_input["_".join((self.DEP, str(j), self.DEP_NAME))])


        ################################################
        # EMBEDDINGS FOR THE CONCEPTS MODEL
        ################################################

        for j, f in enumerate(range(0, self.algorithm.wB)):
            
            d_eC["_".join((self.BUFFER, str(j), self.INTERNAL_WORD))] = e_IWC(dict_input["_".join((self.BUFFER, str(j), self.INTERNAL_WORD))])
            d_eC["_".join((self.BUFFER, str(j), self.EXTERNAL_WORD))] = e_EWC(dict_input["_".join((self.BUFFER, str(j), self.EXTERNAL_WORD))])
            d_eC["_".join((self.BUFFER, str(j), self.POSTAG))] = e_PC(dict_input["_".join((self.BUFFER, str(j), self.POSTAG))])
            d_eC["_".join((self.BUFFER, str(j), self.CONCEPT))] = e_CC(dict_input["_".join((self.BUFFER, str(j), self.CONCEPT))]) 
            d_eC["_".join((self.BUFFER, str(j), self.ENTITY))] = e_EC(dict_input["_".join((self.BUFFER, str(j), self.ENTITY))])      
            d_eC["_".join((self.BUFFER, str(j), self.GENERATED_BY))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.GENERATED_BY))])
            d_eC["_".join((self.BUFFER, str(j), self.DEPTH))] = Reshape((1, 1))(dict_input["_".join((self.BUFFER, str(j), self.DEPTH))])    

        for j, f in enumerate(range(0, self.algorithm.wL1)):

            d_eC["_".join((self.L1, str(j), self.INTERNAL_WORD))] = e_IWC(dict_input["_".join((self.L1, str(j), self.INTERNAL_WORD))])
            d_eC["_".join((self.L1, str(j), self.EXTERNAL_WORD))] = e_EWC(dict_input["_".join((self.L1, str(j), self.EXTERNAL_WORD))])
            d_eC["_".join((self.L1, str(j), self.POSTAG))] = e_PC(dict_input["_".join((self.L1, str(j), self.POSTAG))])
            d_eC["_".join((self.L1, str(j), self.CONCEPT))] = e_CC(dict_input["_".join((self.L1, str(j), self.CONCEPT))]) 
            d_eC["_".join((self.L1, str(j), self.ENTITY))] = e_EC(dict_input["_".join((self.L1, str(j), self.ENTITY))]) 
            d_eC["_".join((self.L1, str(j), self.GENERATED_BY))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.GENERATED_BY))])
            d_eC["_".join((self.L1, str(j), self.DEPTH))] = Reshape((1, 1))(dict_input["_".join((self.L1, str(j), self.DEPTH))])
   
        for j in range(0, len(self.FC_DEP)):
            d_eC["_".join((self.DEP, str(j), self.DEP_NAME))] = e_DEPC(dict_input["_".join((self.DEP, str(j), self.DEP_NAME))])
    
        ########################################################################
        #              CREATES THE INPUT VECTOR FOR THE CONCEPT-NN
        ########################################################################
    
        Cinput = []       
        for input in d_eC:
            
            struct, position, type = input.split("_")
            if struct.startswith(self.BUFFER):
                if type in self.FC_B:
                    Cinput.append(d_eC[input])
            elif struct.startswith(self.L1):
                if type in self.FC_L1:     
                    Cinput.append(d_eC[input])
            elif struct.startswith(self.DEP):
                Cinput.append(d_eC[input])
        Cinput_embedded = keras.layers.concatenate(Cinput, name="emb_c", axis=-1)
        
        ########################################################################
        #    CREATES THE INPUT VECTOR FOR THE AMR-COVINGTON TRANSITIONS NN
        ########################################################################

        Tinput = []        
        for input in d_eT:
            
             struct, position, type = input.split("_")
             if struct.startswith(self.BUFFER):
                 if type in self.FT_B:
                     Tinput.append(d_eT[input])
             elif struct.startswith(self.L1):
                 if type in self.FT_L1:     
                     Tinput.append(d_eT[input])
             elif struct.startswith(self.DEP):
                 Tinput.append(d_eT[input])

        Tinput_embedded = keras.layers.concatenate(Tinput, name="emb_t", axis=-1)        

        ########################################################################
        #    CREATES THE INPUT VECTOR FOR THE AMR LABELS NN
        ########################################################################

        Rinput = []
        for input in d_eR:
            
            struct, position, type = input.split("_")
            if struct.startswith(self.BUFFER):
                if type in self.FR_B:
                    Rinput.append(d_eR[input])
            elif struct.startswith(self.L1):
                if type in self.FR_L1: 
                    Rinput.append(d_eR[input])
            elif struct.startswith(self.DEP):
                Rinput.append(d_eR[input])


        Rinput_embedded = keras.layers.concatenate(Rinput, name="emb_r", axis=-1)

        
        ########################################################################
        #          FEED-FORWARD NEURAL NETWORK TO PREDICT AMR CONCEPTS
        ########################################################################
        x = Dense(200)(Cinput_embedded)
        x = Dropout(self.dropout)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.c_classes)(x)
        Coutput = Activation('softmax', name='Coutput')(x)
        self.concepts_model = Model (inputs=inputs,
                                     outputs=[Coutput])           
        self.concepts_model.compile(loss="categorical_crossentropy",
                    optimizer=keras.optimizers.Adam(lr=3e-4, decay=0),  
                    metrics=['accuracy'])   
        self.concepts_model.summary()
        
        #########################################################################
        #          FEED-FORWARD NEURAL NETWORK USED TO PREDICT AMR labels
        #########################################################################
        x = Dense(200, name="d1_r")(Rinput_embedded)
        x = Dropout(self.dropout)(x)
        x = Activation('relu', name="a1_r")(x)
        x = Flatten(name="f1_r")(x)
        x = Dense(self.r_classes, name="d2_r")(x)
        Routput = Activation('softmax', name="Routput")(x)   

        self.relation_model = Model (inputs=inputs,
                                     outputs=[Routput])

        self.relation_model.compile(loss="categorical_crossentropy",
                    optimizer=keras.optimizers.Adam(lr=3e-4, decay=0),
                    metrics=['accuracy'])   
                      
        self.relation_model.summary()

        ##########################################################################
        #  FEED-FORWARD NEURAL NETWORK USED TO PREDICT AMR-COVINGTON TRANSITIONS #
        ##########################################################################
        x = Dense(400, name="d1_t")(Tinput_embedded)
        x = Dropout(self.dropout)(x)
        x = Activation('relu', name="a1_t")(x)
        x = Flatten(name="f1_t")(x)
        x = Dense(200)(x)
        x = Activation('relu')(x)
        x = Dense(self.t_classes, name="d2_t")(x)
        Toutput = Activation('softmax', name="Toutput")(x)

        self.transition_model = Model(inputs=inputs,
                                      outputs=[Toutput])
        self.transition_model.compile(loss="categorical_crossentropy",
                    optimizer=keras.optimizers.Adam(lr=3e-4, decay=0),
                    metrics=['accuracy'])   
        self.transition_model.summary()

        
    #############################################################################
    #                 END OF THE NEURAL NETWORK ARCHITECTURES                   #
    #############################################################################

    """
    Gets the set of possible AMR-labels for the ARC transitions.
    @param rels: An iterable with the relations
    """
    def _get_labels(self, rels):
        
        all_rels = []
        
        for t in self.algorithm.ARC_TRANSITIONS:
            for r in rels:      
                if t == self.algorithm.MULTIPLE_ARC:
                    if utils.MULTIPLE_DIRECTION_RELATION in r:
                        all_rels.append(str(t) + "_" + r)
                else:
                    if utils.MULTIPLE_DIRECTION_RELATION not in r:
                        all_rels.append(str(t) + "_" + r)
                
        return all_rels


    """
    Returns the indexes, weights and dimension of the embedding for an 
    embedding matrices
    @param path_emb: Path to the pretrained embeddings (word2vec format). None, otherwise.
    @param dims: Dimensions for the embeddings. Ignored if path_emb provided
    @param keys: An iterable of string for which we want to create an embedding 
    """
    def _get_embeddings(self, path_emb, dims, keys):
        
        if path_emb is None:
            indexes = {k:self.INIT_REAL_INDEX + i for i, k in enumerate(keys)}
            weights = np.zeros(shape=(len(keys) + len(self.SPECIAL_INDEXES), dims))     
        else:
            indexes, weights, dims = self._read_embedding_file(path_emb)
        
        return indexes, weights, dims
        

    """
    Reads an embedding file in word2vec format.
    Returns a dict with the indexes, the weights and the dimension
    of each embedding
    @param Reads an embedings file in word2vec format
    """
    def _read_embedding_file(self, file_embedding, vocab=None):
        
        if file_embedding is not None:
     
            external_embedding_fp = open(file_embedding, 'r')
            line = external_embedding_fp.readline()
            esize = int(line.split()[1])
                    
            root_word = [random.random() for i in range(0, esize)]
            unk_element_vector = [random.random() for i in range(0, esize)]
            empty_word = [random.random() for i in range(0, esize)]
            vectors = [root_word, unk_element_vector, empty_word]
            iembeddings = {} 
            line = external_embedding_fp.readline()
            iline = len(vectors)
            iembeddings = {"*root*":0, "*unknown-slot*":1, "*empty-slot*":2}
            while line != '': 
                
                word = line.split(' ')[0]
                
                if vocab is None or (word in vocab):
                
                    vector = [np.float32(f) for f in line.strip().split(' ')[1:]] 
                    vectors.append(vector)
                    if word in iembeddings:
                        raise ValueError("Trying to overwrite the value of an existing embedding")
                    iembeddings[word] = iline
                    iline += 1
                line = external_embedding_fp.readline()
                                    
            external_embedding_fp.close()
            
            lookup = np.array(vectors)
            return iembeddings, lookup, esize
                         
        else:
            raise ValueError("Path in file_embedding: ", file_embedding, " does not exist.")


    #############################################################################
    #                FUNCTIONS FOR FEATURES EXTRACTION
    #############################################################################


    """
    Gets the index of a word form in a dictionary of indexes.
    It first looks for the original word form and if it does not match
    it looks for the normalized form and the lemma.
    
    @param d: A dictionary of indexes {form:index} 
    @param amr_word: An instance of an AMRword 
    """
    def word_index(self, d, amr_word):
        
        form = amr_word.form
        norm = amr_word.norm
        lemma = amr_word.lemma
              
        try:
            float(form)
            index = d[norm] if norm in d else self.EMPTY_INDEX
        except ValueError:
            
            if form in d:
                index = d[form]
            elif norm in d:
                index = d[norm]
            elif lemma in d:
                index = d[lemma]
            else:
                index = self.EMPTY_INDEX
                
        return index
    
    
    """
    Gets "dependency relation" features for a word
    Returns a list of indexes of relations
    @param from_B: Iterable of AMRWord instances from B to be compared
    @param is2: Iterable of AMRWord instances from L1 to be compared
    """
    def _features_from_dependency_tree(self, from_B, from_L1):
        
        feature_indexes = []
        for w1, w2 in list(itertools.product(from_B, from_L1)):         
            #If one of the nodes is not valid, not dependency relation
            #can be extracted
            if w1 is None or w2 is None: 
                feature_indexes.extend([self.EMPTY_INDEX, self.EMPTY_INDEX]); 
                continue

            #We look the dependency relations from w1 to w2, i.e. the dependency relation
            #in w1 ---deprel---> w2
            if  w2.head == w1.index:
                deprel = w2.deprel
                deprel_index = self.idep[deprel] if deprel in self.idep else self.UNK_INDEX
                feature_indexes.append(deprel_index)

            else:
                feature_indexes.append(self.EMPTY_INDEX)
                      
            #We look the dependency relations from w2 to w1, i.e. the dependency relation
            #in w2 ---deprel---> w1                      
            if w1.head == w2.index:
                deprel = w1.deprel
                deprel_index = self.idep[deprel] if deprel in self.idep else self.UNK_INDEX
                feature_indexes.append(deprel_index)
            else:
                feature_indexes.append(self.EMPTY_INDEX)
        
        return feature_indexes




    """
    Returns the list of feature for a given amr_entry
    @param amr_entry: An instance of an AMR_entry
    @param c: An instance of the CovingtonConfiguration class
    """
    def feature_indexes_for_entry(self, entry, c, separator_indexes):
        
        #Assigning default values
        iword = self.EMPTY_INDEX
        iword_external = self.EMPTY_INDEX
        ipostag = self.EMPTY_INDEX
        ientity = self.EMPTY_INDEX
        iconcept = self.EMPTY_INDEX
   #     iedge = self.EMPTY_INDEX
        iedge = self.EMPTY_INDEX #Index of the last assigned edge
        #Leftmost (LM)the head (h), child and grand-child (gc) concepts for an entry
        lm_h, lm_child, lm_gc = self.EMPTY_INDEX, self.EMPTY_INDEX, self.EMPTY_INDEX        
        rm_h, rm_child, rm_gc = self.EMPTY_INDEX, self.EMPTY_INDEX, self.EMPTY_INDEX
        
        n_children = 0
        n_heads = 0 
        depth = -1
        created_by = -1
#       n_generated = 0
        concept_type = -1
        previous_sentences = 0 

        #THESE ARE NOT BEING USED AT THE MOMENT
        #Leftmost (LM)the head (h), child and grand-child (gc) word for an entry
        lmh_w, lmc_w, lmcc_w = self.EMPTY_INDEX, self.EMPTY_INDEX, self.EMPTY_INDEX
        #Leftmost (LM)the head (h), child and grand-child (gc) postags for an entry
        amr_entry, type = entry[0], entry[1] #type indicates if it is a feature from the buffer of l1.
        
        # It is not a feature from the buffer or L1
        if type == self.algorithm.OTHER_FEATURE:   
            n_heads, n_children, depth = -1, -1, -1
        else:
            previous_sentences = sum([1 for index in separator_indexes
                                      if index < amr_entry.word.index])           
             
            iword = self.word_index(self.i_iforms, amr_entry.word)           
            iword_external = self.word_index(self.i_eforms, amr_entry.word)  

            postag = amr_entry.word.pos
            if postag is None:
                ipostag = self.ROOT_INDEX
            else:
                ipostag = self.ipos[postag] if postag in self.ipos else self.UNK_INDEX
            
            try:
                ientity = self.ientity[amr_entry.word.entity]
            except KeyError:  # This type of entity was not found on the training set
                pass    
            
            
            if amr_entry.is_node:
                
                iconcept = self.iconcepts[amr_entry.node.concept] if amr_entry.node.concept in self.iconcepts else self.UNK_INDEX
                iedge = self.iedge[amr_entry.node.last_rel_as_head] if amr_entry.node.last_rel_as_head in self.iedge else self.UNK_INDEX
                created_by = 0 if amr_entry.node.created_by == self.algorithm.CONFIRM else 1    
                n_heads = len(amr_entry.node.pred_relations) 
                depth = amr_entry.node.depth(c,[])
                concept_type = amr_entry.node.concept_type # Calculating the iconcept type   
                
                # Leftmost head
                lm_hn = amr_entry.node.lm_h(c, l=1)
                if lm_hn is None: lm_h, lmh_w = self.EMPTY_INDEX, self.EMPTY_INDEX
                else:
                    lm_h = self.iconcepts[lm_hn.concept] if lm_hn.concept in self.iconcepts else self.UNK_INDEX
                    lmh_w = self.word_index(self.i_iforms, lm_hn.word)
                n_children = len(amr_entry.node.pred_children)
                
                # Leftmost child
                lm_cn = amr_entry.node.lm_child(c, l=1)
                if lm_cn is None: lm_child, lmc_w = self.EMPTY_INDEX, self.EMPTY_INDEX
                else:
                    lm_child = self.iconcepts[lm_cn.concept] if lm_cn.concept in self.iconcepts else self.UNK_INDEX
                    lmc_w = self.word_index(self.i_iforms, lm_cn.word)
                    
                # Grand leftmost child
                lm_gcn = amr_entry.node.lm_child(c, l=2)
                if lm_gcn is None: lm_gc = self.EMPTY_INDEX
                else:
                    lm_gc = self.iconcepts[lm_gcn.concept] if lm_gcn.concept in self.iconcepts else self.UNK_INDEX
                    lmcc_w = self.word_index(self.i_iforms, lm_gcn.word)

                # Rightmost head
                rm_hn = amr_entry.node.rm_h(c, l=1)
                if rm_hn is None: rm_h = self.EMPTY_INDEX
                else:
                    rm_h = self.iconcepts[rm_hn.concept] if rm_hn.concept in self.iconcepts else self.UNK_INDEX
        
                # Rightmost children
                rm_cn = amr_entry.node.rm_child(c, l=1)
                if rm_cn is None: rm_c = self.EMPTY_INDEX
                else:
                    rm_c = self.iconcepts[rm_cn.concept] if rm_cn.concept in self.iconcepts else self.UNK_INDEX                             
                    
                # Grand rightmost children
                rm_gcn = amr_entry.node.rm_child(c, l=2)
                if rm_gcn is None: rm_gc = self.EMPTY_INDEX
                else:
                    rm_gc = self.iconcepts[rm_gcn.concept] if rm_gcn.concept in self.iconcepts else self.UNK_INDEX                    
                    
                                 
        return iword, iword_external, ipostag, iconcept, ientity, lm_h, lm_child, lm_gc, n_heads, n_children, created_by, iedge, depth, concept_type, rm_h, rm_child, rm_gc, lmh_w, lmc_w, lmcc_w, previous_sentences 


    """
    Transforms an input configuration to a list of indexes representing the
    features.
    
    It extracts features from AMRWord in L1 and the Buffer.
    @param input: (A list AMRListEntries from B, A list of AMRListEntries from L1)
    @param c: A CovingtonConfiguration instance
    """
    def conf_to_features(self, input, c, separator_indexes):    
    
        feature_indexes = []
        #Getting features from the Buffer
        for b_index in input[0]:
            for index in self.feature_indexes_for_entry(b_index, c, separator_indexes):
                feature_indexes.append(index)            

        #Getting features from L1
        for l1_index in input[1]:
            for index in self.feature_indexes_for_entry(l1_index, c, separator_indexes):
                feature_indexes.append(index)
            
        # Features from the dependency tree, between B and L1. 
        #TODO: In the current version, this is handled on an ad-hoc way and not together
        #with the rest of the features.
        try:
            b_indexes = [c.sequence[c.b + pos].word if c.b + pos < len(c.sequence) else None 
                         for pos in self.B_DEPS]
            l1_indexes = [c.sequence[c.l1 - pos].word if c.l1 - pos >= 0 else None 
                          for pos in self.L1_DEPS]
        except IndexError:
            raise NotImplementedError("Error when extracting dependency features from B/L1")
         
        dep_inputs = self._features_from_dependency_tree(b_indexes, l1_indexes)   
        feature_indexes.extend(dep_inputs)
        
        return feature_indexes

     

     
    ############################################################################
    #                               TEST PHASE                                 #
    ############################################################################
     
    #TODO: This is not elegant. Change it.
    """
    Auxiliary function to determine which features are actually used by the classifier
    during the training phase
    """
    def _get_valid_indexes(self, input, picks, init_index, global_feature=False):
        

        indexes = []
        if global_feature:
            return [np.array(e).reshape(1, -1) for e in input[-len(self.FEATURE_DEPS):] ]    
        else:   
            for pick in picks:
                indexes.append(self.FEATURES.index(pick))
            return [np.array(input[i + init_index]).reshape(1, -1) for i in indexes]    


    """
    Checks a set of hooks to see whether the predict label is valid in an AMR graph
    """
    def _is_valid_transition(self, arc):
         
        h = arc[0]
        r = arc[1]
        d = arc[2]
            
        if r in ["polarity"]:
            return d == "-"        
        if r in ["month"]:    
            d_con = (d in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] 
                      or d.lower() in ["january", "february", "march", "april", "may", 
                                       "july", "august", "september", "november",
                                       "december"])
            h_con = h == "date-entity"   
            return h_con and d_con    
        if r in ["weekday"]:           
            d_con = d in ["monday", "tuesday", "wednesday", "thrusday", "friday",
                          "saturday", "sunday"] 
            h_con = h == "date-entity"        
            return h_con and d_con
        if r in ["year", "year2", "decade", "century"]:    
            try:
                int(d)
                d_con = True
            except:
                d_con = False
            h_con = h == "date-entity" 
            return h_con and d_con         
        if r in ["value", "wiki"]:
            d_con = (d.startswith("\"") and d.endswith("\"")) or d.isdigit()
            return d_con
        if r.startswith("ARG"):
            narg = int(r.split("-")[0][-1])
            real_head = h if not r.endswith("-of") else d        
            if real_head in self.args_rules:                
                return self.args_rules[real_head][narg] == 1                
        return True


    """
    Gets the first valid relation label for the current state
    @param t: The predicted transition
    @param c: A CovingtonConfiguration instance
    @param oR: A numpy vector of scores for every possible relation label. 
    The score at index i is the score for label i.
    """
    def _get_first_valid_label(self, t, c, oR):
        
        valid = False
        while not valid:
                      
            max_i = argmax(oR)
            l = self.rels_i[max_i] 
            arc = self._get_arc(t, l, c)      
            l = arc[1]
            
            if self._is_valid_transition(arc):
                if t == self.algorithm.MULTIPLE_ARC:
                    if utils.MULTIPLE_DIRECTION_RELATION in l:
                        valid = True
                else:
                    if not utils.MULTIPLE_DIRECTION_RELATION in l:
                        valid = True
                    
            oR[0][max_i] = 0.
        
        return l


    """
    Gets the first valid concept for the current state
    @param oC: A numpy vector of scores for every possible concept learned in the training phase. 
    The score at index i is the score for concept i.
    @param last_concept: Last generated concept. Used to avoid duplications
    """
    def _get_first_valid_concept(self, oC, last_concept):
        concept = self.nodes_i[argmax(oC)]
        valid = False
        while not valid:
     
            if concept != last_concept:
                valid = True
            else:
                oC[0][argmax(oC)] = 0.
                concept = self.nodes_i[argmax(oC)]
        
        return concept        


    """
    Returns an edge of the form (head_concept, relation, dependent_concept).
    Elements of the tuple are strings
    @param t: A transitions id
    @param r: Name of the relation
    @param c: A CovingtonConfiguration instance
    """
    def _get_arc(self, t, r, c):
        
        if t == self.algorithm.LEFT_ARC:
            
            h = c.sequence[c.b].node
            d = c.sequence[c.l1].node
            is_arg = r.startswith("ARG")
            is_of = r.endswith("-of")
            
            if is_arg:

                 if not h.is_propbank and d.is_propbank and not is_of:
                     return (h.concept, r + "-of", d.concept)
                 
                 if h.is_propbank and not d.is_propbank and is_of:
                     return (h.concept, r.replace("-of", ""), d.concept)
                                 
            return  (h.concept, r, d.concept)
        
        
        if t == self.algorithm.RIGHT_ARC:
            
            h = c.sequence[c.l1].node
            d = c.sequence[c.b].node
            is_arg = r.startswith("ARG")
            is_of = r.endswith("-of")
            
            if is_arg:
                 
                if not h.is_propbank and d.is_propbank and not is_of:
                    return (h.concept, r + "-of", d.concept)
                 
                if h.is_propbank and not d.is_propbank and is_of:
                    return (h.concept, r.replace("-of", ""), d.concept)
                 
            return (h.concept, r, d.concept)

        # We are not considering rules for MULTIPLE-ARC
        return None

    

    #TODO: Not elegant, does not genealize well. Change it.
    """
    Auxiliary function to determine if the node root is going to be single or multiple-rooted
    """
    def _predict_root(self, separator_indexes):
        
        if len(separator_indexes) > 0:
            
            root_word = utils.AMRWord(-1, utils.ID_MULTISENTENCE, None, None, None, None, None, None, 'ROOT_ENTITY')
            root_id = utils.ID_ROOT_ID
            root_node = utils.AMRNode(root_word.form, -1, -1, id=root_id, relations=set([]), children=set([]),
                                pred_relations=set([]), pred_children=set([]),
                                created_by="CONFIRM",
                                last_rel_as_head=None,
                                unaligned=False,
                                originated_from=root_word)
            
            root_entry = utils.AMRListEntry(root_word, node=root_node,
                                      edges=set([]))   
            
            return root_entry    

        else:
            root_word = utils.AMRWord(-1, utils.ID_ROOT_SYMBOL, None, None, None, None, None, None, 'ROOT_ENTITY')
            root_id = utils.ID_ROOT_ID
            root_node = utils.AMRNode(root_word.form, -1, -1, relations=set([]), children=set([]), id=root_id,
                                pred_relations=set([]), pred_children=set([]),
                                created_by="CONFIRM",
                                last_rel_as_head=None,
                                unaligned=False,
                                originated_from=root_word)
            
            root_entry = utils.AMRListEntry(root_word, node=root_node,
                                      edges=set([]))  
            
            return root_entry


    #TODO: This is not an elegant solution. Change it.
    """
    Transforms a raw input into the representation needed for each classifier (T,C,R).
    @param c: A CovingtonConfiguration instance
    """
    def _conf_to_predict_input(self, c, separator_indexes):
        
        XT, XR, XC = [], [], []
        input = self.algorithm.get_state_info(c)           
        input = self.conf_to_features(input, c, separator_indexes)

        for b in range(self.algorithm.wB):                
            XT.extend(self._get_valid_indexes(input, self.FT_B, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * b))
            XC.extend(self._get_valid_indexes(input, self.FC_B, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * b))
            XR.extend(self._get_valid_indexes(input, self.FR_B, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * b))
            
        for l1 in range(self.algorithm.wL1):                
            XT.extend(self._get_valid_indexes(input, self.FT_L1, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * (l1 + self.algorithm.wB)))
            XC.extend(self._get_valid_indexes(input, self.FC_L1, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * (l1 + self.algorithm.wB)))
            XR.extend(self._get_valid_indexes(input, self.FR_L1, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * (l1 + self.algorithm.wB)))
        
        XT.extend(self._get_valid_indexes(input, self.FT_DEP, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * (self.algorithm.wB + self.algorithm.wL1), True))
        XC.extend(self._get_valid_indexes(input, self.FC_DEP, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * (self.algorithm.wB + self.algorithm.wL1), True))
        XR.extend(self._get_valid_indexes(input, self.FR_DEP, (len(self.FEATURES) - len(self.FEATURE_DEPS)) * (self.algorithm.wB + self.algorithm.wL1), True))
        
        return XT, XR, XC




    """
    Returns if the word is a named-entity
    NOTE: If so, it will receive an special treatment during the parsing phase.
    The AMR parser will try to apply a set of hooks
    """
    def word_is_entity(self,amr_word):
        is_target_entity = amr_word.entity in ["DATE","PERSON","LOCATION","ORGANIZATION", "PERCENT","MONEY", "TIME"]
        is_nationality = amr_word.form.lower() in self.r_concept_model.nationalities #nationalities
        is_place = False
        if amr_word.form in self.r_concept_model.countries or amr_word.form in self.r_concept_model.cities or amr_word.form in self.r_concept_model.states:
            is_place = True        
            
        is_verbalization = (amr_word.form.lower() in self.r_concept_model.verbalizations or amr_word.norm.lower() in self.r_concept_model.verbalizations
                            or amr_word.lemma.lower() in self.r_concept_model.verbalizations)
        
        is_likely_multinode = (amr_word.form,"O") in self.r_concept_model.graph_templates
        
        is_negation = (amr_word.form in self.r_concept_model.negations
                       or amr_word.norm in self.r_concept_model.negations 
                       or amr_word.lemma in self.r_concept_model.negations
                       or amr_word.form in ["without", "never"])
                
        return is_target_entity or is_nationality or is_place or is_verbalization or is_likely_multinode or is_negation# or is_abbr # or is_common_unaligned



    #Jun in case we forget how to process this type of date format
    def is_seqdate(self,seqdate):
        try:
            int(seqdate)
            is_seqdate = len(seqdate) == 6 or len(seqdate) == 8
            return is_seqdate
        except ValueError:
            return False
    
    def preprocess_seqdate(self, seqdate):
        
        if len(seqdate) == 6:
            year = seqdate[0:2]
            month = seqdate[2:4]
            day = seqdate[4:6]
             
            if year[0] == "0":
                return "20"+year+"-"+month+"-"+day
            elif year[0] == "1":
                return "20"+year+"-"+month+"-"+day
            else:
                return "19"+year+"-"+month+"-"+day
             
        elif len(seqdate) == 8:
            year = seqdate[0:4]
            month = seqdate[4:6]
            day = seqdate[6:8]
            return year+"-"+month+"-"+day





    def phrase_generates_subgraph(self, amr_words):
        
        curdict, curgraph = self.r_concept_model.multiword_graph_templates, None
        first = None
        mached = []
        components = []
        for amr_word in amr_words:
            
            try:
                curdict, curgraph = curdict[amr_word.form]
                components.append(amr_word)
                mached.append(amr_word.form)
                if first is None: 
                    first=amr_word
            except KeyError:
                if curgraph is not None:
                    return True, components
                return False, None
        if curgraph is not None:
            return True, components    
        
        return False, None


    #TODO: This solution is not elegant
    """
    It determines the subsequence of words from amr_words[0] that make a named-entity.
    It is used to manage NERs during parsing.
    @param amr_words: A list of AMRWord instances
    """
    def get_components(self,amr_words):
        
        first = amr_words[0]
        entity_entries = [amr_words[0]]
    
        for e in amr_words[1:]:
            if e.entity == amr_words[0].entity and e.entity != "O":
                entity_entries.append(e)
            else:
                break
         
        return entity_entries




    """
    Predicts an AMR graph from a raw input and a pretrained model
    @param amr_path: 
    @param lookup_path
    """
    def predict(self, path_to_samples):
    
        output_graphs = []
        with codecs.open(path_to_samples, 'rb') as f:
            data = pickle.load(f)  
                
        n_multisentence = 0
    
            
        for iamrgraph, amrwords in enumerate(data):
                
            #We create the initial configuration for iamrgraph
            l1 = 0
            b = 1
            arcs = []
            separator_indexes = self._get_indexes_for_separator_tokens(amrwords)
            
            if len(separator_indexes) > 0:
                n_multisentence+=1
            print "Predicting graph #", iamrgraph
              
            #Workaround to process words that are named-entities
            components = []
            for k,word in enumerate(amrwords):
                
                if self.is_seqdate(word.form):
                    word.form = self.preprocess_seqdate(word.form)
                    word.entity = "DATE"
                  
                if word in components: 
                    continue
                  
                is_multi,mult_components = self.phrase_generates_subgraph(amrwords[k:])
                
                if is_multi:
                    components = mult_components
                    word.add_components(mult_components)
                
                elif self.word_is_entity(word):
                   index = amrwords.index(word)       
                   components = self.get_components(amrwords[index:])
                   word.add_components(components)
                   
            
            sequence = []
            sequence.append(self._predict_root(separator_indexes))
            sequence.extend([utils.AMRListEntry(amrword, None) for amrword in amrwords])
            sequence_words = [e.word for e in sequence]
            nodes = set([sequence[l1].node])
            c = CovingtonConfiguration(l1, b, sequence, nodes, arcs,
                                       pred_concepts={})    
             
                             
            last_concept = None
            nid = 0
            while not self.algorithm._is_final_state(c):
                XT, XR, XC = self._conf_to_predict_input(c, separator_indexes)
    
                # Generate fixed subgraph from the current configuration
                if self.r_concept_model.configuration_starts_graph_template(c):
                    template_subgraph = self.r_concept_model.transition_sequence_from_template(c)
                    if template_subgraph is not None:
                        for t, l in template_subgraph:
                            if t in self.algorithm.NODE_TRANSITIONS:
                              #TODO: workaround
                              if c.b >= len(c.sequence): 
                                  break    
                              nid += 1              
                            c = self.algorithm.update_state((t, l), c, nid)              
                        continue # We check if it is final state and  continue predict remaining subgraphs
    
                #If no template matched, we predict the next transition
                oT = self.transition_model.predict(XT)[0]          
                t = self.algorithm.predict(c, oT)  
                #TODO: There was a problem on assigning MULTIPLE-ARCs, think it is solved
#                 if t == self.algorithm.MULTIPLE_ARC:
#                     t = self.algorithm.LEFT_ARC
    
                #If we predict a LEFT-ARC, RIGHT-ARC or MULTIPLE-ARC
                #We need to predict the label for the relationship between l1 and b
                if t in self.algorithm.ARC_TRANSITIONS:           
                    oR = self.relation_model.predict(XR)
                    l = self._get_first_valid_label(t,c,oR)
                    c = self.algorithm.update_state((t, l), c, nid)
                #If we predict a CONFIRM or BREAKDOWN  
                elif t in self.algorithm.NODE_TRANSITIONS:
                    nid += 1  
                    heuristic_transitions_to_concept = None
                    if not c.sequence[c.b].is_node:
                        heuristic_transitions_to_concept = self.r_concept_model.word_to_concept(c.sequence[c.b].word) 
    
                    if heuristic_transitions_to_concept is not None:
                        for t, l in heuristic_transitions_to_concept:
                            c = self.algorithm.update_state((t, l), c, nid)
                            if t in self.algorithm.NODE_TRANSITIONS:
                                nid += 1
                    else:                           
                        oC = self.concepts_model.predict(XC)
                        last_concept = self._get_first_valid_concept(oC, last_concept)
                        c = self.algorithm.update_state((t, last_concept), c, nid)
                #If we predict a REDUCE, NO-ARC or SHIFT       
                else:
                    c = self.algorithm.update_state((t, None), c, nid)
    
            output_graphs.append(utils.AMRGraph(sequence=c.sequence, nodes={n.id: n for n in c.N}, A=c.A, id="#:: "+str(iamrgraph),
                                                            original_sequence=sequence_words))
              
        return output_graphs   



    ############################################################################
    #                            TRAINING PHASE                                #
    ############################################################################

    """
    Applies transition_sequence, step by step, to the given configuration
    @param transition_sequence: A list of (transition,relation) representing the
    set of transitions to be applied. relation must be node 
    (e.g. in the case of a NO-ARC transition)
    @param c: A CovingtonConfiguration instance
    @return The new configuration
    """
    def _apply_transition_sequence(self, c, transition_sequence):

        for t, a in transition_sequence:
            if t in self.algorithm.NODE_TRANSITIONS:

               #TODO: This is a WORKAROUND
               if c.b >= len(c.sequence): 
                  return c
               nid = None      
            else:
                nid = None
            
            c = self.algorithm.update_state((t, a), c, nid)
    
        return c
        
    """
    Determines if a given configuration should be included as a part of the training set.
    It currently skips breakdown transitions that are not unaligned in the gold file
    @param c: A CovingtonConfiguration sample
    @param t: A transition id
    @param amrGraph: An instance of an gold AMR graph
    @param node_id: Why do we need this way??
    """
    def _include_sample_for_training(self, c, t, amrGraph, node_id):
        
        unaligned = (c.b < len(c.sequence) and 
                      c.sequence[c.b].is_node and
                      node_id in amrGraph.nodes and 
                      t in [self.algorithm.CONFIRM, self.algorithm.DESGLOSE] and
                      amrGraph.nodes[node_id].unaligned)
  
        return not (t == self.algorithm.DESGLOSE and unaligned)
    
    
    """
    Given a sentence, it returns the indexes where the separator tokens occur.
    """
    def _get_indexes_for_separator_tokens(self, sentence):
        separator_indexes = []
        for j, token in enumerate(sentence[0:-1]):
            if token.form in self.SENTENCE_SEPARATOR_TOKENS:
                separator_indexes.append(j)
        return separator_indexes
    

    """
    Prepares the data to be used for training
    """
    def load_data(self, shuffledData, training=True):
    
        #Inputs used to train the (T)ransitions, (C)oncept and (R)elation classifiers
        X_T = []
        X_C = []
        X_R = []
        
        y_t = []
        y_r = []
        y_c = []
        
        
        y_c_alone = []
        
        X_C_all = []
        y_c_all = []
        

        y_r_alone = []
        
        X_R_all = []
        y_r_all = []
        
        for iamrgraph, amrGraph in enumerate(shuffledData):
            if iamrgraph % 100 == 0:
                print "Reading #",iamrgraph, amrGraph.id
         #   len_sequence = len(amrGraph.sequence)
            #Initial configuration
            sequence_words = [e.word for e in amrGraph.sequence[0:]]
            separator_indexes = self._get_indexes_for_separator_tokens(sequence_words)
            l1 = 0
            b = 1
            arcs = []               
            nodes = set([amrGraph.sequence[l1].node])
            c = CovingtonConfiguration(l1, b, amrGraph.sequence, nodes, arcs,
                                             pred_concepts={})
 
            arcs_as_indexes = set([a.unlabeled_as_indexes for a in amrGraph.A])  
            labeled_arcs_as_indexes = [a.labeled_as_indexes for a in amrGraph.A]   
            d_n_r = {nid: amrGraph.nodes[nid].relations for nid in amrGraph.nodes}
             
            while not self.algorithm._is_final_state(c):

                if self.r_concept_model.configuration_starts_graph_template(c):
                    heuristic_node = self.r_concept_model.transition_sequence_from_template(c, 
                                                                                            training_phase=True)
                    if heuristic_node is not None:
                        c = self._apply_transition_sequence(c, heuristic_node)
                        if c.b < len(c.sequence) and c.sequence[c.b].is_node: #and modified
                            indexed = c.sequence[c.b].indexed_at
                            nid = amrGraph.node_id((indexed[0], indexed[1]), c.node_ids)
                            c.sequence[c.b].node.id = nid     
                        continue

                
                action = self.algorithm.true_static_oracle(c, amrGraph)
                t, a = action[0], action[1]
                input = self.algorithm.get_state_info(c)                
                input = self.conf_to_features(input, c, separator_indexes)
                indexed = c.sequence[c.b].indexed_at
                node_id = amrGraph.node_id((indexed[0], indexed[1]),
                                           c.node_ids)       
                c = self.algorithm.update_state((t, a), c, node_id)     
                if self._include_sample_for_training(c, t, amrGraph, node_id): 
                    
                    X_T.append(input)
                    y_t.append(self.T[t])
                    
                    if t in self.algorithm.ARC_TRANSITIONS:
                        
                        if (not self.options.expanded_rels):
                            if a in self.R:                    
                                y_r.append(self.R[a])
                                y_r_alone.append(self.R[a])
                                X_R.append(input)
                        
                            X_R_all.append(input)
                            
                            if a in self.rels_all:
                                y_r_all.append(self.rels_all[a])
                            else:
                                self.rels_all[a] = len(self.rels_all)
                                y_r_all.append(self.rels_all[a])
                                
                        else:            
                            a = str(t) + "_" + a
                            if a in self.R:                    
                                y_r.append(self.R[a])
                                y_r_alone.append(self.R[a])
                                X_R.append(input)
                        
                            X_R_all.append(input)
                            
                            if a in self.rels_all:
                                y_r_all.append(self.rels_all[a])
                            else:
                                self.rels_all[a] = len(self.rels_all)
                                y_r_all.append(self.rels_all[a])
                            

                    if t in self.algorithm.NODE_TRANSITIONS:   
                                                
                        if (a not in self.C_unk):
                            
                            if a in self.nodes:
                                concept = self.nodes[a]
                                X_C.append(input)
                                y_c.append(concept)
                                y_c_alone.append(concept)    
                      
                      
                        X_C_all.append(input)
                        if a in self.nodes_all:
                            y_c_all.append(self.nodes_all[a])
                        else:
                            # To deal with concepts that only occur in the dev set
                            self.nodes_all[a] = len(self.nodes_all)
                            y_c_all.append(self.nodes_all[a])
  
            if amrGraph.sequence[0].node == utils.ID_ROOT_SYMBOL and len(amrGraph.sequence[0].node.pred_children) != 1:
                print "BUG in assigning single root node"
                exit()

        
        X_T = np.array(X_T)
        X_C = np.array(X_C)
        X_R = np.array(X_R)
        X_C_all = np.array(X_C_all)
        X_R_all = np.array(X_R_all)
        
        return X_T, y_t, X_R, y_r_alone, X_C, y_c_alone,X_C_all, y_c_all


    """
    Auxiliary function that prepares the data to be trained on a Keras MLP
    @param model: Keras classifier model instance
    @param name: Name given to the classifier
    @param epochs: Number of epochs
    @param X: Training sample data
    @param y: Training sample labels
    @param X_dev: Development sample data
    @param y: Development sample labels
    @param n_classes:
    @param weights: Dictionary to weight the classes. None otherwise
    """
    def _train_model(self, model, name, epochs, X, y, X_dev, y_dev, n_classes, weights=None):
        
        indexes_to_randomize = [] 
        for nw in range(0, self.algorithm.window_size):
            indexes_to_randomize.extend(target + nw * (len(self.FEATURES) - len(self.FEATURE_DEPS)) for target in self.INDEXES_TO_RANDOMIZE)
        
        data_generator = mlp_utils.ConceptDataGenerator(self.batch_size, X, y, n_classes, True,
                                                          indexes_to_randomize=indexes_to_randomize,
                                                          not_to_randomize=[self.EMPTY_INDEX, self.ROOT_INDEX],
                                                          randomized_prob=self.switch_to_unk_c_prob,
                                                          randomized_value=self.UNK_INDEX)
        
        dev_data_generator = mlp_utils.ConceptDataGenerator(self.batch_size, X_dev, y_dev, n_classes, False,
                                                          indexes_to_randomize=indexes_to_randomize,
                                                          not_to_randomize=[self.EMPTY_INDEX, self.ROOT_INDEX],
                                                          randomized_prob=self.switch_to_unk_c_prob,
                                                          randomized_value=self.UNK_INDEX)        
        
        save_model_path = self.options.params_model
        components = save_model_path.rsplit("/", 1)
        save_model_path = save_model_path.replace(components[-1], name + "." + components[-1])
        save_callback = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_loss', verbose=0,
                                                        save_best_only=True, save_weights_only=False, mode='auto', period=1)
        
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
        LR_cb = mlp_utils.SGDLearningRateTracker()

        history = model.fit_generator(data_generator.generate(),  
                                 steps_per_epoch=len(y) / self.batch_size, epochs=epochs, verbose=1,
                                 class_weight=weights,
                                 callbacks=[save_callback, early_stopping_cb, LR_cb],  
                                 validation_data=dev_data_generator.generate(),
                                 validation_steps=len(y_dev) / self.batch_size,
                                 shuffle=True,)
        return history


    """
    Trains an AMR parser
    @param training_aligned: Path to the training aligned (*.aligned) file
    @param dev_aligned: Path to the development aligned (*.aligned) file
    """
    def train(self, training_aligned, dev_aligned):
        with codecs.open(training_aligned) as f:
            shuffledData = pickle.load(f)
        
        random.shuffle(shuffledData)
        
        XT, yT, XR, yR, XC, yC, _, _ = self.load_data(shuffledData, True)
        
        with codecs.open(dev_aligned) as f:
            dev_data = pickle.load(f)

        XdT, ydT, XdR, ydR, XdC, ydC, _, _ = self.load_data(dev_data, False)
        
        weightsT = None
        hT = self._train_model(self.transition_model, "T", self.epochs, XT, yT, XdT, ydT, self.t_classes, weightsT)
        self.save_plot(hT, "Transitions")   
        hR = self._train_model(self.relation_model, "R", self.epochs, XR, yR, XdR, ydR, self.r_classes)
        self.save_plot(hR, "Relations")
        hC = self._train_model(self.concepts_model, "C", self.epochs, XC, yC, XdC, ydC, self.c_classes)
        self.save_plot(hC, "Concepts")          


#         print "TRANSITIONS"
#         self.evaluate_model_on_gold(XdT, ydT, self.t_classes, self.transition_model, self.eval_function_generic, self.actions_i)
#         print "RELATIONS"
#         self.evaluate_model_on_gold(XdR, ydR, self.r_classes, self.relation_model, self.eval_function_generic, self.rels_i)
#         print "CONCEPTS"
#         self.evaluate_model_on_gold(XdC, y_c_alone_dev, self.c_classes, self.concepts_model, self.eval_function_concepts, self.nodes_i)
#  
#         self.nodes_all_i = {self.nodes_all[n]: n for n in self.nodes_all}
#         self.evaluate_model_on_gold(X_C_all_dev, y_C_all_dev, self.c_classes, self.concepts_model, self.eval_function_concepts, self.nodes_all_i)

  
  
  
  
  
    #############################################################################
    #            EVALUATION FUNCTIONS OVER GOLD CONFIGURATIONS                  #
    #############################################################################


    def eval_function_concepts(self, results, y, lookup):
        
        accuracy = 0.
        length = 0.
        sampleid = 0
   
        for j, (input, output) in enumerate(results):

            if y[sampleid] == 0:
                pass
            else:
                length += 1                  
                if output == lookup[y[sampleid]]:
                    accuracy += 1    
            sampleid += 1
            
        print "Concept accuracy:", accuracy / length


    def eval_function_generic(self, results, y, lookup):
        
        accuracy = 0
        relaxed_accuracy = 0.
        length = 0.
        sampleid = 0
        
        p = {}
        
        y_test = []
        y_pred = []
   
        
        for j, (input, output) in enumerate(results):

            length += 1  

            cur_y = y[sampleid] 
            if cur_y not in p:
                p[cur_y] = (0, 0.)

            if output == lookup[y[sampleid]]:
             #   print output, self.nodes_i[y[sampleid]] 
                accuracy += 1    
                p[cur_y] = (p[cur_y][0] + 1, p[cur_y][1])
            
            y_test.append(lookup[y[sampleid]])
            y_pred.append(output)
            
            p[cur_y] = (p[cur_y][0], p[cur_y][1] + 1) 
                 

            sampleid += 1
              
        print "Accuracy:", accuracy / length
        
        p = {cat:p[cat][0] / p[cat][1] for cat in p}
        
        print "Accuracy per class", sorted(p.items(), reverse=True)


        precision, recall, fscore, support = score(y_test, y_pred)
        
        x = PrettyTable()
        x.add_column("Label", range(len(precision)))
        x.add_column("Precision", precision)
        x.add_column("Recall", recall)
        x.add_column("F-score", fscore)
        x.add_column("Support", support)

        print x
        
    def evaluate_model_on_gold(self, X, y , n_classes, trained_model, eval_function, lookup):

        batch_size = 1
        dev_data_generator = mlp_utils.ConceptDataGenerator(batch_size, X, y, self.c_classes
                                                              , False,
                                                          indexes_to_randomize=None,
                                                          not_to_randomize=None,
                                                          randomized_prob=None,
                                                          randomized_value=None)     
          
        outputs = []
 
        i = 0
        for x in dev_data_generator.generate_inputs():

            out = trained_model.predict(x)
            if out != []:
                for o in out:
                    y_pred = lookup[argmax(o)]
                    outputs.append((x, y_pred))  
            i += 1
            if i >= len(y) / (batch_size):
                break
              
        eval_function(outputs, y, lookup)



    def save_plot(self, history, title):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(title)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.savefig(self.options.output + os.sep + title + '.png')
        plt.close()

        loss_history = np.array(history.history["loss"])
        val_loss_history = np.array(history.history["val_loss"])
        acc_history = np.array(history.history["acc"])
        val_acc_history = np.array(history.history["val_loss"])
        np.savetxt(self.options.output + os.sep + title + "loss_history.txt", loss_history, delimiter=",")
        np.savetxt(self.options.output + os.sep + title + "val_loss_history.txt", val_loss_history, delimiter=",")
        np.savetxt(self.options.output + os.sep + title + "acc_history.txt", acc_history, delimiter=",")
        np.savetxt(self.options.output + os.sep + title + "val_acc_history.txt", val_acc_history, delimiter=",")
