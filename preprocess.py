from collections import OrderedDict
from nltk.tag import StanfordNERTagger
from collections import Counter
from utils import *

import codecs
import numpy as np
import ConfigParser
import re
import codecs
import sys
import os
import subprocess
import tempfile
from numpy.ma.core import ids
import penman
import itertools
from tempfile import NamedTemporaryFile
import urllib2
import nltk
import string
import random
import copy
import argparse
import pickle


"""
Reads an ALIGNED file after running ALIGN.sh
"""

ID_JAMR_GRAPH_ID = "# ::id"
ID_JAMR_TOK = "# ::tok"
ID_JAMR_NODE= "# ::node"
ID_JAMR_ROOT = "# ::root"
ID_JAMR_EDGE= "# ::edge"
ID_MULTISENTENCE = "multi-sentence"
ID_ROOT_SYMBOL = "*root*"
ID_ROOT_REL = "*root-rel*"
ID_ROOT_ID = "-1"

ID_NEW_ROOT_REL = "snt"

MULTIPLE_DIRECTION_RELATION = "|"
COMPOSITE_RELATION = "+"


DUMMY_ROOT = 0
UD_CTAG_VERB = "VERB"
UD_HEAD_COLUMN = 6
UD_CTAG_COLUMN = 3
UD_ID_COLUMN = 0


"""
A simple wrapper for UDpipe
"""
class UDPipe(object):
    
    def __init__(self,path_model, path_udpipe):
        self.path_model = path_model
        self.udpipe = path_udpipe
    

    def run(self,text,options=' --input horizontal --tag --parse'):
        
        f_temp = tempfile.NamedTemporaryFile("w", delete=False)
        f_temp.write("")
        f_temp.close()

        with codecs.open(f_temp.name, 'w', encoding='utf-8') as fh:
            fh.write(text)
        

        command = self.udpipe+' '+options+' '+self.path_model+' '+f_temp.name
        p = subprocess.Popen([command],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True);           
        output, err = p.communicate()

        if err is not None:
            warnings.warn("Something unexpected occurred when running: "+command)
        
        os.unlink(f_temp.name)
    
        return output.decode("utf-8")



"""
Runs the JAMR script ALIGN.sh
@param path_input: Path to a file formatted in AMR
@param path_output: Path to the output file containing the alignments
@param path_jamr: Path where it is located the JAMR system
"""
def run_JAMR_aligner(path_input, path_output, path_jamr):
     
    print "Running JAMR ALIGNER: ",
    ftemp = tempfile.NamedTemporaryFile(delete=False)
    ftemp.write("#!/bin/bash\n\n") 
    ftemp.write(". "+path_jamr+"/scripts/config.sh;")
    print path_jamr+"/scripts/ALIGN.sh < "+path_input+ " > "+path_output,
    ftemp.write(path_jamr+"/scripts/ALIGN.sh < "+path_input+
                   " > "+path_output)
    ftemp.close()
    cmd = [ftemp.name]
    os.chmod(ftemp.name, 0777)
    subprocess.call(cmd, shell=True)
    os.remove(ftemp.name)
    print "[OK]"


"""
Tags, parses and applies NER to a list of sentences
@param udpipe_tagger: An instance of the UDpipe wrapper
@param 
@param sentences: The list of sentences
"""
def udpipe_tag(udpipe_tagger, stanford_ner, sentences):

    aux = stanford_ner.tag(" .NEWSENTENCE. ".join(sentences).split())
    entity_sentences = []
    entity_sentence = []
    
    for word, entity in aux:
        if word == ".NEWSENTENCE.":
            entity_sentences.append(entity_sentence)
            entity_sentence = []
        else:
            entity_sentence.append((word,entity))
    
    if entity_sentence != []:
        entity_sentences.append(entity_sentence)        
    
    dependency_trees = []
    sentences =  "\n".join(sentences)
    output = udpipe_tagger.run(sentences)
    tagged_conll_sentences = "\n".join(output.split("\n")[2:]).split("\n\n")

    for esentence,sentence in zip(entity_sentences,tagged_conll_sentences):
        tags = []
        dt = []  
        for j, l in enumerate(sentence.split("\n")[1:]):
            ls = [element for element in l.split("\t")]
            if ls != ['']:
                dt.append(AMRWord(int(ls[0]),ls[1],ls[2],ls[3],ls[4],ls[5],int(ls[6]),
                           ls[7],esentence[j][1]))
     
        if dt != []:
            dependency_trees.append(dt)
            
    return  dependency_trees

# THIS IS DONE NOW IN mlp.py
# def is_seqdate(seqdate):
#     try:
#         int(seqdate)
#         is_seqdate = len(seqdate) == 6 or len(seqdate) == 8
#         return is_seqdate
#     except ValueError:
#         return False
# 
# def preprocess_seqdate(seqdate):
#     
#     if len(seqdate) == 6:
#         year = seqdate[0:2]
#         month = seqdate[2:4]
#         day = seqdate[4:6]
#          
#         if year[0] == "0":
#             return "20"+year+"-"+month+"-"+day
#         elif year[0] == "1":
#             return "20"+year+"-"+month+"-"+day
#         else:
#             return "19"+year+"-"+month+"-"+day
#          
#     elif len(seqdate) == 8:
#         year = seqdate[0:4]
#         month = seqdate[4:6]
#         day = seqdate[6:8]
#         return year+"-"+month+"-"+day
    
"""
It creates an *.aligned file for path_amrs, containing the alignments
by JAMR.
It also create a *.dependencies file containing a enriched dependency tree (it also contains
info about NER) for path_amrs.
"""
def preprocess(udpipe_bin,udpipe_en_model, stanford_ner, jarm_bin,
                path_amrs, decoding=False):

    udpipe_tagger = UDPipe(udpipe_en_model,udpipe_bin)
    sentences = []

    ########################################
    # Creating the alignment file
    #######################################
    path_aligned_amr = path_amrs+".aligned"
    path_dependencies = path_amrs+".dependencies"
    path_samples_raw = path_amrs+".input"
    run_JAMR_aligner(path_amrs, path_aligned_amr, jamr_bin)
    
    with codecs.open(path_aligned_amr, encoding="utf-8") as fh:
        graphs = fh.read().strip('\n').split('\n\n')

    ########################################
    # Get the AMR graphs an preprocess them
    ########################################    
    for g in graphs:
        lines = g.split("\n")    
        if lines[0].startswith("# AMR release"): continue    
        for l in lines:
            if l.startswith(ID_JAMR_TOK):
                
                tok = l.replace(ID_JAMR_TOK,"").strip()#.split()
#                tok_preproc = []
#                for tok in tok_line:
                    
#                     if is_seqdate(tok) and decoding:
#                         ptok = preprocess_seqdate(tok)
#                         print ptok
#                     else:
#                    ptok = tok    
#                    tok_preproc.append(ptok)
                sentences.append(tok)
                    
    dependency_trees = udpipe_tag(udpipe_tagger, stanford_ner, sentences)

    with codecs.open(path_dependencies,"w") as f_trees:
        pickle.dump(dependency_trees,f_trees)

#     else:
#         
#         with codecs.open(path_amrs) as f:
#             sentences = [l.split() for l in f.readlines()] 
#         dependency_trees = udpipe_tag(udpipe_tagger, stanford_ner, sentences)  
#         with codecs.open(path_dependencies,"w") as f_trees:
#             pickle.dump(dependency_trees,f_trees)
        
        

    print "Saving *.raw",
    with codecs.open(path_samples_raw,"w") as f_raw:
        raw_samples = []
        for tree in dependency_trees:
            raw_samples.append([amr_word for amr_word in tree])
        pickle.dump(raw_samples, f_raw)
    print "[OK]"
        


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Runs JAMR, UDpipe and Stanford NER models on a AMR file')
    argparser.add_argument("-a", "--amrs", dest="amrs", help="Preprocess AMRs", type=str, default=None)
    argparser.add_argument("--decoding", "--decoding", default=False, action="store_true")
#     argparser.add_argument("-r","--raw", dest="raw", help="Preprocesses raw samples. It assumes already tokenized texts. One sentence per line.", 
#                            type=str)
    
    try:
        args = argparser.parse_args()
    except:
        argparser.error("Invalid arguments")
        sys.exit(0)
                
    config = ConfigParser.ConfigParser()
    config.readfp(open("./configuration.conf"))
    udpipe_bin = config.get("Resources","path_udpipe_bin") 
    udpipe_en_model = config.get("Resources","path_udpipe_trained_model")
    st = StanfordNERTagger(config.get("Resources","path_stanford_ner_model"), 
                           config.get("Resources","path_stanford_ner"))
    jamr_bin = config.get("Resources","path_jamr")
    
    if args.amrs is not None:
        preprocess(udpipe_bin,udpipe_en_model, st, jamr_bin,args.amrs, args.decoding)

    else:
        raise NotImplementedError
        
