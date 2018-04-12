from collections import OrderedDict
from utils import *
import codecs
import pickle
import ConfigParser
import argparse
import pickle
import itertools
import operator
from numpy.f2py.rules import aux_rules


"""
Creates a node_id:[alignments] dictionary
@param A list of lines corresponding to the output of the JAMR script ALIGN.sh
@return alignments
"""    
def get_alignments(lines_from_jamr_ALIGN):

    def is_aligned_node(node_line):
        return len(node_line.strip(ID_JAMR_NODE).strip().split('\t')) == 3
    
    alignments = {}
    for l in lines_from_jamr_ALIGN:   
        if l.startswith(ID_JAMR_NODE):
            nl = l.strip(ID_JAMR_NODE).strip().split('\t')
            uid = nl[0]    
            if is_aligned_node(l):
                ialignment = map(int,nl[2].strip('\'').split("-"))
                alignments[uid] = ialignment
    return alignments


#TODO: This solution is not elegant
"""
It determines the subsequence of words from amr_words[0] that make a named-entity.
It is used to manage NERs during parsing.
@param amr_words: A list of AMRWord instances
"""
def get_components(amr_words):
    
    first = amr_words[0]
    entity_entries = [amr_words[0]]

    for e in amr_words[1:]:
        if e.entity == amr_words[0].entity and e.entity != "O":
            entity_entries.append(e)
        else:
            break
     
    return entity_entries


def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


"""
If a node identified by uid was not aligned by JAMR, we try
to align it with couple of hooks
@param uid: Node id
@param d_aligned_at: A Dictionary node_id:[init_alignment, end_alignment]
"""
def heuristic_alignment(uid, d_aligned_at):
            
    #We try to do a match with a child term
    children = sorted([key for key in d_aligned_at
              if key.startswith(uid)])
    if len(children) != 0: #We assign the index of the first aligned child
        return d_aligned_at[children[0]]
    
    uid_split = uid.split(".")
    
    #We try to do a match with a head term
    for i in range(len(uid_split)-1,0,-1):       
        prefix = ".".join(uid_split[0:i])
        heads = sorted([key for key in d_aligned_at if key.startswith(prefix)])
        if len(heads) != 0:
            return d_aligned_at[heads[0]]
    
    return None



"""
Returns if the word is a named-entity
NOTE: If so, it will receive an special treatment during the parsing phase.
The AMR parser will try to apply a set of hooks
"""
def word_is_entity(amr_word, nationalities, d_node):
    is_target_entity = amr_word.entity in ["DATE","PERSON","LOCATION","ORGANIZATION", "PERCENT","MONEY", "TIME"]
    is_nationality = amr_word.form.lower() in nationalities
    if amr_word.entity not in ["DATE","PERSON","LOCATION","ORGANIZATION","O","_", "PERCENT","MONEY", "TIME"]:
        print amr_word.form, amr_word.entity
    return is_target_entity or is_nationality


"""
Returns if a word will generate multiple concepts, that is, 
if during the training phrase the oracle will determine BREAKDOWN transitions for it.
"""
def is_breakdown_node(amr_word, d_node):
    
    is_multiple_node = True if amr_word.index in d_node and len(d_node[amr_word.index]) > 1 else False 
    if is_multiple_node:
        there_is_unaligned = sum([1 for n in d_node[amr_word.index]
                              if n.unaligned]) > 0                  
        is_multiple_node = is_multiple_node and amr_word.entity == "O" and not there_is_unaligned   
    return is_multiple_node



"""
Gets phrases the generate multi-concept subgraphs
"""
def get_phrase_subgraphs(amrwords, d_arcs, aligned_amr_lines):

    graph_form_nodes = {} 
    subgraphs = {}
    
    #Reading the nodes (without Fake alignment)
    for l in aligned_amr_lines:
            if l.startswith(ID_JAMR_NODE):  
                  
                nl = l.replace(ID_JAMR_NODE,"").strip().split('\t')
                uid, c = nl[0], nl[1]
                if len(nl) ==3:
                    ialignment = tuple(map(int,nl[2].strip('\'').split("-")))
                    unaligned=False                       
                    if ialignment[1]-ialignment[0]> 1:
                        if ialignment not in graph_form_nodes:
                            graph_form_nodes[ialignment] = []
                        graph_form_nodes[ialignment].append(uid)                              

    #We build the subraphs
    for alignment in graph_form_nodes:     
                
        ####################################################
        #We create a sequence the sequence of AMRListEntries
        ####################################################
        root_id = ID_ROOT_ID
        root_word = AMRWord(-1,ID_ROOT_SYMBOL, None,None,None,None,None,None,'ROOT_ENTITY')
        root_node = AMRNode(root_word.form,-1,-1,relations=set([]),children=set([]),id=root_id,
                            pred_relations=set([]), pred_children=set([]),
                            created_by = "CONFIRM",
                            last_rel_as_head = None,
                            unaligned=False,
                            originated_from=root_word)
        root_entry = AMRListEntry(root_word,node=root_node,
                                  edges=set([]))  
        sequence = [root_entry]
        
        for j,index in enumerate(range(alignment[0], alignment[1]),1):  
            amr_word = amrwords[index]
            aux = AMRWord(j, amr_word.form, amr_word.lemma, amr_word.cpos,
                          amr_word.pos, amr_word.feats, amr_word.head, amr_word.deprel,
                          amr_word.entity)
            
            sequence.append(AMRListEntry(aux,None,set([])))
            

        ialignment = (len(sequence)-2, len(sequence)-1)      
        subgraph_form = " ".join([aux_word.form for aux_word in amrwords[alignment[0]:alignment[1]]])
        dict_map_node_ids = {}
        dict_map_node_ids[ID_ROOT_ID] =  ID_ROOT_ID
        head_node_id = None
        nid = 0
  
        for nodeid in graph_form_nodes[alignment]:
            dict_map_node_ids[nodeid] = str(nid)
            nid+=1
            
            if head_node_id is None or nodeid < head_node_id:
                head_node_id = nodeid
                                
        ###########################################################
        #We create the edges between the elements of the template
        ###########################################################
        nodes = {}
        nodes[ID_ROOT_ID] = root_node
        all_arcs = {}
        root_found = False
        
        for h,d in d_arcs:
  
            if h.id not in dict_map_node_ids or d.id not in dict_map_node_ids:
                continue

            rel = d_arcs[(h,d)]
            mapped_h_id = dict_map_node_ids[h.id]
            mapped_d_id = dict_map_node_ids[d.id]
            new_h =  AMRNode(h.concept,ialignment[0],ialignment[1],relations=set([]), children=set([]), id=mapped_h_id,
                                                   pred_relations=set([]), pred_children=set([]), created_by = None,
                                                   last_rel_as_head = None, unaligned=False,
                                                   originated_from= None)
            
            new_d = AMRNode(d.concept,ialignment[0],ialignment[1],relations=set([]), children=set([]), id=mapped_d_id,
                                                   pred_relations=set([]), pred_children=set([]), created_by = None,
                                                   last_rel_as_head = None, unaligned=False,
                                                   originated_from= None)                


           # print mapped_h_id, mapped_d_id, head_node.id, root_found
            if (mapped_h_id ==  dict_map_node_ids[head_node_id]  and not root_found):
                
                all_arcs[(root_node, new_h)] = "*root*"
                root_found = True
            

            if mapped_h_id not in nodes:
               nodes[mapped_h_id] = new_h
                                       
            if mapped_d_id not in nodes:
               nodes[mapped_d_id] = new_d
        
            all_arcs[(new_h,new_d)] =  COMPOSITE_RELATION.join(d_arcs[(h,d)])

        #################################
        #We create the AMRGraph instance
        #################################
        arcs = set([])
        for h,d in all_arcs:
        
            h.add_relation((all_arcs[(h,d)], h.id))    
            d.children.add((all_arcs[(h,d)], d.id))            
            arcs.add(AMREdge(h,all_arcs[(h,d)],d))    
                            
        subgraphs.update({(subgraph_form,'O'):AMRGraph(sequence, nodes, arcs, None, {},
                                     nodes_edges=all_arcs)})                 
                    
    return subgraphs
            

        
"""
Gets multi-concept subgraphs for non named-entity words
"""
def get_non_entity_subgraphs(amr_words,d_arcs,d_nodes):
    
    subgraphs = {}
    for amr_word in amr_words:
        subgraph_form = amr_word.form  
        is_content_word =  amr_word.pos.startswith("N") or amr_word.pos.startswith("J") #VERBS and ADVERBS generated lot of false-positives
        if amr_word.entity == "O" and is_content_word and amr_word.index in d_nodes and len(d_nodes[amr_word.index])>=2:   
            ####################################################
            #We create a sequence the sequence of AMRListEntries
            ####################################################
            root_id = ID_ROOT_ID
            root_word = AMRWord(-1,ID_ROOT_SYMBOL, None,None,None,None,None,None,'ROOT_ENTITY')
            root_node = AMRNode(root_word.form,-1,-1,relations=set([]),children=set([]),id=root_id,
                                pred_relations=set([]), pred_children=set([]),
                                created_by = "CONFIRM",
                                last_rel_as_head = None,
                                unaligned=False,
                                originated_from=root_word)
            root_entry = AMRListEntry(root_word,node=root_node,
                                      edges=set([]))  
            sequence = [root_entry] 
            aux = AMRWord(1, amr_word.form, amr_word.lemma, amr_word.cpos,
                              amr_word.pos, amr_word.feats, amr_word.head, amr_word.deprel,
                              amr_word.entity)
            sequence.append(AMRListEntry(aux,None,set([])))            
            ialignment = (len(sequence)-2, len(sequence)-1)  
            dict_map_node_ids = {}
            dict_map_node_ids[ID_ROOT_ID] =  ID_ROOT_ID
            head_node = None
            nid = 0
            subgraph_nodes = []
            for node in d_nodes[amr_word.index]:
                subgraph_nodes.append(node)  
                dict_map_node_ids[node.id] = str(nid)
                nid+=1
                
                if head_node is None or node.id < head_node.id:
                    head_node = node
                    
            ###########################################################
            #We create the edges between the elements of the template
            ###########################################################
            nodes = {}
            nodes[ID_ROOT_ID] = root_node
            all_arcs = {}
            root_found = False
            
            for h,d in d_arcs:          
                if h.id not in dict_map_node_ids or d.id not in dict_map_node_ids:
                    continue
                
                rel = d_arcs[(h,d)]
                mapped_h_id = dict_map_node_ids[h.id]
                mapped_d_id = dict_map_node_ids[d.id]
                new_h =  AMRNode(h.concept,ialignment[0],ialignment[1],relations=set([]), children=set([]), id=mapped_h_id,
                                                       pred_relations=set([]), pred_children=set([]), created_by = None,
                                                       last_rel_as_head = None, unaligned=False,
                                                       originated_from= None)
                
                new_d = AMRNode(d.concept,ialignment[0],ialignment[1],relations=set([]), children=set([]), id=mapped_d_id,
                                                       pred_relations=set([]), pred_children=set([]), created_by = None,
                                                       last_rel_as_head = None, unaligned=False,
                                                       originated_from= None)                

                if (mapped_h_id ==  dict_map_node_ids[head_node.id]  and not root_found):
                    
                    all_arcs[(root_node, new_h)] = "*root*"
                    root_found = True
            
                if mapped_h_id not in nodes:
                   nodes[mapped_h_id] = new_h
                                           
                if mapped_d_id not in nodes:
                   nodes[mapped_d_id] = new_d
            
                all_arcs[(new_h,new_d)] =  COMPOSITE_RELATION.join(d_arcs[(h,d)])
    
            #################################
            #We create the AMRGraph instance
            #################################
            arcs = set([])
            for h,d in all_arcs:
            
                h.add_relation((all_arcs[(h,d)], h.id))    
                d.children.add((all_arcs[(h,d)], d.id))            
                arcs.add(AMREdge(h,all_arcs[(h,d)],d))    
                
            subgraphs.update({(subgraph_form,'O'):AMRGraph(sequence, nodes, arcs, None, {},
                                         nodes_edges=all_arcs)}) 
            
    return subgraphs

"""
Generates a subgraph template for entity and 'breakdown' words
@param d_arcs: The dictionary of arcs for the whole AMR graph
@param d_nodes: The dictionary of nodes for the whole AMR graph
"""
def generate_entity_subgraph_template(sequence, d_arcs, d_nodes):
        
    subgraphs = {}
    for amr_word in sequence:
       if amr_word.components is not None and amr_word.entity in ["LOCATION","ORGANIZATION","PERSON", 
                                                                  "DATE","MONEY","TIME","PERCENT"]:#,"O"]
           
            entity_form =  " ".join([component.form for component in amr_word.components])
            
            ####################################################
            #We create a sequence the sequence of AMRListEntries
            ####################################################
            root_id = ID_ROOT_ID
            root_word = AMRWord(-1,ID_ROOT_SYMBOL, None,None,None,None,None,None,'ROOT_ENTITY')
            root_node = AMRNode(root_word.form,-1,-1,relations=set([]),children=set([]),id=root_id,
                                pred_relations=set([]), pred_children=set([]),
                                created_by = "CONFIRM",
                                last_rel_as_head = None,
                                unaligned=False,
                                originated_from=root_word)
            root_entry = AMRListEntry(root_word,node=root_node,
                                      edges=set([]))  
            sequence = [root_entry]
            
            for index,component in enumerate(amr_word.components,1):                    
                aux = AMRWord(index, component.form, component.lemma, component.cpos,
                              component.pos, component.feats, component.head, component.deprel,
                              component.entity)
                sequence.append(AMRListEntry(aux,None,set([])))
    
            ialignment = (len(sequence)-2, len(sequence)-1)

            component_indexes = [] 
            dict_map_node_ids = {}
            dict_map_node_ids[ID_ROOT_ID] =  ID_ROOT_ID
            nid = 0
            
            ###########################################################
            #We determine the head node of the tree and 
            ###########################################################
            head_node = None
            for component in amr_word.components:
                if component.index in d_nodes:
                    for node in d_nodes[component.index]:
                        component_indexes.append(node)  
                        dict_map_node_ids[node.id] = str(nid)
                        nid+=1
                        
                        if head_node is None or node.id < head_node.id:
                            head_node = node
                        
            ###########################################################
            #We create the edges between the elements of the template
            ###########################################################
            nodes = {}
            nodes[ID_ROOT_ID] = root_node
            all_arcs = {}
            root_found = False
            
            for h,d in d_arcs:
                if h.id not in dict_map_node_ids or d.id not in dict_map_node_ids:
                    continue
                
                rel = d_arcs[(h,d)]
                mapped_h_id = dict_map_node_ids[h.id]
                mapped_d_id = dict_map_node_ids[d.id]
                new_h =  AMRNode(h.concept,ialignment[0],ialignment[1],relations=set([]), children=set([]), id=mapped_h_id,
                                                       pred_relations=set([]), pred_children=set([]), created_by = None,
                                                       last_rel_as_head = None, unaligned=False,
                                                       originated_from= None)
                
                new_d = AMRNode(d.concept,ialignment[0],ialignment[1],relations=set([]), children=set([]), id=mapped_d_id,
                                                       pred_relations=set([]), pred_children=set([]), created_by = None,
                                                       last_rel_as_head = None, unaligned=False,
                                                       originated_from= None)                

                if mapped_h_id ==  dict_map_node_ids[head_node.id] and not root_found:
                    all_arcs[(root_node, new_h)] = "*root*"
                    root_found = True
                    

                if mapped_h_id not in nodes:
                   nodes[mapped_h_id] = new_h
                                           
                if mapped_d_id not in nodes:
                   nodes[mapped_d_id] = new_d
            
                all_arcs[(new_h,new_d)] =  COMPOSITE_RELATION.join(d_arcs[(h,d)])
            
            #################################
            #We create the AMRGraph instance
            #################################
            arcs = set([])
            for h,d in all_arcs:
   
                h.add_relation((all_arcs[(h,d)], h.id))    
                d.children.add((all_arcs[(h,d)], d.id))            
                arcs.add(AMREdge(h,all_arcs[(h,d)],d))    
            
            if len(nodes) > 1:    
                subgraphs.update({(entity_form, amr_word.entity):AMRGraph(sequence, nodes, arcs, None, {},
                                        nodes_edges=all_arcs)}) 
    return subgraphs
    

def _update_subgraphs_occurrences(d_subgraphs,subgraph_form_occurrences, subgraph_occurrences):
    
#     subgraph_form_occurrences = {}
#     subgraph_occurrences = {}

    for key in d_subgraphs:
        if key not in subgraph_form_occurrences:
            subgraph_form_occurrences[key] = 0
        subgraph_form_occurrences[key]+=1
        
        
    for key in d_subgraphs:
        
        if key not in  subgraph_occurrences:
            subgraph_occurrences[key] = {}
    
        subgraph_str_raw = "_".join(sorted([d_subgraphs[key].nodes[naux].concept for naux in d_subgraphs[key].nodes]))
        if subgraph_str_raw not in subgraph_occurrences[key]:
            subgraph_occurrences[key][subgraph_str_raw] = (1, d_subgraphs[key])
        else:
            occs = subgraph_occurrences[key][subgraph_str_raw][0]+1    
            subgraph_occurrences[key][subgraph_str_raw] = (occs, d_subgraphs[key])    
            
    return subgraph_occurrences



def _is_potential_single_word_missing_entity(word, vocab):
    return word[0] == word[0].upper() and word.lower() not in vocab

def _usually_maps_to_one_graph(graphs,threshold=5):
    
    return (graphs[0][1][0] > threshold and len(graphs) ==1 )
    

def _is_negating_term(word, graph):
    
    return (word.endswith("less") or 
            word.startswith("un") or 
            word.startswith("in") or 
            word.startswith("il") or 
            word.startswith("ir") or 
            word.startswith("dis") or 
            word.startswith("non")) and  '-' in [n.concept for n in graph.nodes.values()] 

"""
Reads the aligned file obtained by the JARM script ALIGN.sh

@return: 
1. A set of instance of AMRgraphs.
2. A lookup table mapping uncommon words to the most commont mapped concept
3. A set of template subgraphs for entity and words to be 'breakdown' in the
parsing file
"""

def read_aligned_AMR(path,
                     path_nationalities, path_nationalities2):
    
    path_aligned = args.amrs+".aligned"
    path_dependencies =  args.amrs+".dependencies" 
    path_lookuptable_concepts = args.amrs+".word_concepts"
    
    amr_graphs = []
    template_graphs = {}
    #non_entity_template_graphs = {}
    non_entity_template_form_occ = {}
    non_template_graphs_occ = {}
    multiple_non_entity_template_form_occ = {}
    multiple_non_template_graphs_occ = {}
    
    template_graph_form_occ = {} #NEW
    template_graphs_occ = {}
    n_tok_lines = 0
    
    #We consider nationalities from the lookup table as entities too
    with codecs.open(path_nationalities) as f:
        nationalities = {l.split("=>")[1].strip().lower().replace("'",""):l.split("=>")[0].strip().replace("'","").replace(",","")
                              for l in f.readlines()}
    with codecs.open(path_nationalities2) as f:
        nationalities2 = {l.split("\t")[1].strip().lower():l.split("\t")[0] for l in f.readlines()}              
    nationalities.update(nationalities2)
    
    #########################################
    # Loading the aligned AMRS
    #########################################
    with codecs.open(path_aligned, encoding="utf-8") as fh:
        graphs = fh.read().strip('\n').split('\n\n')
    
    #########################################
    #Loading the dependency trees
    #########################################
    with codecs.open(path_dependencies) as fh:
        dependency_trees = pickle.load(fh)
    
    #########################################
    #Processing graphs and saving templates and lookup
    #tables forr uncommon nodes
    #########################################
    word_counter = Counter()
    word_list = []
    word_concept_dict = {}
    total_entity = 0

    print "Processing AMR graphs... This may take some time",
    for gid,g in enumerate(graphs):
        
        dt = dependency_trees[n_tok_lines]
        nodes = OrderedDict({})
        d_arcs = {}
        substring_to_concepts = {}
        arcs = set([])
        is_multi_sentence = False
        
        lines = g.split("\n")  
        if lines[0].startswith("# AMR release"): continue
        d_aligned_at = get_alignments(lines)
        ialignment = None
        raw_sequence = dt
        n_tok_lines+=1
        d_nodes = {}
        
        for l in lines:

            #Reading the graph id
            if l.startswith(ID_JAMR_GRAPH_ID):
                graph_id = l
                                   
            #Reading the nodes
            if l.startswith(ID_JAMR_NODE):  
                
                nl = l.replace(ID_JAMR_NODE,"").strip().split('\t')
                uid, c = nl[0], nl[1]
                if len(nl) ==3:
                    ialignment = map(int,nl[2].strip('\'').split("-"))
                    unaligned=False                  
                             
                else:
                    aux = heuristic_alignment(uid, d_aligned_at)
                    if ialignment is None:
                        ialignment = (max(0,len(raw_sequence)-2), max(1,len(raw_sequence)-1))
                    ialignment = ialignment if aux is None else aux   
                    unaligned = True
                    
                if ialignment[1] - ialignment[0] > 1:
                    ialignment = (ialignment[1]-1, ialignment[1])

                else:

                    form = raw_sequence[ialignment[0]].lemma
                    if form not in word_concept_dict:
                        word_concept_dict[form] = {}
                    
                    if c not in word_concept_dict[form]:
                        word_concept_dict[form][c] = 0
                    word_concept_dict[form][c] += 1
                word_list.append(form)
                
                node = AMRNode(c,ialignment[0],ialignment[1],relations=set([]), children=set([]), id=uid,
                               pred_relations=set([]), pred_children=set([]), created_by = None,
                               last_rel_as_head = None, unaligned=unaligned,
                               originated_from= None)
                

                
                if uid not in nodes and not (c == ID_MULTISENTENCE and uid == "0"):
                    
                    nodes[uid] = node
                    if (ialignment[0],ialignment[1]) not in substring_to_concepts:
                        substring_to_concepts[(ialignment[0],ialignment[1])] = [c]
                    else:
                        substring_to_concepts[(ialignment[0],ialignment[1])].append(c)

                    if ialignment[1] not in d_nodes:
                        d_nodes[ialignment[1]] = []                
                                        
                    d_nodes[ialignment[1]].append(node)
                                

            #Reading the root of the sentence
            if l.startswith(ID_JAMR_ROOT):
                nl = l.replace(ID_JAMR_ROOT,"").strip().split('\t')
                c = nl[1]
                
                if c != ID_MULTISENTENCE:
                    root_word = AMRWord(-1,ID_ROOT_SYMBOL, None,None,None,None,None,None,'ROOT_ENTITY')
                    root_id = ID_ROOT_ID
                    root_node = AMRNode(root_word.form,-1,-1,relations=set([]),children=set([]),id=root_id,
                                        pred_relations=set([]), pred_children=set([]),
                                        created_by = "CONFIRM",
                                        last_rel_as_head = None,
                                        unaligned=False,
                                        originated_from=root_word)
                    
                    root_entry = AMRListEntry(root_word,node=root_node,
                                              edges=set([]))  

                    arcs.add(AMREdge(root_node, "*root*", nodes["0"]))
                    nodes["0"].add_relation(("*root*", root_node.id))
                    d_arcs[(root_node, nodes["0"])] = ["*root*"]

                else:
                    root_word = AMRWord(-1,ID_MULTISENTENCE, None,None,None,None,None,None,'ROOT_ENTITY')
                    root_id = "0"
                    root_node = AMRNode(root_word.form,-1,-1,id=root_id, relations=set([]), children=set([]),
                                        pred_relations=set([]), pred_children=set([]),
                                        created_by = "CONFIRM",
                                        last_rel_as_head = None,
                                        unaligned=False,
                                        originated_from=root_word)
                    
                    root_entry = AMRListEntry(root_word,node=root_node,
                                              edges=set([]))                    
                nodes[root_id] = root_node
                d_nodes[-1] = [root_node]
                
            #
            #    GETTING THE EDGES
            #
            if l.startswith(ID_JAMR_EDGE):
                nl = l.replace(ID_JAMR_EDGE,"").strip().split('\t')
                source,rel,dest,source_uniq_id, dest_uniq_id = nl[0], nl[1], nl[2], nl[3], nl[4]
                
                head_node = nodes[source_uniq_id]
                dependent_node = nodes[dest_uniq_id]
                
                #Computing composite relations                
                if (head_node,dependent_node) in d_arcs:
                    d_arcs[(head_node,dependent_node)].append(rel)
                else:
                    d_arcs[(head_node,dependent_node)] = [rel]
                              
        #Removing nodes that are part of entities and will be preprocessed
        #during parsing and testing
        if raw_sequence is not None:
        
            indexes_entities = []
            final_nodes = {}
            final_arcs = {}
                
            sequence = [root_entry] 
            components = []
            final_arcs = {}
            final_nodes = {root_id: root_node}

            for nw,amr_word in enumerate(raw_sequence,1):
                sequence.append(AMRListEntry(amr_word,None,set([])))
                 
                if amr_word in components: 
                    continue
                
                #Used for training
                if is_breakdown_node(amr_word, d_nodes):
                    amr_word.add_components([amr_word])
                
                if not word_is_entity(amr_word, nationalities, d_nodes):
                     if nw in d_nodes:    
                         for n in d_nodes[nw]:
                             final_nodes[n.id] = n
                else:        
                    if word_is_entity(amr_word, nationalities, d_nodes):
                         index = raw_sequence.index(amr_word)       
                         components = get_components(raw_sequence[index:])   
                         amr_word.add_components(components)
                         total_entity+=1
                         component_indexes = []
                         for com in components:
                             if com.index in d_nodes:
                                 for node in d_nodes[com.index]:
                                     component_indexes.append(node)
                         
                         component_indexes = sorted(component_indexes, key = lambda x: x.id)
     
                         for word in components:
                             indexes_entities.append(word.index)
     
                         if component_indexes != []:
                             n = component_indexes[0] #nodes_at[0]
                             final_nodes[n.id] = n
                             

            for h,d in d_arcs:
                
                if h.id not in final_nodes or d.id not in final_nodes:
                    pass
                else:
                    final_arcs[(h,d)] = COMPOSITE_RELATION.join(d_arcs[(h,d)])

            
            for h,d in final_arcs:
                h.add_relation((final_arcs[(h,d)], h.id))    
                d.children.add((final_arcs[(h,d)], d.id))            
                arcs.add(AMREdge(h,final_arcs[(h,d)],d))        
            
            phrase_subgraphs = get_phrase_subgraphs(raw_sequence,d_arcs, lines)
            _update_subgraphs_occurrences(phrase_subgraphs,multiple_non_entity_template_form_occ, 
                                          multiple_non_template_graphs_occ)
            
            non_entity_token_subgraphs = get_non_entity_subgraphs(raw_sequence, d_arcs, d_nodes) 
            _update_subgraphs_occurrences(non_entity_token_subgraphs,
                                          non_entity_template_form_occ,
                                          non_template_graphs_occ)
            
            entity_token_subgraphs = generate_entity_subgraph_template(raw_sequence, d_arcs, d_nodes)
            _update_subgraphs_occurrences(entity_token_subgraphs,
                                          template_graph_form_occ,
                                          template_graphs_occ)            
            
            amr_graphs.append(AMRGraph(sequence, final_nodes, arcs, graph_id, substring_to_concepts,
                                       nodes_edges=final_arcs))   
            
            
    word_counter.update(word_list)
    

    #Gets most the most common node generate for uncommon words
    with codecs.open(path_lookuptable_concepts,"w",encoding="utf-8") as f:
    #    print "Building lookup table word:concept for uncommon words...",
        for form in word_concept_dict:
            most_common_concept = sorted(word_concept_dict[form].items(), key = lambda x: x[1], reverse=True)[0][0]
            f.write("\t".join((singularize(form.lower()), most_common_concept))+"\n")                 
    #    print "[OK]"
    
    #Gets most frequent subgraph for entity template graphs
    template_graphs = {}
   # print "Building template graphs for entity and 'breakdown' words...",
    for form_graph in template_graphs_occ:  
        graph_options = sorted(template_graphs_occ[form_graph].items(), 
                               key = lambda x: x[1][0], reverse=True)
        template_graphs[form_graph] = graph_options[0][1][1]
    
    #Gets most frequent subgraph for non-entity template graphs that are likely to be
    #reliable based on simple heuristics
    #non_template_graphs = {}
   # print "Building template graphs for entity and 'breakdown' words...",
    for form_graph in non_template_graphs_occ:
        graph_options = sorted(non_template_graphs_occ[form_graph].items(), 
                               key = lambda x: x[1][0], reverse=True)
   #     non_template_graphs[form_graph] = graph_options[0][1][1]            
         
        if ( _is_potential_single_word_missing_entity(form_graph[0], word_counter)
             or _usually_maps_to_one_graph(graph_options)
             or  _is_negating_term(form_graph[0],  graph_options[0][1][1])):
            
            template_graphs[form_graph] = graph_options[0][1][1]
        
        
    #Gets the most frequent subgraphs for non-entity n-grams (n >=2).
    #They are saved in a different file.
    #We use a nested dict for a more efficient search 
    multiple_template_graphs = {}  
    for form_graph in multiple_non_template_graphs_occ:
        graph_options = sorted(multiple_non_template_graphs_occ[form_graph].items(), 
                               key = lambda x: x[1][0], reverse=True)        
        current_level_dict = multiple_template_graphs
        
        for j,form in enumerate(form_graph[0].split()):
            if form not in current_level_dict:
                current_level_dict[form] = ({}, None)
            if j+1 == len(form_graph[0].split()):
                current_level_dict[form] = (current_level_dict[form][0] , graph_options[0][1][1] )
            else:
                current_level_dict = current_level_dict[form][0]        
            

    return amr_graphs, template_graphs, multiple_template_graphs


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description='Runs UDpipe and the Stanford NER')
    argparser.add_argument("-a", "--amrs", dest="amrs", help="Preprocess AMRs", type=str)
    
    try:
        args = argparser.parse_args()
    except:
        argparser.error("Invalid arguments")
        sys.exit(0)

    path_graphs = args.amrs+".graphs"
    path_graph_templates = args.amrs+".word_subgraphs"
    path_multipleword_graph_templates = args.amrs+".phrase_subgraphs"
    config = ConfigParser.ConfigParser()
    config.readfp(open("./configuration.conf"))
    path_nationalities = config.get("Resources", "path_nationalities")
    path_nationalities2 = config.get("Resources", "path_nationalities2")

    graphs, templates, multiple_word_templates = read_aligned_AMR(args.amrs, path_nationalities, path_nationalities2)
    
  #  print "Saving file *.graphs and *.word_concepts and subgraphs",
    with codecs.open(path_graphs,"w") as f_graphs:
        pickle.dump(graphs,f_graphs)
          
    with codecs.open(path_graph_templates,"w") as f_templates:
        pickle.dump(templates,f_templates)
        
    with codecs.open(path_multipleword_graph_templates,"w") as f_templates:
        pickle.dump(multiple_word_templates,f_templates)
    
    with codecs.open(path_graphs) as f_graphs:
        read_graphs = pickle.load(f_graphs)
  #  print "[OK]"
    
    