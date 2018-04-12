from collections import Counter
from pattern.en import verbs, conjugate, INFINITIVE, singularize

import re
import copy
import penman
import string
import random
#import wikipedia


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

#Proof
ID_MULTISENTENCE_ABBR = "mm"
EMPTY_GRAPH_ABBR = "e"
EMPTY_GRAPH_CONCEPT = "empty-graph"
INSTANCE_TRIPLET = "instance"


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word):
    if word is None: return word
    return 'num' if numberRegex.match(word) else word.lower()



class AMRWord(object):
    """
    Class implementing the information that is stored for A WORD in AMR.
    
    It basically store the same information of a word in a CoNLL file commonly
    used for dependency parsing.
    
    @param index: Index of the word in the (original, unprocessed) sentence.
    @param form: A string
    @param norm: Normalized form
    @param lemma: Lemma of form
    @param cpos: Coarse PoS-tag
    @param pos: Fine PoS-tag
    @param Feats: Additional.
    
    Additionally, it stores:
    
    @param entity Type of entity for that word (it is part of a larger entity it will
    store which type is it too)
    @param components A list of next AMRwords in the sentence. It is used to transform
    entities into graphs in amr_concepts.RuleBasedConceptModel.
    
    """
    
    def __init__(self, index, form, lemma, cpos, pos, feats, head, deprel, entity):
    
        self.index = index
        self.form = form
        self.lemma = lemma
        self.norm = normalize(form)
        self.cpos = cpos
        self.pos = pos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.entity = entity
        
        #TODO: Not a good solution. Think of a better one
        #Keeps a list with the upcoming AMRWord's that form an entity together
        #with this one
        self.components = None
        
    def __str__(self):
        
        return (self.form+"("+str(self.index)+")"+ " E:"+self.entity).encode('utf-8')
            
    def add_components(self,components):
        self.components = components
        
    @property
    def has_predefined_components(self):
        return self.components is not None



class AMRNode(object):
    """
    Class that contains the information that is kept for a node an AMRGraph.
    @param concept: A string. It stores "the form" of the concept
    @param id: ID of the node
    """
    
    PROBANK_CONCEPT = 0
    CONSTANT_CONCEPT = 1
    ROOT_CONCEPT = 2
    OTHER_CONCEPT = 3

    def __init__(self,concept,start,end,relations=None, children=None,id=None, 
                 pred_relations=None, pred_children =None, created_by=None,
                 last_rel_as_head = None, unaligned=False, originated_from=None):
        
        self.id = id
        self.concept = concept
        
        
        self.start = start
        self.end = end
        
        self.relations = relations 
        self.children = children
        self.pred_relations = pred_relations
        self.pred_children = pred_children
        
        self.created_by = created_by
        self.originated_from = originated_from

        self.last_rel_as_head = None
        
        self.unaligned = unaligned
        
    
    @property
    def word(self):
        return self.originated_from
    
    @property
    def is_root(self):
        return self.concept == ID_ROOT_SYMBOL or self.concept == ID_MULTISENTENCE
    
    @property
    def indexed_at(self):
        return (self.start, self.end, self.id)
    
    @property
    def is_propbank(self):
        return len(self.concept.split("-"))== 2 and self.concept.split("-")[-1].isdigit()
        
    def add_relation(self,rel):
        self.relations.add(rel)

    def add_child(self,rel_child):
        self.children.add(rel_child)

    def add_pred_relation(self,pred_rel):
        self.pred_relations.add(pred_rel)
        
    def add_pred_children(self,pred_child):
        self.pred_children.add(pred_child)
        self.last_rel_as_head = pred_child[0]
    
    def remove_pred_relation(self, nid):
        element = None
        for r, id in self.pred_relations:
            if nid == id: element = (r,id)
            
        self.pred_relations.remove(element)
    
    def remove_pred_children(self, nid):
        element = None
        for r, id in self.pred_children:
            if nid == id: element = (r,id)
            
        self.pred_children.remove(element)
    
    
    @property
    def is_constant(self):
        """
        Determines if the node is a constant node, i.e. a number or a string.
        """
        def _is_float(value):
        
            try:
                float(value)
                return True
            except ValueError:
                return False
        
        return (self.concept.startswith("\"") and self.concept.endswith("\"")
                 or (_is_float(self.concept) and self.pred_children == set([]) ) ) or (self.concept =="-" and self.pred_children == set([]) )   


    @property    
    def concept_type(self):
    
        concept_type = self.OTHER_CONCEPT#3
    
        if self.is_propbank:
            concept_type = self.PROBANK_CONCEPT #0
        elif self.is_constant:
            concept_type = self.CONSTANT_CONCEPT #1
        elif self.is_root:
            concept_type = self.ROOT_CONCEPT #2
 
        return concept_type

    @property
    def has_head(self):
        return len(self.pred_relations) > 0



    """
    Computes the detph of the graph
    
    @param c: A CovingtonConfiguration
    @param visited_nodes: A list of IDS of the nodes that have been already analized. 
    Set to [] in the external/initial call to the function
    """
    def depth(self, c, visited_nodes):
        
        heads = [nid for rel,nid in self.pred_relations]
        head_nodes = [n for n in c.N if n.id in heads]
        
        visited = copy.deepcopy(visited_nodes)
        
        if len(heads) == 0 or self.id in visited:
            return 0
        else:
            visited.append(self.id)
            h_depth = []
            for h in head_nodes:
                h_depth.append(1+h.depth(c, visited))
            return max(h_depth)
    

    """
    Gets the leftmost head of the AMRnode
    """
    def lm_h(self,c,l=1):
        

        heads_id = [nid for rel,nid in self.pred_relations]
        head_nodes = [(n, n.start) for n in c.N 
                      if n.id in heads_id and n.start < self.start]
        
        if len(head_nodes) == 0:
            return None   
        else:
            lm_hn = sorted(head_nodes, key=lambda x: x[1], reverse=False)[0]
            if l == 1:
                return lm_hn[0]
            else:
                return lm_hn[0].lm_h(c,l-1)
    
    """
    Gets the leftmost child(ren) of the AMRnode
    """    
    def lm_child(self, c, l=1):
        
        children_id = [nid for rel,nid in self.pred_children]
        #child_nodes = [(n, n.start) for n in c.N if n.id in children_id]
        child_nodes =  [(n, n.start) for n in c.N 
                        if n.id in children_id and n.start < self.start]
        
        if len(child_nodes) == 0:
            return None
        else:

            lm_childn = sorted(child_nodes, key=lambda x: x[1], reverse=False)[0]
            if l == 1:
                return lm_childn[0]#_concept
            else:
                return lm_childn[0].lm_child(c,l-1)
    
    
    """
    Gets the rightmost head of the AMRnode
    """
    def rm_h(self,c,l=1):
        
        heads_id = [nid for rel,nid in self.pred_relations]
        head_nodes = [(n, n.start) for n in c.N 
                      if n.id in heads_id and n.start > self.start]
        if len(head_nodes) == 0:
            return None
        
        else:
            lm_hn = sorted(head_nodes, key=lambda x: x[1], reverse=True)[0]
            if l == 1:
                return lm_hn[0]
            else:
                return lm_hn[0].lm_h(c,l-1)
    

    """
    Gets the rightmost child(ren) of the AMRnode
    """    
    def rm_child(self, c, l=1):
        
        children_id = [nid for rel,nid in self.pred_children]
        child_nodes = [(n, n.start) for n in c.N 
                       if n.id in children_id and n.start > self.start]
        if len(child_nodes) == 0:
            return None
        else:

            lm_childn = sorted(child_nodes, key=lambda x: x[1], reverse=True)[0]

            if l == 1:
                return lm_childn[0]
            else:
                return lm_childn[0].lm_child(c,l-1)
    

    #TODO: This is not elegant and the list() could cause problems
    """
    Get the first relation of the node
    """
    def get_relation(self):
        if len(self.pred_relations) != 1:
            return None
        else:
            return list(self.pred_relations)[0][0]       
       

    def __str__(self):
        return ("("+",".join([self.concept, str(self.start), str(self.end), 
                              str(self.id), str(self.created_by)])+")").encode("utf-8")
        
        
        
class AMRListEntry(object):
    
    """
    Class containing the basic information of what is already know about each entry
    in a list transition-based parsing algorithm for AMR.
    
    @param word: An instance of an AMRWord
    @param node: An instance of an AMRNode
    """
    
    def __init__(self, word, node=None, edges=set([])):

        self.word = word
        self.node = node

    @property
    def is_node(self):
        return self.node is not None

    @property
    def indexed_at(self):
        return (self.word.index-1, self.word.index)
 
    @property
    def full_form(self):
        return  self.word.form




class AMREdge(object):
    """
    Class implementing an edge in a AMRGraph
    @param head: An instance of an AMRNode
    @param rel: A string indicating the semantic relationship
    @param dependent: An instance of an AMRNode
    """
    
    def __init__(self,head,rel,dependent):
        
        self.head = head
        self.rel = rel
        self.dependent = dependent

    def __str__(self):
        return "e("+",".join([str(self.head), str(self.rel), str(self.dependent)])+")".encode("utf-8")
    
    @property
    def unlabeled_as_indexes(self):
        return (self.head.indexed_at, self.dependent.indexed_at)
    
    @property
    def labeled_as_indexes(self):
        return (self.head.indexed_at, self.rel, self.dependent.indexed_at)





class AMRGraph(object):
    """
    Class implementing an AMR graph
    @param id: An ID for the graph
    """
    
    def __init__(self, sequence =[],nodes={}, A=set([]),id=None, substring_to_concepts={}, nodes_edges={},
                 original_sequence=[]):
        
        self.sequence = sequence
        self.nodes = nodes
        self.A = A
        self.id = id
        
        #self.substring_to_concepts = substring_to_concepts
        self.nodes_edges = nodes_edges
        
        self.original_sequence = original_sequence
        self.original_length = len(self.sequence)
        
        #Provisional
        self.nodes_indexed_at = {}
        for n in self.nodes.values():
            if not (n.start, n.end) in self.nodes_indexed_at:
                self.nodes_indexed_at[(n.start,n.end)] = [n]
            else:
                self.nodes_indexed_at[(n.start,n.end)].append(n)
        
        self.arcs_at = {}
        for h,d in self.nodes_edges:
            self.arcs_at[(h.id,d.id)] = self.nodes_edges[(h,d)]
            
            
    def node_id(self,indexed, unvalid_ids, aligned=True):
 
        ids = sorted([self.nodes[n].id for n in self.nodes
                      if ((self.nodes[n].start, self.nodes[n].end) ==  indexed) 
                      and self.nodes[n].id not in unvalid_ids], reverse=True) 
             
        return ids[0] if len(ids) > 0 else None
         


    #TODO: Why we need the nid? 
    """
    Gives an id in the form of abbreviation for a AMRnode and
    updates the dictionary of abbreviations
    @paran d_abbr: Dictionary of already existent abbreviations
    @param node: An AMRNode
    @param nid: ID of the current node
    """
    def abbreviation_for_concept(self, d_abbr, node, nid):
        
        concept = node.concept
        if concept == ID_ROOT_SYMBOL:
            return ID_MULTISENTENCE_ABBR
         
        #If the node is a constant, we do not abbreviate it
        if node.is_constant:
            return concept
        
        #If the node is a float, we assign as ID a chard that
        #still has not been assigned
        try:
            float(concept)
            chars = set([l.lower() for l in string.ascii_letters])
            available_chars = chars.difference(d_abbr.values())  
            abbr = random.choice(list(available_chars))
        except ValueError:
            abbr = concept[0]+"2"
            
        #Otherwise, we try to predict the abbreviation. If an abbreviation x
        #exists, we will create x2, x3, x4,... until xN is not in the graph
        n_abbr = 2
        if nid not in d_abbr:   
            next_abbr = abbr     
            while next_abbr in d_abbr.values():
                n_abbr+=1
                next_abbr = abbr[0]+str(n_abbr)
        
            d_abbr[nid] = next_abbr
            return next_abbr
        else:
          return d_abbr[nid]



    """
    Determines if the AMR graph starting at root_node has more
    than one children
    PreCD: root_node must be the dummy root of the AMRgraph
    """
    def is_multisentence_graph(self, root_node):
        return len(root_node.pred_children) > 1


    """
    Returns the top node of graph and the list of triplets that compose
    such graph in the form of a tuple (top,triplets)
    """
    def get_graph_root(self, triplets, is_multisentence):
        
        if is_multisentence:
            
            for t in triplets:
                if t.relation == INSTANCE_TRIPLET and (t.target==ID_MULTISENTENCE or t.target == "and"):
                    return t.source, triplets

        else:
            
            final_triplets = []     
            for i,triplet in enumerate(triplets):
                if triplet.source == ID_ROOT_SYMBOL or triplet.source == ID_MULTISENTENCE_ABBR:
                    if triplet.relation != INSTANCE_TRIPLET or len(triplets) == 1:
                        top = triplets[i].target
                
                if (triplet.source != ID_MULTISENTENCE_ABBR and triplet.target != ID_MULTISENTENCE_ABBR):

                    final_triplets.append(triplet)
                
            
            return top,final_triplets
            
    
    
    def path_to_root(self, node, visited_nodes,gold):
        
        visited_nodes.append(node.id)
        
        id_root_id = "-1"
        if gold:
            if "-1" in self.nodes:
                id_root_id = "-1"
            else:
                id_root_id = "0"
        
        if node.id == id_root_id:
            return True   
        
        else:
            reach_root = []
            for r,h in node.pred_relations:
                
                if h in visited_nodes:
                    reach_root.append(False)
                else:
                    visited_nodes.append(h)
                    reach_root.append(self.path_to_root(self.nodes[h],visited_nodes,gold))
            return any(reach_root)



    #TODO: Troublesome code to print gold and predicted AMR graphs
    """
    Print the instance of the AMRGraph into a valid raw AMR format
    """
    def print_graph(self, gold=False):

        #Fill the nodes attribute for the gold graph
        if gold: 

            for a in self.A:    
                h = a.head
                r = a.rel
                d = a.dependent
                self.nodes[h.id].add_pred_children((r,d.id))
                self.nodes[d.id].add_pred_relation((r,h.id))
                
        id_root_id = "-1"
        if gold:
            if "-1" in self.nodes:
                id_root_id = "-1"
            else:
                id_root_id = "0"

        #Post-processing steps: In the predicted graph some nodes might
        #have not been attached to any node. 
        #We define an heuristic and connect them tho the as a "snt" branch 
        #of the AMR graph
        nodes_without_root = True
        while nodes_without_root:
 
            nodes_to_connect = []
            for nid in self.nodes:
                if not self.path_to_root(self.nodes[nid], [], gold):
                     nodes_to_connect.append(self.nodes[nid])
                                 
            nodes_without_root = (nodes_to_connect != [])
            if nodes_without_root:
                
                aux = sorted(nodes_to_connect, key = lambda x: len(x.pred_children), reverse=True)[0]
                self.nodes[aux.id].add_pred_relation(("snt", id_root_id))
                self.nodes[id_root_id].add_pred_children(("snt", aux.id))
                nodes_to_connect = []
                     
                    
        is_multisentence = self.is_multisentence_graph(self.nodes[id_root_id])
        root_abr = ID_MULTISENTENCE_ABBR
        
        if is_multisentence:
            if self.nodes[id_root_id].concept == ID_ROOT_SYMBOL or self.nodes[id_root_id].concept == "and":
                self.nodes[id_root_id].concept = "and"
                root_abr = "a"
            else:
                self.nodes[id_root_id].concept = ID_MULTISENTENCE
                root_abr = ID_MULTISENTENCE_ABBR         
        d_abbr = {id_root_id:root_abr}
        
     
        added_nodes = set([])
        triplets = []
        
        nodes_id = [e[0] for e in sorted(self.nodes.items(), 
                                         key = lambda kv: len(kv[1].pred_children), 
                                         reverse=True)]
        
        for n in nodes_id:

            node = self.nodes[n]
            n_concept = self.nodes[n].concept
            n_id = self.nodes[n].id

            n_abbr = self.abbreviation_for_concept(d_abbr, node, n_id)
            
            if n_id not in added_nodes and not node.is_constant:
                triplets.append(penman.Triple(source=n_abbr.encode("utf-8"), relation=INSTANCE_TRIPLET,target=n_concept.encode("utf-8")))
                added_nodes.add(n_id)
            
            for r, childid in self.nodes[n].pred_children:
                
                child_node = self.nodes[childid]
                c_concept = self.nodes[childid].concept
                c_abbr = self.abbreviation_for_concept(d_abbr, child_node, childid)
         
                if childid not in added_nodes and not child_node.is_constant:
                    triplets.append(penman.Triple(source=c_abbr.encode("utf-8"),relation=INSTANCE_TRIPLET, target=c_concept.encode("utf-8")))
                    added_nodes.add(childid)
                
                triplets.append(penman.Triple(source=n_abbr.encode("utf-8"),relation=r,target=c_abbr.encode("utf-8")))
           

        top,triplets = self.get_graph_root(triplets, is_multisentence)
        
        
        new_triplets,n_snt = [], 1
        #new_triplets = []
        for t in triplets:
                        
            if t.relation == "*root*" or t.relation.startswith("snt"):  
                     
                if is_multisentence and self.nodes[id_root_id].concept == "and":
                    new_triplets.append(penman.Triple(source=t.source,relation="op"+str(n_snt), target=t.target))
                    n_snt +=1                    
                else:
                    new_triplets.append(penman.Triple(source=t.source,relation="snt"+str(n_snt), target=t.target))
                    n_snt +=1
            else:
                new_triplets.append(t)
         
        #Create a dummy AMR graph (for evaluation purposes) if an empty graph was predicted.       
        if new_triplets == []:
            top = EMPTY_GRAPH_ABBR #"e"
            new_triplets.append(penman.Triple(source=top,relation=INSTANCE_TRIPLET, target=EMPTY_GRAPH_CONCEPT))
            #new_triplets.append(penman.Triple(source="e",relation=INSTANCE_TRIPLET, target="empty-graph"))
        
        g = penman.Graph(data=new_triplets, top= top)
        return penman.encode(g)





########################################################################################
#                             OTHER USEFUL FUNCTIONS                                   #
########################################################################################



"""
Given a list of AMRGraphs it returns a tuple of Counters for the following information:
-words
-lemmas
-pos
-rels
-nodes
-entity
-deps
"""
def vocab(amr_graphs):
    
    wordsCount = Counter()
    lemmasCount = Counter()
    posCount = Counter()
    relCount = Counter()
    nodesCount = Counter()
    entityCount = Counter()
    depsCount = Counter()

    pos = []
    words = []
    rels = []
    nodes = []
    entities = []
    deps = []
    
    w_t_l = []
    
    for g in amr_graphs:
        
        for entry in g.sequence:
   
            pos.append(entry.word.pos)
            words.append(entry.word.form)
            entities.append(entry.word.entity)
            deps.append(entry.word.deprel)
            
        for arc in g.A:
            rels.append(arc.rel)
            
        for node in g.nodes.values():
            nodes.append(node.concept)
                     
    wordsCount.update(words)
    posCount.update(pos)
    relCount.update(rels)     
    nodesCount.update(nodes)   
    entityCount.update(entities)
    depsCount.update(deps)
   
    return (wordsCount, 
            lemmasCount,
            posCount, 
            relCount,
            nodesCount,
            entityCount,
            depsCount)
