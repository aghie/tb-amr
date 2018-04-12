from operator import itemgetter
from itertools import chain

import utils


"""
Class implementing a 5-tuple that represents a configuration 
for the AMR-Covington algorithm
"""
class CovingtonConfiguration(object):
    """
    Based on Nivre, J. (2008). Algorithms for deterministic incremental dependency parsing. 
    Computational Linguistics, 34(4), 513-553.
    
    l1: Word Id of the word at the top of the lambda one list
    b: Word Id of the word at the top of the buffer
    sentence: The sequence of AMRListEntries
    A: Set of edges created
    """
    
    def __init__(self,l1,b,sequence, N, A, pred_concepts={}):
        
        self.l1 = l1
        self.b = b
        self.sequence = sequence
        self.N = N
        self.A = A
                
        self.pred_concepts = pred_concepts
        self.pred_current_id = -1
        
        self.node_ids = []
        for n in self.N:
        
            self.node_ids.append(n.id)
    

    def _str_list(self, sequence):
        
        entries = []
        for entry in sequence:
            
            if entry.is_node:
                entries.append("n("+entry.node.concept+"["+str(entry.word.index)+","+str(entry.word.index+1)+","+str(entry.node.id)+"])")
            else:
                entries.append(entry.full_form+"["+str(entry.word.index)+"]")
        
        return ",".join(entries)        
        
        
    def __str__(self):
        
        str_L1 = "L1["+self._str_list(self.sequence[0:(self.l1+1)])+"]"
        str_L2 = "L2["+self._str_list(self.sequence[(self.l1+1):self.b])+"]"
        str_B = "B["+self._str_list(self.sequence[self.b:])+"]"
        str_A = ""#"\nA={"+",".join([str(a) for a in self.A])+"}\n"
        
        return (" ".join([str_L1,str_L2,str_B])+"\t l1 focus word: "+str(self.l1)+" "+str_A+" "+"\t b focus word: "+str(self.b)).encode("utf-8")





"""
Class implementing the AMR-Covington transition-based algorithm as defined in:

David Vilares and Carlos Gomez-Rodriguez. A Transition-based Algorithm for Unrestricted AMR Parsing,
NAACL HLT 2018 - The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 
"""
class AMRCovington(object):
    
    
    #TRANSITIONS
    LEFT_ARC = 0
    RIGHT_ARC = 1
    SHIFT = 2
    NO_ARC = 3
    CONFIRM = 4
    DESGLOSE=5
    REDUCE = 6
    MULTIPLE_ARC= 7
    
    TRANSITIONS = [LEFT_ARC, RIGHT_ARC, SHIFT, NO_ARC, CONFIRM, DESGLOSE, REDUCE, MULTIPLE_ARC]  
    ARC_TRANSITIONS = [LEFT_ARC,RIGHT_ARC,MULTIPLE_ARC]  
    NODE_TRANSITIONS = [CONFIRM, DESGLOSE]
    SIZE_TRANSITIONS = len(TRANSITIONS)
    
    
    EMPTY_ENTRY= utils.AMRListEntry(["EMPTY_WORD"])
    
    BUFFER_FEATURE = "B"
    L1_FEATURE = "L1"
    L2_FEATURE = "L2"
    OTHER_FEATURE = "O"
    
    
    def __init__(self,wB=2, wL1=2, wL2r=0, wL2l=0, use_shift=True):

        self.wB = wB
        self.wL1 = wL1
        self.wL2r = 0 #Still not supported
        self.wL2l = 0 #Still not supported
        
        self.use_shift = use_shift


    @property
    def window_size(self):
        return self.wB+self.wL1+self.wL2l+self.wL2r



    def generate_rels_per_arc(self, rels, arcs= ARC_TRANSITIONS):
        
        arcs_rels = []
        for arc in self.ARC_TRANSITIONS:
            for r in rels:
                
                if utils.MULTIPLE_DIRECTION_RELATION in r:
                    if arc == self.MULTIPLE_ARC: arcs_rels.append(arc+"_"+r)
                else: 
                    if arc in [self.LEFT_ARC, self.RIGHT_ARC]:
                        arcs_rels.append(str(arc)+"_"+r)
                
        return arcs_rels



    def true_static_oracle(self,c, amrGraph):
         
 
        cur_index = c.sequence[c.b].indexed_at[0:2]
        n_to_create =  amrGraph.nodes_indexed_at[cur_index] if cur_index in amrGraph.nodes_indexed_at else []
        n_to_create = [n.id for n in n_to_create]   
        n_created = [n.id for n in c.N
                     if cur_index == n.indexed_at[0:2]]
        #             and n.id is not None]
         

         
        if self._is_valid_reduce(c) and (len(n_to_create) - len(n_created) == 0):
            return (self.REDUCE, None)
 
        if (self._is_valid_confirm(c) and (len(n_created) + 1 == len(n_to_create))):
            node_id = amrGraph.node_id(cur_index[0:2],c.node_ids)
             
            return (self.CONFIRM, amrGraph.nodes[node_id].concept)
         
        if self._is_valid_desglose(c) and len(n_created) + 1 < len(n_to_create):
            node_id = amrGraph.node_id(cur_index[0:2],c.node_ids)
            return (self.DESGLOSE, amrGraph.nodes[node_id].concept)
         
 
        #Checking what ARC/NO-ARC to create
        if c.sequence[c.l1].is_node and c.sequence[c.b].is_node:
 
            l1_id = c.sequence[c.l1].node.id
            b_id  = c.sequence[c.b].node.id
             
            multiarc = (l1_id, b_id) in amrGraph.arcs_at and (b_id, l1_id) in amrGraph.arcs_at
 
            if multiarc:
                multiarc_label = utils.MULTIPLE_DIRECTION_RELATION.join([amrGraph.arcs_at[(b_id,l1_id)], amrGraph.arcs_at[(l1_id,b_id)]])
 
                 
            if self._is_valid_multiple_arc(c) and multiarc:
                return (self.MULTIPLE_ARC, multiarc_label)             
             
 
            if self._is_valid_left_arc(c) and (b_id,l1_id) in amrGraph.arcs_at:
                return (self.LEFT_ARC, amrGraph.arcs_at[(b_id,l1_id)])
 
            if self._is_valid_right_arc(c) and (l1_id,b_id) in amrGraph.arcs_at:
                return (self.RIGHT_ARC, amrGraph.arcs_at[(l1_id,b_id)])
             
             
        is_shift = True
        bn = c.sequence[c.b].node
        for i in range(c.l1,-1,-1):
             
            l1n = c.sequence[i].node
 
            if (l1n.id,bn.id) in amrGraph.arcs_at or (bn.id, l1n.id) in amrGraph.arcs_at:
                is_shift = False 
                break
             
        if (self._is_valid_shift(c) and is_shift and self.use_shift) or (not self.use_shift and c.l1 < 0):
            return (self.SHIFT, None)     
         
        if (self._is_valid_no_arc(c)) or (not self.use_shift and c.l1 <=0):    
            return (self.NO_ARC, None)

    
        

    """
    Predicts the transition to take based on a configuration c and 
    @param: c: CovingtonConfiguration
    @param scores_t: A numpy array. Each index i contains the score 
    the transition i 
    """
    def predict(self, c, scores_t):

        desglosed = sum([1 for e in c.sequence[:c.l1] 
                         if c.sequence[c.b].word.index == e.word.index])        
        
        def single_root_assigned(c):
            entry = c.sequence[0]
            
            return c.l1 == 0 and entry.node.concept == utils.ID_ROOT_SYMBOL and len(entry.node.pred_children) > 0
        
            
        left_arc = [(self.LEFT_ARC, scores_t[self.LEFT_ARC])] if self._is_valid_left_arc(c) else []    
        right_arc = [(self.RIGHT_ARC, scores_t[self.RIGHT_ARC])] if self._is_valid_right_arc(c) else []
        multiple_arc = [(self.MULTIPLE_ARC, scores_t[self.MULTIPLE_ARC])] if self._is_valid_multiple_arc(c) else []
        shift =  [(self.SHIFT, scores_t[self.SHIFT])] if ((self._is_valid_shift(c) and self.use_shift) or (not self.use_shift and c.l1 <= 0)) else []                   
        no_arc = [(self.NO_ARC,  scores_t[self.NO_ARC])] if self._is_valid_no_arc(c) or (not self.use_shift and c.l1 <= 0) else []                      
        reduce =  [(self.REDUCE,  scores_t[self.REDUCE])] if self._is_valid_reduce(c) else []      
        confirm =  [(self.CONFIRM,  scores_t[self.CONFIRM])] if self._is_valid_confirm(c) else []                                                       
        desglose = [(self.DESGLOSE, scores_t[self.DESGLOSE])] if self._is_valid_desglose(c) and desglosed < 4 else []  #len(self.MAX_DESGLOSE) else []                                           
        valid_actions = [left_arc, right_arc, shift, no_arc, reduce, confirm, desglose, multiple_arc]

        best = max(chain(*valid_actions), key = itemgetter(1))
        t = best[0]
        
        return t
    
    
    """
    Extracts the entries from the top positions in L1 and B
    @param c: A CovingtonConfiguration instance
    """    
    def get_state_info(self, c):

        top_L1 = [(c.sequence[c.l1-j], self.L1_FEATURE) 
                  for j,s in enumerate(c.sequence[0:c.l1+1]) if j < self.wL1]
        top_B = [(c.sequence[c.b+j], self.BUFFER_FEATURE) 
                 for j,s in enumerate(c.sequence[c.b:]) if j < self.wB]
      #  top_L2l = [(c.sequence[c.l1+j], self.L2_FEATURE) for j,s in enumerate(c.sequence[c.l1+1:c.b]) if j < self.wL2l]
        top_S = self._pad(top_L1,self.wL1)
        top_B = self._pad(top_B, self.wB)
        return (top_B,top_S)
        



    """
    Updates c to a new configuration, based on best
    @param best: A tuple (action, label) with the most likely action
    predicted by the classifier
    @param c: A Covington Configuration
    @param node_id: Node ID to be assigned in case the predicted action
    is a CONFIRM or BREAKDOWN
    """
    def update_state(self, best, c, node_id = None):
        
        l1 = c.l1
        b = c.b
        
        nodes = c.N
        arcs = c.A
    
        if best[0] == self.LEFT_ARC:
            
            pred_rel = best[1]
            head_node = c.sequence[c.b].node
            dependent_node = c.sequence[c.l1].node
            best_op = self.LEFT_ARC
            l1 = l1 -1
            
            for r in pred_rel.split(utils.COMPOSITE_RELATION):
            
                edge = utils.AMREdge(head_node, r ,dependent_node)
                arcs.append(edge)                  
                dependent_node.add_pred_relation((r, head_node.id))
                head_node.add_pred_children((r,dependent_node.id))
                              
        elif best[0] == self.RIGHT_ARC:
            
            pred_rel = best[1]
            head_node = c.sequence[c.l1].node
            dependent_node = c.sequence[c.b].node
            best_op = self.RIGHT_ARC                    
            l1 = l1-1
            for r in pred_rel.split(utils.COMPOSITE_RELATION):
            
                edge = utils.AMREdge(head_node, r ,dependent_node)
                arcs.append(edge)                  
                dependent_node.add_pred_relation((r, head_node.id))
                head_node.add_pred_children((r,dependent_node.id))
            
        #This transition is not learned in practice, but just in case
        #Multi-arc created an edge from b to l1 (a LEFT-ARC_l) and an
        #edge from l1 to b (a RIGHT-arc l2)
        elif best[0] == self.MULTIPLE_ARC:
            
            pred_rel = best[1].split(utils.MULTIPLE_DIRECTION_RELATION)
            ori_edge = pred_rel
            pred_rLEFT = pred_rel[0]
            pred_rRIGHT = pred_rel[1]

            l1_node = c.sequence[c.l1].node
            b_node = c.sequence[c.b].node
            
            #LEFT-ARC edge
            for r in pred_rLEFT.split(utils.COMPOSITE_RELATION):
                edge = utils.AMREdge(b_node, r, l1_node)
                arcs.append(edge)
                l1_node.add_pred_relation((r, b_node.id))
                b_node.add_pred_children((r,l1_node.id))
            
            #RIGHT-ARC edge
            for r in pred_rRIGHT.split(utils.COMPOSITE_RELATION):
                edge2 = utils.AMREdge(l1_node, r ,b_node)                
                arcs.append(edge2)                
                b_node.add_pred_relation((r, l1_node.id))
                l1_node.add_pred_children((r,b_node.id))

            best_op = self.MULTIPLE_ARC
            l1 = l1-1                      
        
        elif best[0] == self.SHIFT:
            l1 = b
            b = b + 1
        elif best[0] == self.NO_ARC:
            l1 = l1 - 1
        elif best[0] == self.CONFIRM:
            concept = best[1]
            indexed = c.sequence[c.b].indexed_at 
            node = utils.AMRNode(concept,indexed[0], indexed[1],id=node_id,
                                pred_relations=set([]), pred_children=set([]), created_by=self.CONFIRM,
                                last_rel_as_head = None,
                                unaligned=False,
                                originated_from = c.sequence[c.b].word)
            
            c.sequence[c.b].node = node
            nodes.add(node)
             
            if indexed not in c.pred_concepts:
                c.pred_concepts[indexed] = [concept]
            else:
                c.pred_concepts[indexed].append(concept)
                   
        elif best[0] == self.REDUCE:
            c.sequence.pop(c.b)
        elif best[0] == self.DESGLOSE:     
            concept = best[1]
            indexed = c.sequence[c.b].indexed_at    
            if indexed not in c.pred_concepts:
                c.pred_concepts[indexed] = [concept]
            else:
                c.pred_concepts[indexed].append(concept)

            node = utils.AMRNode(concept,indexed[0],indexed[1],
                                 id=node_id, pred_relations=set([]), pred_children=set([]),
                                 created_by = self.DESGLOSE,
                                 last_rel_as_head = None,
                                 unaligned=False,
                                 originated_from = c.sequence[c.b].word)

            nodes.add(node)
              
            #Dummy word
            if node_id is not None:
                word = c.sequence[c.b].word 
            else:
                word = c.sequence[c.b].word
            
            amr_list_entry = utils.AMRListEntry(word, node=node, edges = set([]))    
            c.sequence.insert(c.b, amr_list_entry)
            
        c = CovingtonConfiguration(l1,b,c.sequence,nodes,arcs, c.pred_concepts)
        return c






    def _pad(self, l, expected_size):        
        if expected_size > len(l):
            l.extend([(self.EMPTY_ENTRY, self.OTHER_FEATURE)]*(expected_size-len(l)))
        return l
    
    def _is_final_state(self,c):
        return c.b >= len(c.sequence)    
            
    def _is_valid_confirm(self,c):
        return not c.sequence[c.b].is_node
    
    def _is_valid_reduce(self,c):
        return not c.sequence[c.b].is_node
    
    def _is_valid_shift(self,c):
        return c.sequence[c.b].is_node
    
    def _is_valid_no_arc(self,c):
        return c.l1 > 0 and c.sequence[c.b].is_node
    
    def _is_valid_left_arc(self,c):
        return (c.l1 >= 0 and c.sequence[c.b].is_node 
                and not self._is_leaf_node(c.sequence[c.b]))

    def _is_valid_right_arc(self,c):
        return (c.sequence[c.b].is_node and c.l1 >= 0 
                and not self._is_leaf_node(c.sequence[c.l1]))

    def _is_valid_multiple_arc(self,c):
        return (c.l1 >= 0 and c.sequence[c.b].is_node 
                and not self._is_leaf_node(c.sequence[c.b])
                and not self._is_leaf_node(c.sequence[c.l1]))


    """
    It is is a node, we can breakdown it.
    """    
    def _is_valid_desglose(self,c):
        return not (c.sequence[c.b].is_node) 
    
    
    """
    Returns if the node can be the head of the relationship
    If the node is a "string" node, it cannot be a head node in the relationship
    """
    def _is_leaf_node(self,entry):
        if entry.is_node:
            return entry.node.concept.startswith("\"") and entry.node.concept.endswith("\"")
        else:
            return False



    
    