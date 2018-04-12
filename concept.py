# -*- coding: utf-8 -*-
from datetime import datetime
from dateutil.parser import parse
from collections import Counter
from pattern.en import verbs, conjugate, INFINITIVE, singularize
from algorithm import CovingtonConfiguration

import codecs
import copy
import string

class ConceptModel(object):
    
    def __init__(self):
        pass
    
"""
Class implementing a RuleBasedConceptModel. It transforms single-
and multi-word expressions into a sequence of transitions that allow
to transform the expression into a concept.
"""
class RuleBasedConceptModel(object):
    
    POLARITY_DEP = "polarity"
    VALUE_DEP = "value"
    QUANTITY_DEP = "quant"
    UNIT_DEP = "unit"
    DAY_DEP = "day"
    MONTH_DEP = "month"
    YEAR_DEP = "year"
    DECADE_DEP = "decade"
    WEEKDAY_DEP = "weekday"
    WIKI_DEP = "wiki"
    NAME_DEP = "name"
    
    POLARITY_NODE = "-"
    
    PERCENT_CONCEPT = "percentage-entity"
    TIME_CONCEPT = "temporal-quantity"
    MONETARY_CONCEPT = "monetary-quantity"
    DATE_CONCEPT = "date-entity"
    COUNTRY_CONCEPT  = "country"
    
    LOCATION_ENTITY = "LOCATION"
    PERSON_ENTITY = "PERSON"
    ORGANIZATION_ENTITY = "ORGANIZATION"
    DATE_ENTITY = "DATE"
    PERCENT_ENTITY = "PERCENT"
    TIME_ENTITY = "TIME"
    MONEY_ENTITY = "MONEY"
    OTHER_ENTITY = "O"
    
    ENTITIES = [LOCATION_ENTITY,PERSON_ENTITY,ORGANIZATION_ENTITY,DATE_ENTITY,
                OTHER_ENTITY,PERCENT_ENTITY,TIME_ENTITY,MONEY_ENTITY]
    
    """
    path_nationalities, path_nationalities2, path_cities, path_countries,
    path_states, path_negations follow the format of the files that can be obtained
    from:
    https://github.com/mdtux89/amr-eager
    
    path_verbalizations follows the format of the file provided at:
    https://github.com/c-amr/camr/blob/master/resources/verbalization-list-v1.01.txt
    
    @param
    @param algorithm: A transiton-based alfogithm from algorithm
    @param graph_templates:
    @param path_nationalities: 
    @param path_nationalities2:
    @param path_cities: A path to a file containing a list of cities
    @param path_countries: A path to a file containing a list of countries
    @param path_states: A path to a file containing a list of states
    @param path_negations: A path to a list of words that are mapped to polarity nodes
    @param path_verbalizations: A path to a list of verbalizations.
    """
    def __init__(self, vocab, algorithm, graph_templates,path_nationalities,
                 path_nationalities2, path_cities,path_countries, 
                 path_states, path_negations, path_verbalizations,
                 multiword_graph_templates):
        
            
        self.vocab = vocab
        
        self.algorithm = algorithm
        self.graph_templates = graph_templates
        self.multiword_graph_templates = multiword_graph_templates
        
        with codecs.open(path_cities) as f:
            self.cities = set([l.strip(",\n") for l in f.readlines()])

        with codecs.open(path_states) as f:
           self.states = set([l.strip(",\n") for l in f.readlines()])             

        with codecs.open(path_countries) as f:
           self.countries = set([l.strip(",\n") for l in f.readlines()])            

        with codecs.open(path_negations) as f:
           self.negations = {l.split()[0]:l.split()[1].strip("\n") 
                             for l in f.readlines()}         
         
        with codecs.open(path_nationalities) as f:
            self.nationalities = {l.split("=>")[1].strip().lower().replace("'", "").replace(",", ""):self._uppercase_first_char(l.split("=>")[0].strip().replace("'", "").replace(",", ""))
                                  for l in f.readlines()}

        with codecs.open(path_nationalities2) as f:
            nationalities2 = {l.split("\t")[1].strip().lower():l.split("\t")[0] 
                  for l in f.readlines()}        
            
        self.nationalities.update(nationalities2)
        self.verbalizations = self.load_verbalizations(path_verbalizations)



    def _uppercase_first_char(self,element):
        
        final = []
        for e in element.split():
            final.append(e[0].upper()+e[1:])
            
        return " ".join(final)


    def set_lookup_concepts(self, path_lookup):
        with codecs.open(path_lookup) as f:
            self.lookup_concepts = {l.split("\t")[0]:l.split("\t")[1].strip() 
                                    for l in f.readlines()}



    #############################################################################
    #             SINGLE-WORD TO CONCEPT TRANSFORMATION FUNTIONS                #
    #############################################################################


    """
    Transforms a single word concept into the corresponding
    set of transitions to create the corresponding subgraph
    @param An isntance of an AMRWord
    @param threshold: Threshold based on the frequency of the word
    in the training set, to determine if apply an ad-hoc procedure
    to map the word into a concept
    """
    def word_to_concept(self, amr_word, threshold=5):
        
        concept = None
        if amr_word.pos.startswith("N"):

            form = amr_word.form
            norm = amr_word.norm
            lemma = amr_word.lemma        
        else:
            form = amr_word.form.lower()
            norm = amr_word.norm.lower()
            lemma = amr_word.lemma.lower()
        pos = amr_word.pos   
             
        try:
            float(form)
            return self._transition_sequence_from_verbalization(form)
        except ValueError:
            pass
        
        """
        'sum' of punctuation symbols is intended to avoid problems 
        with the AMR format
        """
        if self.vocab[form] < threshold and sum([1 for char in form
                                                if char in string.punctuation]) == 0:
                                                    
             if singularize(lemma) in self.lookup_concepts:
                 concept = self._transition_sequence_from_verbalization(self.lookup_concepts[singularize(lemma)])
             elif self._is_verb(pos):          
                 lemma = unicode(conjugate(lemma, tense=INFINITIVE))
                 concept = self._transition_sequence_from_verbalization(lemma + "-01")
             else:                  
                 concept = self._transition_sequence_from_verbalization(singularize(lemma))        
          
        return concept


    """
    Determines if the entry at the top of the buffer is the beginning
    of a graph template
    @param A parsing configuration
    """
    def configuration_starts_graph_template(self, c):
        is_node = c.sequence[c.b].is_node
        is_nationality = c.sequence[c.b].word.form.lower() in self.nationalities
        return (is_nationality or c.sequence[c.b].word.components is not None) and not is_node      
    
    
    def _phrase_subgraph(self, amr_words):
    
        curdict, curgraph = self.multiword_graph_templates, None
        first = None
        for amr_word in amr_words:            
            try:
                curdict, curgraph = curdict[amr_word.form]
                if first is None: 
                    first=amr_word
            except KeyError:
                if curgraph is not None:
                    return curgraph
                return None
        return curgraph  


    """
    Looks at the current configuration and checks if it can identify
    some template, mapping it to the corresponding transition sequence
    that leads to its AMR graph
    
    It maps words found in the verbalization list
    
    It maps entities identified by the Stanford CoreNLP as
    "LOCATION","PERSON","ORGANIZATION", "DATE","O","PERCENT","TIME","MONEY".
    
    It also maps nationatilies from path_nationalities and path_nationalities2
    into.
    
    Returns None if mapped to nothing. Otherwise, returns the list of transitions
    that lead to the creation of the AMR subgraph
    
    @param c: A parsing configuration
    """
    def transition_sequence_from_template(self, c, training_phase=False):

        if c.sequence[c.b].word.form.lower() in self.nationalities:
            return self._generate_nationality(c.sequence[c.b].word)
        
        form = c.sequence[c.b].word.form.lower()
        norm = c.sequence[c.b].word.norm.lower()
        lemma = c.sequence[c.b].word.lemma.lower()
        
        if form in self.verbalizations:
            return self._transition_sequence_from_verbalization(self.verbalizations[form])         
        elif norm in self.verbalizations:
            return self._transition_sequence_from_verbalization(self.verbalizations[norm])      
        elif lemma in self.verbalizations:
            return self._transition_sequence_from_verbalization(self.verbalizations[lemma])  

        entity_type = c.sequence[c.b].word.entity      
        components = c.sequence[c.b].word.components
        
        #We look if the entity detected is stored as a AMR template graph
        if (not training_phase):
            aux_graph = self._phrase_subgraph(components)
            if aux_graph is not None:
                return self._amr_template_to_transition_sequence(aux_graph)
            
            if (" ".join([e.form for e in components]), entity_type) in self.graph_templates and entity_type in self.ENTITIES:            
                return self._amr_template_to_transition_sequence(self.graph_templates[(" ".join([e.form for e in components]),entity_type)])         
        
        
        #Otherwise we apply an ad-hoc procedure to transforms the entity into 
        #a sequence of transitions   
        entry_term = "_".join([e.form for e in components])
        #TODO: Not an elegant way to deal with AMR reserved symbols
        components = [com for com in components if com.form != "\""]
        
        if entity_type == self.LOCATION_ENTITY:                   
            if entry_term in self.countries:
                return self._generate_location(components,"country")
            elif entry_term in self.cities:
                return self._generate_location(components,"city")
            elif entry_term in self.states:
                return self._generate_location(components,"state")
            else:                
                return self._generate_location(components,"city")
              
        if entity_type ==self. PERSON_ENTITY:
            return self._generate_location(components, "person") 
        
        #TODO: See how to improve the way to predict the particular
        #type of organization
        if entity_type == self.ORGANIZATION_ENTITY:
            if "university" in [e.form.lower() for e in components]:
                organization_type = "university"
            elif "department" in [e.form.lower() for e in components] or "ministry" in [e.form.lower() for e in components]:
                organization_type = "government-organization"       
            elif "center" in [e.form.lower() for e in components]:
                organization_type = "research-institute"
            else:
                organization_type = "organization"    
                
            return self._generate_location(components, organization_type)

        if entity_type == self.DATE_ENTITY:
            return self._generate_date(components) 
        
        if entity_type == self.MONEY_ENTITY:
            return self._generate_money(components)
        
        if entity_type == self.PERCENT_ENTITY:
            return self._generate_percent(components)
        
        if entity_type == self.TIME_ENTITY:
            return self._generate_time(components)
        
        
        #NEGATIONS COMMONLY NOT ALIGNED BY THE AMR-ALIGNER: without and never
        #We map then directly during the test phase
        if not (training_phase):
        
            if e.form in ["without","never"]:
                return self._transition_sequence_from_verbalization("-")
            
            if e.form in self.negations:
                return self._generate_negation(e.form)
            elif e.norm in self.negations:
                return self._generate_negation(e.norm)
            elif e.norm in self.negations:
                return self._generate_negation(e.lemma)
        
        return None    
        

    """
    Transform an AMR template graph from graph templates into the
    transition sequence that will lead to create it.
    """
    def _amr_template_to_transition_sequence(self, graph_template):
        
        sequence = [] 
        template_copy = copy.deepcopy(graph_template)
        sequence_words = [e.word for e in template_copy.sequence[0:]]     
        l1 = 0
        b = 1
        arcs = []               
        nodes = set([template_copy.sequence[l1].node])
        
        c = CovingtonConfiguration(l1, b, template_copy.sequence, nodes, arcs,
                                         pred_concepts={})
      
        while not self.algorithm._is_final_state(c):
            t, label = self.algorithm.true_static_oracle(c, template_copy)
            #The root edge must not be created for subgraphs
            if t == self.algorithm.RIGHT_ARC and label == "*root*":
                break  
            
            sequence.append((t, label))
            indexed = c.sequence[c.b].indexed_at
            
            node_id = graph_template.node_id((indexed[0], indexed[1]),
                                        c.node_ids)
            
            c = self.algorithm.update_state((t, label), c, node_id)
 
        return sequence

    #TODO: This has a neglibile impact
    def _generate_negation(self, negation):
        
        sequence = []
        sequence.append((self.algorithm.DESGLOSE, self.POLARITY_NODE))
        sequence.append((self.algorithm.SHIFT, None))
        sequence.append((self.algorithm.CONFIRM, negation))
        sequence.append((self.algorithm.LEFT_ARC, self.POLARITY_DEP))
        return sequence


    """
    Given a list of AMREntry's representing a percent entity, returns
    the sequence of transitions to create the corresponding subgraph
    """
    def _generate_percent(self, entries):
        
        sequence = []
        arcs = []
                
        for entry in entries:
            try:
                value = float(entry.form)
                sequence.append((self.algorithm.DESGLOSE, entry.form))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.VALUE_DEP)
            except ValueError:
                sequence.append((self.algorithm.REDUCE, None))
                
        for i in range(len(arcs)):
            sequence.append((self.algorithm.REDUCE, None))        
        
        sequence.append((self.algorithm.CONFIRM, self.PERCENT_CONCEPT))
            
        for arc in reversed(arcs):
            sequence.append((self.algorithm.LEFT_ARC, arc))
            
        return sequence    


    """
    Given a list of AMREntry's representing a time entity, returns
    the sequence of transitions to create the corresponding subgraph
    """
    def _generate_time(self, entries):
        def time_unit(w):
            time_unit = [u"year", u"month", u"day", u"hour", u"minute", u"second"]
            for unit in time_unit:
                if w.startswith(unit): return unit
                
            return None
        
        
        sequence = []
        arcs = []
        quantity_found = False   
        for entry in entries:
            
            try:   
                value = float(entry.form)
                quantity_found = True
                sequence.append((self.algorithm.DESGLOSE, entry.form))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.QUANTITY_DEP)   
            except ValueError:
                
                if time_unit(entry.form):
                    sequence.append((self.algorithm.DESGLOSE, time_unit(entry.form)))
                    sequence.append((self.algorithm.SHIFT, None))
                    arcs.append(self.UNIT_DEP)
                else:                                  
                    sequence.append((self.algorithm.REDUCE, None))
        
        if not quantity_found:
            sequence.append((self.algorithm.DESGLOSE, "1"))
            sequence.append((self.algorithm.SHIFT, None))
            arcs.append(self.QUANTITY_DEP)
            
        
        for i in range(len(arcs)):
            sequence.append((self.algorithm.REDUCE, None))        
        
        sequence.append((self.algorithm.CONFIRM, self.TIME_CONCEPT))
            
        for arc in reversed(arcs):
            sequence.append((self.algorithm.LEFT_ARC, arc))
            
        return sequence    



    """
    Given a list of AMREntry's representing a money entity, returns
    the sequence of transitions to create the corresponding subgraph
    """
    def _generate_money(self, entries):
        
        def monetary_unit(w_aux):
            monetary_unit = [u"yuan", u"dollar", u"euro", u"€", u"$",u"rupee",u"pound",u"£"]
            
            monetary_dict = {u"yuan":u"yuan",
                             u"dollar":u"dollar",
                             u"euro": u"euro",
                             u"$":u"dollar",
                             u"€":u"euro",
                             u"£":u"pound",
                             u"rupee":u"rupee",
                             u"pound":u"pound"}
            
            for unit in monetary_unit:
                if w_aux.startswith(unit): return monetary_dict[unit]
                
            return None
        
        
        
        sequence = []
        arcs = []
        has_country_mod = False
        country_mod_nodes = 0
        for nw, w in enumerate(entries):
            
            w_aux = w.form.lower()
            
            if self._is_float(w_aux.replace(",","")):
                
                quantity = w_aux.replace(",","")
                #We check the next word to see if there is a million/billion
                if nw+1 < len(entries) and entries[nw+1].form.lower() in ["million"]:
                    quantity = str(int(float(w_aux)*1000000))
                #We check the next word to see if there is a million/billion
                if nw+1 < len(entries) and entries[nw+1].form.lower() in ["billion"]:
                    quantity = str(int(float(w_aux)*1000000000))
                    
                sequence.append((self.algorithm.DESGLOSE, quantity))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.QUANTITY_DEP)
        
            elif monetary_unit(w_aux) is not None:
                sequence.append((self.algorithm.DESGLOSE, monetary_unit(w_aux)))
                
                if has_country_mod:
                    sequence.append((self.algorithm.LEFT_ARC, "mod"))
                    
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.UNIT_DEP)
#             
            elif (w.form,"LOCATION") in self.graph_templates:
                has_country_mod = True
                sequence.extend(self._amr_template_to_transition_sequence(self.graph_templates[(w.form,"LOCATION")]))
                
                country_mod_nodes = sum([1 for (t,l) in self._amr_template_to_transition_sequence(self.graph_templates[(w.form,"LOCATION")])
                                     if t in [self.algorithm.DESGLOSE, self.algorithm.CONFIRM]])
                sequence.append((self.algorithm.SHIFT, None))
                
                    
            
            else:
                sequence.append((self.algorithm.REDUCE, None))
        
        for i in range(len(arcs)):
            sequence.append((self.algorithm.REDUCE, None))
                
        
        sequence.append((self.algorithm.CONFIRM, self.MONETARY_CONCEPT))
        
        for arc in reversed(arcs):
            sequence.append((self.algorithm.LEFT_ARC, arc))
            if arc == self.UNIT_DEP:
                sequence.extend([(self.algorithm.NO_ARC,None)]*country_mod_nodes)
        return sequence
    


    """
    Given a list of AMREntry's representing a date entity, returns
    the sequence of transitions to create the corresponding subgraph
    """
    def _generate_date(self, date_terms):
        
        def is_int(number):
            try:
                int(number)
                return True
            except ValueError:
                return False
            
        days_ordinal = {"1st":"1",
                        "2nd":"2",
                        "3rd":"3",
                        "4th":"4",
                        "5th":"5",
                        "6th":"6",
                        "7th":"7",
                        "8th":"8",
                        "9th":"9",
                        "10th":"10",
                        "11th":"11",
                        "12th":"12",
                        "13th":"13",
                        "14th":"14",
                        "15th":"15",
                        "16th":"16",
                        "17th":"17",
                        "18th":"18",
                        "19th":"19",
                        "20th":"20",
                        "21st":"21",
                        "22nd":"22",
                        "23rd":"23",
                        "24th":"24",
                        "25th":"25",
                        "26th":"26",
                        "27th":"27",
                        "28th":"28",
                        "29th":"29",
                        "30th":"30",
                        "31st":"31"}
        
        months = {"january":"1",
                  "february":"2",
                  "march":"3",
                  "april":"4",
                  "may":"5",
                  "june":"6",
                  "july":"7",
                  "august":"8",
                  "september":"9",
                  "october":"10",
                  "november":"11",
                  "december":"12",
                  "jan":"1",
                  "feb":"2",
                  "mar":"3",
                  "apr":"4",
                  "may":"5",
                  "jun":"6",
                  "jul":"7",
                  "aug":"8",
                  "sep":"9",
                  "oct":"10",
                  "nov":"11",
                  "dec":"12",
                  "jan.":"1",
                  "feb.":"2",
                  "mar.":"3",
                  "apr.":"4",
                  "may.":"5",
                  "jun.":"6",
                  "jul.":"7",
                  "aug.":"8",
                  "sep.":"9",
                  "oct.":"10",
                  "nov.":"11",
                  "dec.":"12"}
        
        weekday = {"monday":"monday",
                   "tuesday":"tuesday",
                   "wednesday":"wednesday",
                   "thrusday":"thrusday",
                   "friday":"friday",
                   "saturday":"saturday",
                   "sunday":"sunday",
                   "mon":"monday",
                   "tue":"tuesday",
                   "wed":"wednesday",
                   "thr":"thrusday",
                   "fri":"friday",
                   "sat":"saturday",
                   "sun":"sunday"}
        
        sequence = []
        arcs = []
        for nw, w in enumerate(date_terms):
            
            w = w.form.lower()
            
            try:
                steps = None
                date_split = w.split("-") 
                if len(date_split) == 3:
                    if date_split[1] == "00":
                        dt = datetime.strptime(date_split[1], '%Y')
                        steps = [(str(dt.year), "year")]                   
                    elif date_split[2] == "00":
                        dt = datetime.strptime("-".join(date_split[0:2]), '%Y-%m')
                        steps = [(str(dt.month), "month"), (str(dt.year), "year")]                        
                    else:                
                        dt = datetime.strptime(w, '%Y-%m-%d')
                        steps = [(str(dt.day), "day"), (str(dt.month), "month"), (str(dt.year), "year")]
                
                if steps is None: raise ValueError
                
                for value, r in steps:
            
                    if value != 0:
                        sequence.append((self.algorithm.DESGLOSE, value))
                        sequence.append((self.algorithm.SHIFT, None))
                        arcs.append(r)                       
                datetime_found = True
                continue
            except ValueError:
                datetime_found = False
            
            if w in months:
                sequence.append((self.algorithm.DESGLOSE, months[w]))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.MONTH_DEP)
        
            elif (is_int(w) and int(w) in range(1, 32)):
                day = w if not w.startswith("0") else w[1:]                
                sequence.append((self.algorithm.DESGLOSE, day))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.DAY_DEP)
            elif w in days_ordinal:
                sequence.append((self.algorithm.DESGLOSE, days_ordinal[w]))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.DAY_DEP)      
            elif (is_int(w) and len(w) == 4):
                sequence.append((self.algorithm.DESGLOSE, w))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.YEAR_DEP)
            elif  (w.endswith("s") and is_int(w[:-1])):
                sequence.append((self.algorithm.DESGLOSE, w[:-1]))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.DECADE_DEP)
            elif w in weekday:
                sequence.append((self.algorithm.DESGLOSE, w))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append(self.WEEKDAY_DEP)  
            else:
                sequence.append((self.algorithm.REDUCE, None))
  
        sequence.append((self.algorithm.CONFIRM, self.DATE_CONCEPT))    
        for arc in reversed(arcs):
            sequence.append((self.algorithm.LEFT_ARC, arc))
        return sequence


    """
    Given an AMREntry representing a nationality, returns
    the sequence of transitions to create the corresponding subgraph
    """
    def _generate_nationality(self, entry):
        sequence = []
        
        nationality = self.nationalities[entry.form.lower()]
        arcs = []
        
        # Generating the subgraph for the 'name' node
        sequence.append((self.algorithm.DESGLOSE, "\"" + nationality + "\""))
        sequence.append((self.algorithm.SHIFT, None))
            
        sequence.append((self.algorithm.DESGLOSE, self.NAME_DEP))

        sequence.append((self.algorithm.LEFT_ARC, "op1"))
        sequence.append((self.algorithm.SHIFT, None))
        
        # Generating the wiki    
        sequence.append((self.algorithm.DESGLOSE, "\"" + nationality + "\""))   
        sequence.append((self.algorithm.SHIFT, None))  
            
        sequence.append((self.algorithm.CONFIRM, self.COUNTRY_CONCEPT))
        sequence.append((self.algorithm.LEFT_ARC, self.WIKI_DEP))
        sequence.append((self.algorithm.LEFT_ARC, self.NAME_DEP))

        return sequence



    """
    Given an AMREntry representing a location entity, returns
    the sequence of transitions to create the corresponding subgraph
    """
    def _generate_location(self, location_entries, type_location):
        sequence = []
        
        entry_words = [entry for entry in location_entries]
        entry_term = "_".join([e.form for e in entry_words])
        arcs = []
        
        # Generating the subgraph for the 'name' node
        for nw, entry in enumerate(reversed(entry_words)):
            
            w = entry.form
            pos = entry.pos
            
            if pos[0] in ["N", "A", "V", "R"]:
            
                sequence.append((self.algorithm.DESGLOSE, "\"" + w + "\""))
                sequence.append((self.algorithm.SHIFT, None))
                arcs.append("op" + str(len(entry_words) - nw))
            
        sequence.append((self.algorithm.DESGLOSE, self.NAME_DEP))

        for a in reversed(arcs):
            sequence.append((self.algorithm.LEFT_ARC, a))
        sequence.append((self.algorithm.SHIFT, None))
        
        # Generating the wiki    
        sequence.append((self.algorithm.DESGLOSE, "\"" + entry_term + "\""))   
        sequence.append((self.algorithm.SHIFT, None))  
            
        for i in range(nw):
            sequence.append((self.algorithm.REDUCE, None))

        sequence.append((self.algorithm.CONFIRM, type_location))
        sequence.append((self.algorithm.LEFT_ARC, self.WIKI_DEP))
        sequence.append((self.algorithm.LEFT_ARC, self.NAME_DEP))
        return sequence
    
    
    """
    TODO: Postag are currently assumed to match those of 
    """
    def _is_verb(self, postag):
        return postag.startswith("V")
    

    def _is_float(self,w):
        try:
            float(w)
            return True
        except ValueError:
            return False
    
    

    #############################################################################
    #                        VERBALIZATION FUNCTIONS                            #
    #############################################################################

    """
    Loads the verbalization file following the format of: 
    https://github.com/c-amr/camr/blob/master/resources/verbalization-list-v1.01.txt 
    @param A path to the verbalization txt file
    """
    def load_verbalizations(self, path_verbalizations):
         
        DNV = "DO-NOT-VERBALIZE"
        V = "VERBALIZE"
        MV = "MAYBE-VERBALIZE"
         
        verbalizations = {}
        types = []
        with codecs.open(path_verbalizations) as f:
            l = f.readline()
            
            while l != '':   
                #We only consider the reliable verbalizations
                if l.startswith(V) or l.startswith(MV):
                    verbalization = " ".join(l.split()[1:]).split(" TO ")
                    input = verbalization[0]
                    output = verbalization[1].strip("\n")                    
                    verbalizations[input] = output               
                l = f.readline()
                
        return verbalizations  


    """
    Transforms a verbalization of the form: government-organization :ARG0-of govern-01
    into the correspondent sequence of transitions to create the corresponding
    subgraph
    @param verbalization: A string representing a verbalization 
    (e.g. government-organization :ARG0-of govern-01 OR force-01)
    """
    def _transition_sequence_from_verbalization(self, verbalization):
        
        sequence = []      
        verbalization = verbalization.split(" ")
        
        for e in reversed(verbalization):        
            if e.startswith(":"):
                arc_transition = (self.algorithm.LEFT_ARC, e[1:])
            else:      
                if verbalization.index(e) == 0:
                    sequence.append((self.algorithm.CONFIRM, e))
                else:
                    sequence.append((self.algorithm.DESGLOSE, e))
                    sequence.append((self.algorithm.SHIFT, None))      
                          
        if len(verbalization) > 1:
            sequence.append(arc_transition)
                 
        return sequence   