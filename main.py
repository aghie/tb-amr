
import os
#Uncomment/Comment these lines to determine whether and which GPU(s) to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from argparse import ArgumentParser
import keras
import ConfigParser
import mlp
import utils
import pickle
import yaml
import codecs


"""
This is the script used to train and evaluate and AMR-Covington parser
"""

if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", dest="input", help="Path to the input file",default=None)
#    arg_parser.add_argument("--input_type", dest="graphs", help="[*.graphs, *raw]", default="raw")
    arg_parser.add_argument("--train", dest="amr_train", help="Annotated AMR train file", metavar="FILE", default="../data/PTB_SD_3_3_0/train.conll")
    arg_parser.add_argument("--dev", dest="amr_dev", help="Annotated AMR dev file", metavar="FILE", default="../data/PTB_SD_3_3_0/dev.conll")
    arg_parser.add_argument("--test", dest="amr_test", help="Annotated AMR test file", metavar="FILE", default="../data/PTB_SD_3_3_0/test.conll")
    arg_parser.add_argument("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    arg_parser.add_argument("--params_model", dest="params_model", help="Parameters file", metavar="FILE", default="params_model.pickle")
#    arg_parser.add_argument("--extrn_pos", dest="pos_external_embedding", help= "PoStag external embeddings", metavar="FILE")
#    arg_parser.add_argument("--wembedding", type=int, dest="wembedding_dims", default=0)
    arg_parser.add_argument("--epochs", type=int, dest="epochs", default=30)
    arg_parser.add_argument("--outdir", type=str, dest="output", default="results")
    arg_parser.add_argument("--predict", action="store_true", dest="predictFlag", default=False)
    arg_parser.add_argument("--expanded_rels", action="store_true", dest="expanded_rels", default=False)     #TODO: Not properly tested when set to true
    arg_parser.add_argument("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE", default=None)
    arg_parser.add_argument("--verbalize", type=str, dest="verbalize", default=None)
    arg_parser.add_argument("--cities", dest="cities", default=None)
    arg_parser.add_argument("--states", dest="states", default=None)
    arg_parser.add_argument("--countries", dest="countries", default=None)
    arg_parser.add_argument("--negations", dest="negations", default=None)
    arg_parser.add_argument("--nationalities", dest="nationalities", default=None)
    arg_parser.add_argument("--nationalities2", dest="nationalities2", default=None)
    arg_parser.add_argument("--arg_rules",dest="arg_rules", default=None)
    args = arg_parser.parse_args()
#     arg_parser.add_argument("--verbalize", type=str, dest="verbalize", default="/data/AMR/verbalization-list-v1.06.txt")
#     arg_parser.add_argument("--cities", dest="cities", default="/home/david.vilares/git/amr-eager/resources/cities.txt")
#     arg_parser.add_argument("--states", dest="states", default="/home/david.vilares/git/amr-eager/resources/states.txt")
#     arg_parser.add_argument("--countries", dest="countries", default="/home/david.vilares/git/amr-eager/resources/countries.txt")
#     arg_parser.add_argument("--negations", dest="negations", default="/home/david.vilares/git/amr-eager/resources/negations.txt")
#     arg_parser.add_argument("--nationalities", dest="nationalities", default="/home/david.vilares/git/amr-eager/resources/nationalities.txt")
#     arg_parser.add_argument("--nationalities2", dest="nationalities2", default="/home/david.vilares/git/amr-eager/resources/nationalities2.txt")
#     arg_parser.add_argument("--arg_rules",dest="arg_rules", default="/home/david.vilares/git/amr-eager/resources/args_rules.txt")

    
    #TODO: Not sure this is the best way to do this
    #Adding additional arguments from the configuration file
    #(mainly text resources)
    config = ConfigParser.ConfigParser()
    config.readfp(open("./configuration.conf"))
    if args.external_embedding is None: args.external_embedding = config.get("Resources","path_embeddings")
    if args.verbalize is None: args.verbalize = config.get("Resources","path_verbalization_list")
    if args.cities is None: args.cities = config.get("Resources","path_cities")
    if args.states is None: args.states = config.get("Resources","path_states")
    if args.countries is None: args.countries = config.get("Resources","path_countries")
    if args.negations is None: args.negations = config.get("Resources","path_negations")
    if args.nationalities is None: args.nationalities = config.get("Resources","path_nationalities")
    if args.nationalities2 is None: args.nationalities2 = config.get("Resources","path_nationalities2")
    if args.arg_rules is None: args.arg_rules = config.get("Resources","path_arg_rules") 


    if not args.predictFlag:
        #########################################################################
        #                          TRAINING PHASE                               #
        #########################################################################
        print "Training..."

        if not os.path.exists(args.output):
            os.mkdir(args.output)
    
    
        #########################################################################
        #        Loading the training and development resources                 #
        #########################################################################
        f_aligned_AMR =  args.amr_train+".aligned"
        path_amrs = args.amr_train+".graphs"
        path_amr_templates = args.amr_train+".word_subgraphs"        
        path_multiword_templates = args.amr_train+".phrase_subgraphs"
        
        f_aligned_dev_AMR =args.amr_dev+".aligned"        
        path_amrs_dev = args.amr_dev+".graphs"
        
        with codecs.open(path_amrs,'rb') as f:
            amr_graphs = pickle.load(f)

        with codecs.open(path_amrs_dev,'rb') as f:
            dev_amr_graphs = pickle.load(f)
            
        with codecs.open(path_amr_templates,'rb') as ft:
            amr_graph_templates = pickle.load(ft)
        
        with codecs.open(path_multiword_templates,'rb') as ft:
            amr_multiword_graph_templates = pickle.load(ft)          

        
        words,lemmas, pos,rels, nodes, entities,deps = utils.vocab(amr_graphs)
    
        _, _, _, dev_rels, dev_nodes,_, _ =  utils.vocab(dev_amr_graphs)
                        
        with open(os.path.join(args.output, args.params), 'wb') as paramsfp:
            pickle.dump((words, lemmas, pos, rels, nodes, entities,deps,args), paramsfp)

        parser = mlp.PerceptronAMR(words,pos,rels,nodes,entities,deps,args.external_embedding, 
                                               None
                                               #args.pos_external_embedding
                                               ,None,None, None, 
                                               amr_graph_templates,
                                               amr_multiword_graph_templates,
                                               None,
                                               args)
    
        parser.train(path_amrs, path_amrs_dev)
        
    else:
        #########################################################################
        #                              TEST PHASE                               #
        #########################################################################
        #Information used from the training set
#        path_graphs = args.amr_train+".graphs"
        path_templates = config.get("Resources","path_base_templates")+".word_subgraphs" #args.amr_train+".word_subgraphs"
        path_multiword_templates = config.get("Resources","path_base_templates")+".phrase_subgraphs" #args.amr_train+".phrase_subgraphs"
        path_lookup_concepts = config.get("Resources","path_base_templates")+".word_concepts" #args.amr_train+".word_concepts"
        
#         with codecs.open(path_graphs,'rb') as f:
#             amr_graphs = pickle.load(f)        
        with codecs.open(path_templates,'rb') as ft:
            amr_graph_templates = pickle.load(ft)          
        with codecs.open(path_multiword_templates,'rb') as ft:
            amr_multiword_graph_templates = pickle.load(ft)          
                
        #########################################################################
        #                       Loading the trained model                       #
        #########################################################################
        with open(os.path.join(args.output, args.params), 'rb') as paramsfp:
            words, lemmas, pos, rels, nodes, entities,deps, options = pickle.load(paramsfp) 

        parser = mlp.PerceptronAMR(words,pos,rels,nodes,entities,
                                               deps,args.external_embedding, 
                                               None,
                                               #args.pos_external_embedding,
                                               None,None, None, 
                                               amr_graph_templates,
                                               amr_multiword_graph_templates,
                                               path_lookup_concepts
                                               ,args)
        
        #Path to obtain the weights of the model
        save_model_path = args.params_model
        components = save_model_path.rsplit("/",1)
        Tpath = save_model_path.replace(components[-1],"T."+components[-1])
        Rpath = save_model_path.replace(components[-1],"R."+components[-1])
        Cpath = save_model_path.replace(components[-1],"C."+components[-1])
          
        parser.transition_model = keras.models.load_model(Tpath)
        parser.relation_model = keras.models.load_model(Rpath)
        parser.concepts_model = keras.models.load_model(Cpath)
    
        #########################################################################
        #     Loading the test RAW! files, already containing the AMRWORDs      #
        #########################################################################
        path_amrs_raw = args.amr_test+".input"

        #########################################################################
        #                        Evaluating the model                           #
        #########################################################################
        pred_graphs = parser.predict(path_amrs_raw)
        
        #It writes the AMR graphs into a file (stored in *output)
        #and makes a call to Damonte et al (2017) evaluation script (results stores in *.metrics)
        path_test_output = os.path.join(args.output, 'amr-covington'+'.output')
        with codecs.open(path_test_output,"w", encoding="utf-8") as f:
             
            for g in pred_graphs:
                g_raw =  g.print_graph()
                f.write(g.id+"\n"+g_raw.decode("utf-8")+"\n\n")
     
        config = ConfigParser.ConfigParser()
        config.readfp(open("./configuration.conf"))
        amr_eager_evaluation = config.get("Resources","path_evaluation_script") 
        os.chdir(amr_eager_evaluation)
        os.system("./evaluation.sh "+path_test_output+" "+args.amr_test+" > "+path_test_output+".metrics")
             
