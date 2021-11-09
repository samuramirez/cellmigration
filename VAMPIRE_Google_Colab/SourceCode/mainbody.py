# built-in libraries
import pickle
import ntpath
from datetime import datetime
# external libraries
# my wrapper
from SourceCode.collect_selected_bstack import *
from SourceCode.update_csv import *
# my core
from SourceCode.bdreg import *
from SourceCode.pca_bdreg import *
from SourceCode.clusterSM import *


def mainbody(build_model, csv, entries, outpth=None, clnum=None):
    print('## main.py')
    experimental = True
    realtimedate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    N = int(entries['Number of coordinates'])
    if build_model:
        bstack= collect_seleced_bstack(csv, outpth, build_model, entries)
        vampire_model = {
            "N": [],
            "bdrn": [],
            "bdpc2": [],
            "mdd": [],
            "eigenvalues": [],
            "pc": [],
            "clnum": [],
            "pcnum": [],
            "mincms": [],
            "testmean": [],
            "teststd": [],
            "boxcoxlambda": [],
            "C": [],
            "Z": []
        }
        bdpc, vampire_model = bdreg(bstack[0], N, vampire_model, build_model)
        score, vampire_model = pca_bdreg(bdpc, vampire_model, build_model)
        pcnum = None # none is 20 by default
        IDX, IDX_dist, vampire_model, _ = clusterSM(outpth, score, bdpc, clnum, pcnum, vampire_model, build_model, None, None, entries)
        modelname = entries['Model name']
        if os.path.exists(os.path.join(*[outpth, modelname, modelname+'.pickle'])):
            f = open(os.path.join(*[outpth, modelname, modelname+'_'+realtimedate+'.pickle']), 'wb')
        else:
            f = open(os.path.join(*[outpth, modelname, modelname+'.pickle']), 'wb')
        pickle.dump(vampire_model, f)
        f.close()

    else:
        UI = pd.read_csv(csv)
        setpaths = UI['set location']
        tag = UI['tag']
        condition = UI['condition']
        setID = UI['set ID'].astype('str')
        for setidx, setpath in enumerate(setpaths):
            pickles = [_ for _ in os.listdir(outpth) if _.lower().endswith('pickle')]
            bdstack = [pd.read_pickle(os.path.join(outpth, pkl)) for pkl in pickles if condition[setidx] in pkl]
            bdstacks = pd.concat(bdstack, ignore_index=True)

            f = open(entries['Model to apply'], 'rb')

            vampire_model = pickle.load(f)
            N = vampire_model['N']
            bdpc, vampire_model = bdreg(bdstacks[0], N, vampire_model, build_model)
            score, vampire_model = pca_bdreg(bdpc, vampire_model, build_model)
            clnum = vampire_model['clnum']
            pcnum = vampire_model['pcnum']

            if experimental:
                IDX, IDX_dist, vampire_model, goodness = clusterSM(outpth, score, bdpc, clnum, pcnum, vampire_model,
                                                                   build_model, condition[setidx], setID[setidx],
                                                                   entries)
                update_csv(IDX, IDX_dist, tag[setidx], condition[setidx], setpath, outpth, goodness=goodness)
            else:
                IDX, IDX_dist, vampire_model, _ = clusterSM(outpth, score, bdpc, clnum, pcnum, vampire_model,
                                                                   build_model, condition[setidx], setID[setidx],
                                                                   entries)
                update_csv(IDX, IDX_dist, tag[setidx], condition[setidx], setpath, outpth)


