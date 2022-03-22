import multiprocessing as mp
import os
import warnings
from random import shuffle
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles,AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import DataStructs
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, silhouette_score, average_precision_score
import datetime
import argparse
import pandas as pd
import itertools
import numpy as np
import pickle
warnings.filterwarnings("ignore")
dictionary0 = {
    'n_neighbors':[1,3,5,7,9,11]
}
basePath=os.getcwd()
hyperParams0 = pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys())
hyperParams0.index=np.arange(len(hyperParams0.index.values))

parser = argparse.ArgumentParser()
parser.add_argument("-cl_file", help="cl per target",type=str, default=basePath+'/test_data/cl/')
parser.add_argument("-pertarget_file", help="smi per target", type=str, default=basePath+'/test_data/pertargetdata/')
parser.add_argument("-datasetNames", help="Dataset Name",type=str, default="ecfp6fcfp6MACCS")
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+'/res_test_data/')
parser.add_argument("-ofolds", help="Outer Folds", nargs='+', type=int, default=[0, 1, 2])
parser.add_argument("-ifolds", help="Inner Folds", nargs='+', type=int, default=[0, 1, 2])
parser.add_argument("-pStart", help="Parameter Start Index", type=int, default=0)
parser.add_argument("-pEnd", help="Parameter End Index", type=int, default=6)
args = parser.parse_args()
cl_file = args.cl_file
pertarget_file = args.pertarget_file
datasetNames = args.datasetNames
saveBasePath = args.saveBasePath
compOuterFolds = args.ofolds
compInnerFolds = args.ifolds
paramStart = args.pStart
paramEnd = args.pEnd
compParams = list(range(paramStart, paramEnd))


def bestSettingsSimple(perfFiles, nrParams):
    aucFold=[]
    for foldInd in range(0, len(perfFiles)):
        innerFold=-1
        aucParam=[]
        for paramNr in range(0, nrParams):
            #try:
            saveFile=open(perfFiles[foldInd][paramNr], "rb")
            aucRun=pickle.load(saveFile)
            saveFile.close()
            #except:
            #  pass
            if(len(aucRun)>0):
                aucParam.append(aucRun[-1])
    
        aucParam=np.array(aucParam)
    
        if(len(aucParam)>0):
            aucFold.append(aucParam)
    aucFold=np.array(aucFold)
    aucMean=np.nanmean(aucFold, axis=0)
    paramInd=np.nanmean(aucMean, axis=1).argmax()
  
    return (paramInd, aucMean, aucFold)

def ClusterCV(csv_file):
    tar_id = csv_file.split('.')[0]
    file_name = pertarget_file + csv_file
    clusterSampleFilename = os.path.join(cl_file, 'cl' + tar_id + ".info")
    chembl_data = file_reader(file_name)
    target_name = chembl_data.iloc[0,0]
    labels = chembl_data.active_label
    features = batchECFP(chembl_data.canonical_smiles)
    clusterTab = pd.read_csv(clusterSampleFilename, header=None, index_col=False, sep=",")
    df = clusterTab.values
    folds = df[:, 0]
    return folds, features, labels, target_name


def batchECFP(smiles, radius=3, nBits=2048):
    smiles = np.array(smiles)
    n = len(smiles)
    fingerprints_0 = np.zeros((n, nBits), dtype=int)
    fingerprints_1 = np.zeros((n, nBits), dtype=int)
    MACCSArray = []
    for i in range(n):
        mol = MolFromSmiles(smiles[i])
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fp_1 = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=True)
        MACCSArray.append(MACCSkeys.GenMACCSKeys(mol))
        fingerprints_0[i] = np.array(list(fp.ToBitString()))
        fingerprints_1[i] = np.array(list(fp_1.ToBitString()))
    fingerprints_2 = np.array(MACCSArray)
    fingerprints = np.hstack((fingerprints_0,fingerprints_1,fingerprints_2))
    fingerprints_3=np.hstack((fingerprints_0,fingerprints_1))
    fingerprints_4=np.hstack((fingerprints_0,fingerprints_2))
    fingerprints_5=np.hstack((fingerprints_1,fingerprints_2))
    if datasetNames=="ecfp6fcfp6MACCS":
        fingerprints_out=fingerprints
    elif datasetNames=="ecfp6":
        fingerprints_out=fingerprints_0
    elif datasetNames=="fcfp6":
        fingerprints_out=fingerprints_1
    elif datasetNames=="MACCS":
        fingerprints_out=fingerprints_2
    elif datasetNames=="ecfp6fcfp6":
        fingerprints_out=fingerprints_3
    elif datasetNames=="ecfp6MACCS":
        fingerprints_out=fingerprints_4
    elif datasetNames=="fcfp6MACCS":
        fingerprints_out=fingerprints_5
    
    return fingerprints_out


def get_file_list(file_folder):
    # method one: file_list = os.listdir(file_folder)
    for root, dirs, file_list in os.walk(file_folder):
        return file_list


def file_reader(file_path):
    data = pd.read_csv(file_path)
    return data

def data_split_modeling(target_file):
    # validation data construction
    target_id = target_file.split('.')[0]

    cluster_res = ClusterCV(csv_file=target_file)
    folds = cluster_res[0]
    features = cluster_res[1]
    active_label = cluster_res[2]
    target_name = cluster_res[3]
    save_path = saveBasePath+"/ML/"+datasetNames+"/KNN/" + target_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dbgPath=save_path+"dbg/"
    if not os.path.exists(dbgPath):
        os.makedirs(dbgPath)
    # modeling
    
    


    for outerFold in compOuterFolds:
        savePrefix0="o"+'{0:04d}'.format(outerFold+1)
        savePrefix=save_path+savePrefix0
        dbgOutput=open(dbgPath+savePrefix0+".dbg", "w")
        perfFiles=[]
        for innerFold in compInnerFolds:
            if innerFold == outerFold:
                continue
            perfFiles.append([])   
                 
            for paramNr in range(0, hyperParams0.shape[0]):
                perfFiles[-1].append(save_path+"o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(innerFold+1)+"_p"+'{0:04d}'.format(hyperParams0.index.values[paramNr])+".test.auc.pckl")
        paramNr, perfTable, perfTableOrig=bestSettingsSimple(perfFiles, hyperParams0.shape[0])
        print(hyperParams0.iloc[paramNr], file=dbgOutput)
        print(perfTable, file=dbgOutput)
        dbgOutput.close()
        
        roc_auc = []
        n_neighbors1 = hyperParams0.iloc[paramNr].n_neighbors
        knn = KNeighborsClassifier(n_neighbors=n_neighbors1)
        print('knn')

        # Get training and testing set
        test_index = np.where(folds == outerFold)[0]
        train_index = np.where(folds != outerFold)[0]
        #print(train_index)
        # Get train and test samples
        X_test = features[test_index, :]
        X_train = features[train_index, :]
        y_test = active_label.iloc[test_index]
        y_train = active_label.iloc[train_index]
        # Modeling and Evaluate
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # compute roc-auc
        y_pred_proba = knn.predict_proba(X_test)
        roc_auc.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
        a = np.array(roc_auc)
        
        
        reportTestAUC = []
        reportTestAUC.append(a)
        
        saveFilename = savePrefix + ".test.auc.pckl"
        saveFile = open(saveFilename, "wb")
        pickle.dump(reportTestAUC, saveFile)
        saveFile.close()

startime = datetime.datetime.now()
print("Start time", startime)
folder_path = pertarget_file
files_list = get_file_list(folder_path)
p = mp.Pool(processes=1)
for tar_file in files_list:
    result = p.apply_async(data_split_modeling, args=(tar_file,))  # distribute one task to one pool
p.close()  # finished load task
p.join()  # start
print("Sub-process(es) done.")
endtime = datetime.datetime.now()
print("End time", endtime)
costime = endtime - startime
print("Cost time", costime)
#data_split_modeling(files_list[0])
