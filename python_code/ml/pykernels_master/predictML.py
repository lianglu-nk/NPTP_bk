import pandas as pd
import itertools
import numpy as np
import pickle
import os
import multiprocessing as mp
import os
import warnings
from random import shuffle
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import joblib




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
    return fingerprints

def get_file_list(file_folder):
  # method one: file_list = os.listdir(file_folder)
  for root, dirs, file_list in os.walk(file_folder):
    return dirs,file_list
        
filepath = '/data/MLlearning/noweakchembl26end/ML/ECFP6FCFP6MACCS/SVM/'
files_list = get_file_list(filepath)[0]
num = len(files_list)

pred_data = pd.read_csv('/data/liuye3/one_mol/two_mol/CID_46343421_CID_135939450.csv')
mol_features = batchECFP(pred_data.canonical_SMILES)


for i in range(1):
  datapath= filepath + files_list[i]
  tar_id= files_list[i]
  target = datapath+'/'+tar_id+'_model_'+'svm'+'.pckl'
  print(target)
  tar_model = joblib.load(target)
  y_pred = tar_model.predict(mol_features)
  y_pred_proba = tar_model.predict_proba(mol_features)

  print(len(y_pred))
  dataout = pd.DataFrame(columns=[tar_id], data=y_pred)

  
for i in range(1,num):
  datapath= filepath + files_list[i]
  tar_id= files_list[i]
  target = datapath+'/'+tar_id+'_model_'+'svm'+'.pckl'
  print(target)
  tar_model = joblib.load(target)
  y_pred = tar_model.predict(mol_features)
  y_pred_proba = tar_model.predict_proba(mol_features)
  
  print(len(y_pred))
  dataout[tar_id] = y_pred


print(dataout)
print(dataout.shape)
dataout.to_csv('/data/liuye3/one_mol/two_mol/result/SVM899.csv', index=True, header=True)
