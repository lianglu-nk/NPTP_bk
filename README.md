
## Instruction

### 1) Install prerequriement and create a python virtual environment

```bash
./install.sh
conda env create -f environment.yml -n tfgpu
conda activate tfgpu
```


### 2) Data preparation and Clustering (via KNIME):

- Open KNIME in current directory and select the knime-workspace folder as the working directory;
- Import data_process.knwf in KNIME and run the nodes step-by-step;


### 3) Features generation for Deep learning:

generate input features for FNN or LSTM##

```bash
python ./python_code/FNN_LSTM_Feature.py
python ./python_code/LSTM_Feature_supplement.py
python ./python_code/GNN_Feature.py
```

### 4) Training the FNN model:

```bash
python ./python_code/fnn/estGPUSize.py
python ./python_code/fnn/step1.py
python ./python_code/fnn/step2.py
python ./python_code/model/fnn_model/step_model.py
```

### 5)Training the GC model:

```bash
python ./python_code/gc/estGPUSize.py
python ./python_code/gc/step1.py
python ./python_code/gc/step2.py
python ./python_code/model/gc_model/step_model.py
```

### 6) Training the Weave model:

```bash
python ./python_code/weave/estGPUSize.py
python ./python_code/weave/step1.py
python ./python_code/weave/step2.py
python ./python_code/model/weave_model/step_model.py
```

### 7) Traning the LSTM model:

```bash
python ./python_code/lstm/step2.py
python ./python_code/model/lstm_model/step_model.py
```

### 8) Training other ML models:

```bash
python ./python_code/ml/SVM.py
python ./python_code/ml/SVM_2.py
python ./python_code/ml/SVM_3.py

python ./python_code/ml/XGboost.py
python ./python_code/ml/XGboost_2.py
python ./python_code/ml/XGboost_3.py

python ./python_code/ml/RF.py
python ./python_code/ml/RF_2.py
python ./python_code/ml/RF_3.py

python ./python_code/ml/KNN.py
python ./python_code/ml/KNN_2.py
python ./python_code/ml/KNN_3.py

python ./python_code/ml/NB_2.py
python ./python_code/ml/NB_3.py
```

### 9) Result and charting commands:
- Draw ROC plot
  ```bash
  python ./python_code/result_scrip/dnn_result_out_step1.py
  python ./python_code/result_scrip/dnn_result_out_step2.py
  python ./python_code/result_scrip/ml_result_out.py
  ```
- Draw Box plot
  ```bash
  python ./python_code/result_scrip/box_figure.py
  ```

## Script Usages: 
The custom options of previous scripts are listed below:

```bash
Usage: readdata_cl.py [options]
Options:
  -h, --help            show this help message and exit.
  -destPath             location where the final clustering file is generated;
                        output files:folds0.pckl, labelsHard.pckl.
  -infilna1             out_put_folder_name(same as above step1-6).
  -infilna2             interactions_file_name(same as above step6).
  -tarnum               the number of corresponding targets.
```
```bash
Usage: FNN_LSTM_Feature.py; LSTM_Feature_supplement.py; GNN_Feature.py; [options]
Options:
  -h, --help            show this help message and exit.
  -supplfile            sdf file path.
  -lstmfeature          default=False;
                        change to True when genrating features for lstm.
  -featureoutfile       the output path of the pckl file.
  -featureoutname       pckl file name.
```

```bash
Usage: FNN:estGPUSize.py, step1.py, step2.py, step3.py, step_model.py [options]
Options:
  -h, --help            show this help message and exit.
  -maxProc              Max. Nr. of Processes(default=10)
  -availableGPU         GPU ID for Test.
  -sizeFact             Size Factor GPU Scheduling(default=1.0).
  -originalData         the output path of the pckl file(featureoutfile).
  -featureoutname       pckl file name(featureoutfile).
  -datasetNames         finger pattern(options:ecfp6,fcfp6,MACCS,ecfp6maccs,fcfp6maccs,ecfp6fcfp6,ecfp6fcfp6MACCS).
  -saveBasePath         output model folder.
  -ofolds               Outer Folds(default=[0, 1, 2]).
  -ifolds               inner Folds(default=[0, 1, 2]).
  -pStart               Parameter Start Index.
  -pEnd                 Parameter End Index.
  -continueComputations continueComputations.
  -saveComputations     saveComputations.
  -startMark            startMark.
  -finMark              finMark.
  -epochs               Epochs number(default=300).
  -modelname            model file name.
  -dataset              the finger pattern for the training model(default="ecfp6fcfp6MACCS").
```

```bash
Usage: GC:estGPUSize.py, step1.py, step2.py, step3.py, step_model.py [options]
Options:
  -datasetNames         graph convolution type(default="graphConv").
  -dataset              graph convolution type(default="graphConv").
  the others are the same as those in FNN.
```
```bash
Usage: Weave: estGPUSize.py, step1.py, step2.py, step3.py,step_model.py [options]
Options:
  -datasetNames         graph convolution type(default="graphWeave").
  -dataset              graph convolution type(default="graphWeave").
  the others are the same as those in FNN.

Usage: LSTM:step2.py, step3.py, step_model.py [options]
Options:
  -dataset              lstm output name(default="lstm_ecfp6fcfp6MACCS").
  the others are the same as those in FNN.

```

```bash
Usage: ML:SVM.py, SVM_2.py,SVM_3.py  [options]
Options:
  -cl_file              the folder where the clustering files for each target are located.
  -pertarget_file       the folder where SMILES files for each targets are located.
  -datasetNames         the finger pattern(default="ecfp6fcfp6MACCS").
  -saveBasePath         output model folder.
  -ofolds               Outer Folds(default=[0, 1, 2]).
  -ifolds               inner Folds(default=[0, 1, 2]).
  -pStart               Parameter Start Index.
  -pEnd                 Parameter End Index.
```

```bash
Usage: dnn_result_out_step1.py, dnn_result_out_step2.py [options]
Options:
  -saveBasePath         output model folder(res file path).
  -dataPathSave         original file path.
  -writeDir             write file path.
  -datasetNames         methods_out_put_folder_name(options:ecfp6,fcfp6,MACCS,ecfp6maccs,fcfp6maccs,ecfp6fcfp6,
                        ecfp6fcfp6MACCS,graphConv,graphWeave,lstm_ecfp6fcfp6MACCS).
```

```bash						
Usage: ml_result_out.py [options]
Options:
  -datasetNames         finger pattern(options:ecfp6,fcfp6,MACCS,ecfp6maccs,fcfp6maccs,ecfp6fcfp6,ecfp6fcfp6MACCS).
  -methodNames          ML method Name(options:SVM,RF,KNN,XGboost,NB).
  the others are the same as those in deep result.
```

```bash
Usage: box_figure.py [options]
Options:
  -deepDir              deep result write file path.
  -dataPathSave         ml result write file path.
  -writeDir             chart write file path.
```