The python scripts in this folder are used in the four bash scripts showing a exemplary pipeline of data preparation, model training and benchmarking.

# bash scripts

The scripts below must either be called from the activated conda environment with the pkasolver package and its dependencies installed, or the conda path must be set manually inside each script.

The variable `data_path` must be set to your working directory.


1. `script_data_prep_chembl.sh` --contains the pipeline to download molecular data from the CHEMBL-database, predict its pka-data with Schrödinger's Epik and finally convert it the pytorch geometric graph data, to be used in the initial training.

2. `script_data_prep_chembl.sh` --instructions for creating pyg graph data from molecules with known pka 

3. `script_prediction_testset.sh` --benchmarks a list of models on the two test sets, Novartis and Literature.

4. `script_training.sh` --script for training new model with an initial training dataset consisting of pkas predicted by a different program, Epik, and experimental pka data

# python scripts

`00_download_mols_from_chembl.py`:
--input: None 
--output: path to output file (sdf.gz, sdf) 

--filters the molecules of the chembl database by the specified criteria (e.g. max number of rule of five violation = 1) and outputs them to a gzipped sdf file.

`01_convert_sdf_to_mae.py` 
--input: path to input file (sdf.gz, sdf)
--output: path to output file (mae.gz, mae)

--takes sdf file (can be gzipped) and converts it to Schrödinger maestro (mae) file. Schrödinger "ligprep" CLI-binary must be installed and path must be specified inside the script.  

`02_predict_pka_with_epik.py` 
--input: path to input file (mae.gz, mae)
--output: path to output file (mae.gz, mae)

--takes molecules from Schrödinger maestro (mae) file and returns new mae file containing Epik pka prediction data for each molecule. Schrödinger "Epik" CLI-binary must be installed and path must be specified inside the script.

`03_convert_mae_to_sdf.py` 
--input: path to input file (mae.gz, mae)
--output: path to output file (sdf.gz, sdf)

--takes Schröndiger maestro (mae) file (can be gzipped) and converts it to sdf file. Schrödinger "convert" CLI-binary must be installed and path must be specified inside the script.



`04_0_filter_testmols.py` 
--input: path to input file (mae.gz, mae)
--output: path to output file (sdf.gz, sdf)
--filter: path to output file (sdf, sdf.gz)
 

--takes sdf file of initial training molecules and sdf file of training molecules (both optionally gzipped) and returns only those initial training molecules not contained in the training molecules file as sdf file. 

`04_1_split_epik_output.py` 
--input: path to input file (sdf.gz, sdf)
--output: path to output file (pkl)

--takes sdf file with molecules containing Epik pka predictions in their properties and outputs a new sdf where those molecules containing more than one pka get duplicated so that every molecules only contains one pka value. The molecule associated with each pka is the protonated form of the respective pka reaction

`04_2_prepare_rest.py` 
--input: path to input file (sdf.gz, sdf)
--output: path to output file (pkl)

--takes sdf of molecule set containing pka data and returns it as a pkl file.

`05_data_preprocess.py` 
--input: path to input file (pkl)
--output: path to output file (pkl)

--takes pkl file of molecules containing pka data and returns pytorch geometric graph data containing protonated and deprotonated graphs for every pka

`06_training.py` 
--input: set of training molecules as pyg graphs (pkl)
--output: path to output file (pkl)
--model_name: name of model architecture used (string)

Optional parameters:
--model: path for saving model or containing model for retraining (pkl)
--val: set of validation molecules as pyg graphs (pkl)
--epochs: set number of training epochs (default == 1000)
--reg: ?
-r: flag for retraining model at path given by --model

--takes training set as pkl file and trains new model or retrains existing one. 

`07_test_set_performance.py` 
--model: model path(s), (pkl) 
--testset: path to testset file (pkl)
--name: filename for reg-plot (png)

--takes the path to models, a test set as pyg graphs (pkl) and the name for the output file and returns regression plot containing the mean prediction of the models for each pka, including error bars. The figures additionally contain the MAE, RMSE and R2 with confidence interval.