# MEnKF_ANN_Chembl_Multivariate_Simulations
Simulations to assess the performance of the Matrix Ensemble Kalman Filter Approach in the case of multivariate response

# MEnKF_ChemBL_Multivariate
Matrix Ensemble Kalman Filter for Multivariate Target Prediction using the Chembl database

## Data Preparation

To create the feature files to run the Matrix Ensemble Kalman Filter Method please follow the following steps. On following steps 1 through 6 you will get many interim files which will be required to create the final feature file 

1. Download the datasets [No_Filters.csv](https://drive.google.com/drive/folders/1clnJGyRuriZFKXiN_ctzws03Cp3-qlgK) , [Small_Molecule_Phase_3.csv](https://drive.google.com/file/d/1NMzBgvLj1m2RqGZaRMkeeqZ-pDkqQhxe/view?usp=drive_link) and place in the [Data](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Data) folder.

2. Run the script [Combine_Small_and_Big_Datasets.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Data_Preparation/Combine_Small_and_Big_Datasets.ipynb) that will create the file `combined_data.csv` in the [Data](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Data) folder.

3. Run the script [Make_Rdkit_Features.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Data_Preparation/Make_Rdkit_Features.ipynb) which will map the Rdkit features to the Smiles in the `combined_data.csv` dataset. It will then create a file called `combined_data_with_rdkit.csv` in the [Data](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Data) folder.

4. Run the script [Train_Test_Splits.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Data_Preparation/Train_Test_Splits.ipynb) which will create the files `x_train.csv`, `x_valid.csv`, `y_train.csv`, `y_valid.csv`, `smiles_with_rdkit_with_small_phase_3_features.csv`, and `smiles_with_rdkit_with_small_phase_3_outputs.csv` in the [Data](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Data) folder. In this script, the rdkit features are standardized to have a column mean and standard deviation of 0 and 1, respectively. The standardizer object will be placed in the [Data](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Data) folder by the name of `std_scaler.pkl`. 

5. Train the multi-arm base model using the script [PSA_AlogP_Base_Model.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Base_Model_Training/PSA_AlogP_Base_Model.ipynb). The stored model `Model_BOTH` will be placed in the [Base_Models](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Base_Models) folder. The multivariate targets are standardized to have column means and standard deviation of 0 and 1, respectively. The standardizer object can be found in the [Data](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Data) folder by the name of `target_scaler.pkl`.

6. Use the trained multi-arm base model from 5 to extract the smiles and rdkit embeddings for the samples in the `smiles_with_rdkit_with_small_phase_3_features.csv` dataset using the script [Extract_BottleNeck_Features_Both.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Data_Preparation/Extract_BottleNeck_Features_Both.ipynb). The extracted embeddings will be in a file called `small_mol_phase_3_features_for_both.npy` in the [Data](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/tree/main/Data) folder.

## Running the MEnKF Simulations

1. Generate the data for the simulations under three different settings:
   
   i. Smiles Output having a weight of 70% and the Rdkit Outputs having a weight of 30%. The covariance matrix for the target has 0.3 along the diagonal and 0.06 for the covariance between the two targets. [Generate_Simulation_Data70_30_with_cov_positive_0.06_var.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Simulation_Data_Generation/Generate_Simulation_Data70_30_with_cov_positive_0.06_var.ipynb)

   ii. Smiles Output having a weight of 70% and the Rdkit Outputs having a weight of 30%. The covariance matrix for the target has 0.3 along the diagonal and -0.27 for the covariance between the two targets. [Generate_Simulation_Data_70_30_with_cov_minus_0.27_var.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Simulation_Data_Generation/Generate_Simulation_Data_70_30_with_cov_minus_0.27_var.ipynb)

   iii. Smiles Output having a weight of 80% and the Rdkit Outputs having a weight of 20%. The covariance matrix for the target has 0.2 along the diagonal and -0.18 for the covariance between the two targets. [https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Simulation_Data_Generation/Generate_Simulation_Data_80_20_cov_minus_0.2_var.ipynb](https://github.com/Ved-Piyush/MEnKF_ANN_Chembl_Multivariate_Simulations/blob/main/Simulation_Data_Generation/Generate_Simulation_Data_80_20_cov_minus_0.2_var.ipynb)

