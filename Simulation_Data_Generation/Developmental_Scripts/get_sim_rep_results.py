import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
import random
import pickle
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf
from tqdm.notebook import tqdm
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')

def get_targets_with_weights(batch_data, initial_ensembles, size_ens): 
    
    target_dim = 1
    
    # weights_ann_1 = ann.get_weights()
    
    # h1  = ann.layers[1].output.shape[-1]

    n_hidden_1 = len(weights_ann_1[0].ravel())
    
    hidden_weights_1 = initial_ensembles[:,:n_hidden_1].reshape( size_ens, batch_data.shape[1], h1)
    
    
    hidden_output_1 = np.einsum('ij,kjl->kil', batch_data, hidden_weights_1)

    
    hidden_layer_bias_1 = initial_ensembles[:,n_hidden_1:(n_hidden_1 + h1)].reshape(size_ens, 1,  h1)


    hidden_output_1 = hidden_output_1 + hidden_layer_bias_1

    n_pred_weights_1 = len(weights_ann_1[2].ravel())

    output_weights_1 = initial_ensembles[:,(n_hidden_1 + h1):(n_hidden_1 + h1 + n_pred_weights_1) ].reshape(size_ens, h1, target_dim)


    output_1 = np.einsum('ijk,ikl->ijl', hidden_output_1, output_weights_1)


    output_layer_bias_1 = initial_ensembles[:,(n_hidden_1 + h1 + n_pred_weights_1):(n_hidden_1 + h1 + n_pred_weights_1 + target_dim)].reshape(size_ens, 1, target_dim)


    final_output_1 = output_1 + output_layer_bias_1
    
    final_output_1 = final_output_1[:,:, 0]
    
    # print(final_output_1.shape, initial_ensembles.shape)
    
    stack = np.hstack((final_output_1, initial_ensembles))

    
    return final_output_1, stack

def ann(hidden = 32, input_shape = 256, output_shape = 1): 
    input_layer = tf.keras.layers.Input(shape = (input_shape))
    hidden_layer = tf.keras.layers.Dense(hidden)
    hidden_output = hidden_layer(input_layer)
    pred_layer = tf.keras.layers.Dense(output_shape, activation = "relu")
    pred_output = pred_layer(hidden_output)
#     pred_output = tf.keras.layers.Activation("softmax")(pred_output)
    model = tf.keras.models.Model(input_layer, pred_output)
    return model

def generate_initial_ensembles(num_weights, lambda1, size_ens):
    mean_vec = np.zeros((num_weights,))
    cov_matrix = lambda1*np.identity(num_weights)
    mvn_samp = mvn(mean_vec, cov_matrix)
    return mvn_samp.rvs(size_ens)

def expit(x):
    """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
    return 1 / (1 + np.exp(-x))

samp_ann =  ann(hidden = 16, input_shape = 32, output_shape = 1)\

weights_ann_1 = samp_ann.get_weights()

h1  = samp_ann.layers[1].output.shape[-1]

hidden_neurons = h1

samp_ann_params = samp_ann.count_params()

def get_initial_X_t(data1, data2, size_ens, var_weights = 1.3):
    # samp_ann =  ann(hidden = hidden_neurons, input_shape = 32, output_shape = 1)
    
    initial_ensembles1 = generate_initial_ensembles(samp_ann_params, var_weights, size_ens)
    data1_out1, data1_stack1 = get_targets_with_weights(data1, initial_ensembles1, size_ens = size_ens)
    
    initial_ensembles2 = generate_initial_ensembles(samp_ann_params, var_weights, size_ens)
    data1_out2, data1_stack2 = get_targets_with_weights(data1, initial_ensembles2, size_ens = size_ens)
    
    initial_ensembles3 = generate_initial_ensembles(samp_ann_params, var_weights, size_ens)
    data2_out1, data2_stack1 = get_targets_with_weights(data2, initial_ensembles3, size_ens = size_ens)
    
    initial_ensembles4 = generate_initial_ensembles(samp_ann_params, var_weights, size_ens)
    data2_out2, data2_stack2 = get_targets_with_weights(data2, initial_ensembles4, size_ens = size_ens)   
    
    X_t = np.concatenate((np.expand_dims(data1_stack1, -1), np.expand_dims(data1_stack2, -1), 
                         np.expand_dims(data2_stack1, -1), np.expand_dims(data2_stack2, -1)), axis = -1)
    
    initial_ensembles_for_weights = generate_initial_ensembles(4, var_weights, size_ens)
    initial_ensembles_for_weights = np.expand_dims(initial_ensembles_for_weights,1)
    
    initial_ensembles_for_L = generate_initial_ensembles(4, var_weights, size_ens)
    initial_ensembles_for_L = np.expand_dims(initial_ensembles_for_L,1)    
    
    initial_ensembles_for_D1 = generate_initial_ensembles(1, var_weights, size_ens).reshape(-1,1)
    initial_ensembles_for_D2 = generate_initial_ensembles(1, var_weights, size_ens).reshape(-1,1)
    
    initial_ensembles_for_D1_zero = np.zeros((size_ens,1,1)).reshape(-1,1)
    initial_ensembles_for_D2_zero = np.zeros((size_ens,1,1)).reshape(-1,1)
    
    initial_ensembles_for_D = np.concatenate((np.expand_dims(initial_ensembles_for_D1,1),
                                                       np.expand_dims(initial_ensembles_for_D1_zero,1), 
                                                      np.expand_dims(initial_ensembles_for_D2,1),
                                                       np.expand_dims(initial_ensembles_for_D2_zero,1)), axis = 2)
    
    # print(X_t.shape, initial_ensembles_for_weights.shape)
    
    X_t = np.concatenate((X_t, initial_ensembles_for_weights, initial_ensembles_for_L, initial_ensembles_for_D), axis = 1)
    
    initial_ensembles = np.hstack((initial_ensembles1, initial_ensembles2, initial_ensembles3, initial_ensembles4))
    
    return X_t, initial_ensembles, initial_ensembles_for_weights[:,0,:], initial_ensembles_for_L[:,0,:], initial_ensembles_for_D[:,0,:]

def get_weighted_targets_with_weights(batch_data, initial_ensembles, size_ens, weights): 
    
    target_dim = 1
    
    # weights_ann_1 = ann.get_weights()
    
    # h1  = ann.layers[1].output.shape[-1]

    n_hidden_1 = len(weights_ann_1[0].ravel())
    
    hidden_weights_1 = initial_ensembles[:,:n_hidden_1].reshape( size_ens, batch_data.shape[1], h1)
    
    
    hidden_output_1 = np.einsum('ij,kjl->kil', batch_data, hidden_weights_1)

    
    hidden_layer_bias_1 = initial_ensembles[:,n_hidden_1:(n_hidden_1 + h1)].reshape(size_ens, 1,  h1)


    hidden_output_1 = hidden_output_1 + hidden_layer_bias_1

    n_pred_weights_1 = len(weights_ann_1[2].ravel())

    output_weights_1 = initial_ensembles[:,(n_hidden_1 + h1):(n_hidden_1 + h1 + n_pred_weights_1) ].reshape(size_ens, h1, target_dim)


    output_1 = np.einsum('ijk,ikl->ijl', hidden_output_1, output_weights_1)


    output_layer_bias_1 = initial_ensembles[:,(n_hidden_1 + h1 + n_pred_weights_1):(n_hidden_1 + h1 + n_pred_weights_1 + target_dim)].reshape(size_ens, 1, target_dim)


    final_output_1 = output_1 + output_layer_bias_1
    
    final_output_1 = final_output_1[:,:, 0]
    
    final_output_1 = final_output_1*weights
    
    # print(final_output_1.shape, initial_ensembles.shape)
    
    stack = np.hstack((final_output_1, initial_ensembles))

    
    return final_output_1, stack

std_targets = pickle.load( open('..//Data//target_scaler.pkl', 'rb'))


def forward_operation(data1, data2, combined_ensembles , size_ens ):
    # samp_ann =  ann(hidden = hidden_neurons, input_shape = 32, output_shape = 1)
    params = samp_ann_params
    initial_ensembles1 = combined_ensembles[:, :params]
    initial_ensembles2 = combined_ensembles[:, params:(2*params)]
    initial_ensembles3 = combined_ensembles[:, (2*params):(3*params)]
    initial_ensembles4 = combined_ensembles[:, (3*params):(4*params)]

    
    initial_ensembles_for_weights = combined_ensembles[:, (4*params):(4*params + 4)]
    
    initial_ensembles_for_L = combined_ensembles[:, (4*params + 4):(4*params + 4 + 4)]
    
    initial_ensembles_for_D = combined_ensembles[:,(4*params + 4 + 4):(4*params + 4 + 4 + 4)]
    
    
    softmax_weights = tf.math.softmax(initial_ensembles_for_weights).numpy()
    
    model_1 = softmax_weights[:,:2].sum(1).reshape(-1,1)
    
    model_2 = softmax_weights[:,2:].sum(1).reshape(-1,1)
    
    data1_out1, data1_stack1 = get_weighted_targets_with_weights(data1, initial_ensembles1, size_ens = size_ens,
                                                                  weights=model_1)
    
    data1_out2, data1_stack2 = get_weighted_targets_with_weights(data1, initial_ensembles2, size_ens = size_ens,
                                                                weights=model_1)
    
    data2_out1, data2_stack1 = get_weighted_targets_with_weights(data2, initial_ensembles3, size_ens = size_ens,
                                                                 weights=model_2)
    
    data2_out2, data2_stack2 = get_weighted_targets_with_weights(data2, initial_ensembles4, size_ens = size_ens,
                                                                  weights=model_2)   
    
    X_t = np.concatenate((np.expand_dims(data1_stack1, -1), np.expand_dims(data1_stack2, -1), 
                         np.expand_dims(data2_stack1, -1), np.expand_dims(data2_stack2, -1)), axis = -1)
    
    initial_ensembles = np.hstack((initial_ensembles1, initial_ensembles2, initial_ensembles3, initial_ensembles4, 
                        initial_ensembles_for_weights, initial_ensembles_for_L, initial_ensembles_for_D))
    
    # print(X_t.shape)
    
    initial_ensembles_for_weights = np.expand_dims(initial_ensembles_for_weights,1)
    
    initial_ensembles_for_L = np.expand_dims(initial_ensembles_for_L,1)
    
    initial_ensembles_for_D = np.expand_dims(initial_ensembles_for_D,1)
    
    # print(initial_ensembles_for_weights.shape)
    
    X_t = np.concatenate((X_t, initial_ensembles_for_weights, initial_ensembles_for_L, initial_ensembles_for_D), axis = 1)
    
    weighted_alogp = data1_out1 + data2_out1
    
    weighted_psa = data1_out2 + data2_out2
    
    return X_t, initial_ensembles, weighted_alogp, weighted_psa, model_1, model_2

total_weights = 4*(samp_ann.count_params() + 1 + 1 + 1)


reduction = 12

size_ens = total_weights//reduction

G_t = [[1, 0, 1, 0], [0, 1, 0, 1]]
G_t = np.array(G_t).T

def get_predictions(data1, data2, initial_ensembles): 
    _,_, weighted_alogp, weighted_psa, w1, w2 = forward_operation(data1, data2, initial_ensembles, size_ens = size_ens)
    weighted_alogp = np.expand_dims(weighted_alogp,-1)
    weighted_psa = np.expand_dims(weighted_psa,-1)
    preds = np.concatenate((weighted_alogp, weighted_psa),-1)
    return preds, w1, w2

def calculate_mu_bar_G_bar(data1, data2, initial_ensembles):
    H_t = np.hstack((np.identity(data1.shape[0]), np.zeros((data1.shape[0], samp_ann_params + 1 + 1 + 1))))
    mu_bar = initial_ensembles.mean(0)
    X_t,_, _, _, _, _ = forward_operation(data1, data2, initial_ensembles, size_ens = size_ens)
    X_t = X_t.transpose((0,2,1))
    X_t = X_t.reshape(X_t.shape[0], X_t.shape[1]*X_t.shape[2])
    script_H_t = np.kron(G_t.T, H_t)
    G_u = (script_H_t@X_t.T)
    G_u = G_u.T
    # weighted_alogp = np.expand_dims(weighted_alogp,-1)
    # weighted_psa = np.expand_dims(weighted_psa,-1)
    # G_u = np.concatenate((weighted_alogp, weighted_psa), axis = -1)
    # G_u = G_u.transpose((0,2,1))
    # G_u = G_u.reshape(G_u.shape[0], G_u.shape[1]*G_u.shape[2])
    # G_u
    G_bar = (G_u.mean(0)).ravel()
    return mu_bar.reshape(-1,1), G_bar.reshape(-1,1), G_u


def calculate_C_u(initial_ensembles, mu_bar, G_bar, G_u): 
    u_j_minus_u_bar = initial_ensembles - mu_bar.reshape(1,-1)
    G_u_minus_G_bar = G_u -  G_bar.reshape(1,-1)
    c = np.zeros((total_weights, G_bar.shape[0]))
    for i in range(0, size_ens): 
        c += np.kron(u_j_minus_u_bar[i, :].T.reshape(-1,1), G_u_minus_G_bar[i,:].reshape(-1,1).T)
    return c/size_ens, G_u_minus_G_bar


def calculate_D_u( G_bar, G_u): 
    G_u_minus_G_bar = G_u -  G_bar.reshape(1,-1)
    d = np.zeros((G_bar.shape[0], G_bar.shape[0]))
    for i in range(0, size_ens): 
        d += np.kron(G_u_minus_G_bar[i,:].T.reshape(-1,1), G_u_minus_G_bar[i,:].reshape(-1,1).T)
    return d/size_ens


def get_updated_ensemble(data1, data2, initial_ensembles, y_train, size_ens = size_ens, inflation_factor = 1.0):
    mu_bar, G_bar, G_u = calculate_mu_bar_G_bar(data1, data2, initial_ensembles)
    C, G_u_minus_G_bar = calculate_C_u(initial_ensembles, mu_bar, G_bar, G_u)
    D = calculate_D_u( G_bar, G_u)
    _, R_t = create_cov(data1.shape[0],initial_ensembles)
    inflation = np.identity(R_t.shape[0])*inflation_factor
    D_plus_cov = D + (R_t *inflation_factor)
    # all_covs = np.array(all_covs)
    # D_plus_cov = D + R_t
    D_plus_cov_inv = np.linalg.inv(D_plus_cov)
    mid_quant = C@D_plus_cov_inv
    noise_vec_mean = np.zeros((R_t.shape[0], ))
    noise_mvn = mvn(noise_vec_mean, R_t)
    fudging = noise_mvn.rvs(size_ens)
    interim = (y_train.T.flatten().reshape(1,-1) + fudging)
    right_quant = interim - G_u
    # print(mid_quant.shape, right_quant.shape)
    mid_times_right = mid_quant@right_quant.T
    updated_ensemble = (initial_ensembles + mid_times_right.T)
    return updated_ensemble


target_dim = 2


def inverse_transform(data, idx):
    data_cur = data[idx, :, :]
    inv_data_cur = std_targets.inverse_transform(data_cur)
    return inv_data_cur


from joblib import Parallel, delayed


def create_cov(shape, initial_ensembles):
    cov_part = initial_ensembles[:, -8:-4]
    cov_part = cov_part.mean(0)
    # variances = tf.math.softplus(cov_part[:2]).numpy()
    variances = cov_part[:2]
    covariances = cov_part[2:]
    base_cov = np.identity(target_dim)
    base_cov[0,0] = variances[0]
    base_cov[1,1] = variances[1]
    base_cov[0,1] = covariances[0]
    base_cov[1,0] = covariances[1]
    
    variances1 = tf.math.softplus(initial_ensembles[:, -4:]).numpy()
    variances1 = variances1.mean(0)
    base_variances = np.identity(target_dim)
    base_variances[0,0] = variances1[0]
    base_variances[1,1] = variances1[2]
    
    final = np.linalg.cholesky(base_cov@base_cov.T + base_variances)
    cov_mat = final@final.T
    cov_mat_final = cov_mat
    # cov_mat_final = cov_mat@cov_mat.T
    
    if is_pos_def(cov_mat_final) != True:
        print("resulting cov matrix is not positive semi definite")
        pass
    
    # print(np.linalg.det(cov_mat_final))
    
    var1 = cov_mat_final[0,0]
    var2 = cov_mat_final[1,1]
    cov = cov_mat_final[1,0]

    n = shape
    
    ul = var1*np.identity(n)
    lr = var2*np.identity(n)
    ur = cov*np.identity(n)
    ll = ur.T    
    
    first_row = np.hstack((ul, ur))
    second_row = np.hstack((ll, lr))
    
    R_t = np.vstack((first_row, second_row))
    
    # R_t = block_diag(*([cov_mat_final] * n))
    
    # R_t = np.linalg.inv(R_t)
    
    return cov_mat_final, R_t
    
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

with open("..//Data//smiles_to_rdkit_80_20_with_cov_minus_018.pickle", "rb") as f: 
    catch = pickle.load(f)
    
    
    
def prepare_data(idx): 
    catch_idx = catch[idx]
    x_train, x_valid, y_train, y_valid = catch_idx[0], catch_idx[1], catch_idx[2], catch_idx[3]
    y_train_actual, y_train = y_train[:,:2], y_train[:,2:]
    y_valid_actual, y_valid = y_valid[:,:2], y_valid[:,2:]
    smiles_feats_train = x_train[:, :32]
    rdkit_feats_train = x_train[:, 32:]
    smiles_feats_valid = x_valid[:, :32]
    rdkit_feats_valid = x_valid[:, 32:]

    X_t, initial_ensembles, initial_ensembles_for_weights, initial_ensembles_for_L, initial_ensembles_for_D = get_initial_X_t(smiles_feats_train, rdkit_feats_train, size_ens = size_ens)
    initial_ensembles = np.hstack((initial_ensembles, initial_ensembles_for_weights, initial_ensembles_for_L, initial_ensembles_for_D))
    
    return smiles_feats_train, rdkit_feats_train, smiles_feats_valid, rdkit_feats_valid, y_train, y_train_actual, y_valid, y_valid_actual, initial_ensembles 


from scipy.linalg import norm

def get_results(idx):
    smiles_feats_train, rdkit_feats_train, smiles_feats_valid, rdkit_feats_valid, y_train, y_train_actual, y_valid, y_valid_actual, initial_ensembles  = prepare_data(idx)
    # print(R_t.shape)
    best_train_width_mean = 100000
    
    for i in range(0,10000):
        # print(i)
    
        c = np.zeros((2,2))
        initial_ensembles = get_updated_ensemble(smiles_feats_train, rdkit_feats_train, initial_ensembles, y_train)
        G_u_train, w1, w2 = get_predictions(smiles_feats_train, rdkit_feats_train, initial_ensembles)

        catch = Parallel(n_jobs = 15, verbose = 0)(delayed(inverse_transform)(G_u_train, i)  for i in range(G_u_train.shape[0]))
        G_u_train = np.array(catch)
    
        y_train_cur = std_targets.inverse_transform(y_train_actual)
    
        li_train = np.percentile(G_u_train, axis = 0, q = (2.5, 97.5))[0,:,:]   
        ui_train = np.percentile(G_u_train, axis = 0, q = (2.5, 97.5))[1,:,:]
    
        width_train = ui_train - li_train
        avg_width_train = width_train.mean(0)
    
        ind_train = (y_train_cur >= li_train) & (y_train_cur <= ui_train)
        coverage_train= ind_train.mean(0)
    
        averaged_targets_train = G_u_train.mean(0)
        rmse_train = np.sqrt(((y_train_cur -averaged_targets_train)**2).mean(0))
    # print(rmse_train, coverage_train, avg_width_train)
    
        G_u_test, _, _ = get_predictions(smiles_feats_valid, rdkit_feats_valid, initial_ensembles)
    
        catch = Parallel(n_jobs = 15, verbose = 0)(delayed(inverse_transform)(G_u_test, i)  for i in range(G_u_test.shape[0]))
        G_u_test = np.array(catch)
    
        y_valid_cur = std_targets.inverse_transform(y_valid_actual)    
    
        li_test = np.percentile(G_u_test, axis = 0, q = (2.5, 97.5))[0,:,:]   
        ui_test = np.percentile(G_u_test, axis = 0, q = (2.5, 97.5))[1,:,:]
    
        width_test = ui_test - li_test
        avg_width_test = width_test.mean(0)
    
        ind_test = (y_valid_cur >= li_test) & (y_valid_cur <= ui_test)
        coverage_test= ind_test.mean(0)
    
        averaged_targets_test = G_u_test.mean(0)
        rmse_test = np.sqrt(((y_valid_cur -averaged_targets_test)**2).mean(0))    
    
        # weight_norms = np.array(norm(initial_ensembles, ord = 2, axis = 1))
        # weight_norm_mean.append(weight_norms.mean())
        # weight_norm_sd.append(weight_norms.std())
    
        cov_mat_final, _ = create_cov(smiles_feats_train.shape[0],initial_ensembles)
        
        # print("standardized_scale_R_t")
        # print(np.diag(cov_mat_final), cov_mat_final[0,1])
        
        # print(w1.shape)
        
        li_smiles_weight = np.percentile(w1, axis = 0, q = (2.5, 97.5))[0][0]
        
        # print(np.percentile(w1, axis = 0, q = (2.5, 97.5)))
        
        ui_smiles_weight = np.percentile(w1, axis = 0, q = (2.5, 97.5))[1][0]      
        
        # print(coverage_train.tolist(), avg_width_train.tolist(), rmse_train.tolist())
        # print(coverage_test.tolist(), avg_width_test.tolist(), rmse_test.tolist())
        # print(w1.mean())
        # print(li_smiles_weight, ui_smiles_weight)
        # print(avg_width_train.tolist(), coverage_train.tolist(), rmse_train.tolist(), avg_width_test.tolist(), coverage_test.tolist(), rmse_test.tolist(), w1.mean())

        if (avg_width_train.mean() < best_train_width_mean) & (coverage_train.mean() > 0.95): 
            # print("went here")
            best_train_width_mean = avg_width_train.mean()
            best_train_width = avg_width_train
            best_smiles_weight = w1.mean()
            best_coverage_train = coverage_train
            best_rmse_train = rmse_train
        
            best_test_width = avg_width_test

            best_coverage_test = coverage_test    
            best_rmse_test = rmse_test
            
            best_li_smiles_weight = li_smiles_weight
            
            best_ui_smiles_weight = ui_smiles_weight
    
        if coverage_train.mean() < 0.95:
            # print(best_train_width.tolist(), best_coverage_train.tolist(), best_rmse_train.tolist(), best_test_width.tolist(), best_coverage_test.tolist(), best_rmse_test.tolist(), best_smiles_weight, flush = True)
            print("done idx " + str(idx), flush = True)
            return [best_train_width.tolist(), best_coverage_train.tolist(), best_rmse_train.tolist(), best_test_width.tolist(), best_coverage_test.tolist(), best_rmse_test.tolist(), best_smiles_weight, best_li_smiles_weight, best_ui_smiles_weight]

        
import multiprocessing

cpu = multiprocessing.cpu_count()-1
        
catch_all = Parallel(n_jobs = cpu, verbose = 10)(delayed(get_results)(idx) for idx in range(0,50))


all_catch = []
for item in catch_all:
    catch_inner = []
    for inner in item:
        if type(inner) == list:
            for inner1 in inner:
                catch_inner.append(inner1)
    if type(inner) != list:
        catch_inner.append(inner)
    all_catch.append(catch_inner)
    
    
results_df = pd.DataFrame(all_catch)

col_names = ["Alop_Train_Width", "PSA_Train_Width", "Alop_Train_Coverage", "PSA_Train_Coverage", 
            "Alop_Train_RMSE", "PSA_Train_RMSE", "Alop_Test_Width", "PSA_Test_Width", "Alop_Test_Coverage", "PSA_Test_Coverage", 
            "Alop_Test_RMSE", "PSA_Test_RMSE", "Smiles_Avg_Weight", "Lower_Interval_Smiles_Weight", "Upper_Interval_Smiles_Weight"]

results_df.columns = col_names

results_df.to_csv("..//Data//smiles_rdkit_80_20__with_cov_minus_018_Simulation.csv", index = False)