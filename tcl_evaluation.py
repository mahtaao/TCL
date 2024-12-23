""" Evaluation
    Main script for evaluating the model trained by tcl_training.py
"""





import os
import numpy as np
import pickle

from subfunc.preprocessing import pca
from subfunc.showdata import *
from sklearn.decomposition import FastICA
import torch
from tcl_pytorch.custom_datase import SimulatedDataset
from tcl_pytorch.model import TCL,TCL_new
import torch.utils.data as data
from sklearn.metrics import confusion_matrix,accuracy_score
from subfunc.munkres import Munkres
import seaborn as sns


# parameters ==================================================
# =============================================================

eval_dir=f'./experiment/layer{5}-seg{8}' 
parmpath = os.path.join(eval_dir, 'parm.pkl')
modelpath = os.path.join(eval_dir, 'model.pth')
apply_fastICA = True
nonlinearity_to_source = 'abs' # Assume that sources are generated from laplacian distribution



# =============================================================
def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort

# =============================================================
# =============================================================

# Load trained file -------------------------------------------


state_dict = torch.load(modelpath)


# Load parameter file
with open(parmpath, 'rb') as f:
    model_parm = pickle.load(f)

num_comp = model_parm['num_comp']
num_segment = model_parm['num_segment']
num_segmentdata = model_parm['num_segmentdata']
num_layer = model_parm['num_layer']
list_hidden_nodes = model_parm['list_hidden_nodes']
moving_average_decay = model_parm['moving_average_decay']
random_seed = model_parm['random_seed']
pca_parm = model_parm['pca_parm']
batch_size = 8 

# Generate sensor signal --------------------------------------
eval_dataset = SimulatedDataset(num_comp=num_comp,
                                                 num_segment=num_segment,
                                                 num_segmentdata=num_segmentdata,
                                                 num_layer=num_layer,
                                                 random_seed=random_seed)


# Preprocessing -----------------------------------------------

model = TCL(input_size=eval_dataset.__getinputsize__(), list_hidden_nodes=list_hidden_nodes, num_class=num_segment)
model.load_state_dict(state_dict)

data_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)


test_acc =0
abscorr = []
labels=[]
predictions=[]
if apply_fastICA:
    ica = FastICA(random_state=random_seed)
# Evaluate model ----------------------------------------------
feat_vals =[]
for data_inputs, data_labels in data_loader:
    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
    x_batch = data_inputs
    y_batch = data_labels
    # Forward pass
    logits, feats = model(x_batch)
    # Calculate predictions.
    top_values, pred = torch.topk(logits, k=1)
    test_acc += torch.sum(pred == y_batch)
    labels.extend(y_batch.detach().numpy())
    predictions.extend(pred.detach().numpy())
        # Apply fastICA -----------------------------------------------
    feat_vals.append(feats.detach().numpy())
    if apply_fastICA:
        feateval = feats.T 
        feat_val = ica.fit_transform(feateval.detach().numpy())
    else:
        feat_val = feats.detach().numpy()
    # Evaluate ----------------------------------------------------5
    if nonlinearity_to_source == 'abs':
        xseval = np.abs(x_batch) # Original source
    else:
        raise ValueError
    # Estimated feature
    #
    corrmat, sort_idx, _ = correlation(feat_val.T, xseval.detach().numpy(), 'Pearson')
    abscorr.extend(np.abs(np.diag(corrmat)))

# accuracy = test_acc/eval_dataset.__len__()
accuracy = accuracy_score(labels, predictions)
confmat= confusion_matrix(labels, predictions)
meanabscorr=np.mean(abscorr)
from sklearn.linear_model import LinearRegression

source, pca_parm = pca(eval_dataset.source, num_comp=num_comp)

feat_vals = np.stack(feat_vals, axis=0).reshape(-1,20)
reg1 = LinearRegression().fit(feat_vals, source.transpose())
tcl_score= reg1.score(feat_vals, source.transpose())

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(reg1.coef_ ,cmap='hot', interpolation='nearest')
ax = sns.heatmap(reg1.coef_, cmap="Reds")
plt.savefig("tcl.png")
reg2 = LinearRegression().fit(eval_dataset.sensor.transpose(), source.transpose())
pca_score= reg2.score(eval_dataset.sensor.transpose(),source.transpose())
print("PCA coef_")
ax = sns.heatmap(reg2.coef_, cmap="Reds")
plt.savefig("pca.png")

# Display results ---------------------------------------------
print("Result...")
print("    accuracy(test) : {0:7.4f} [%]".format(accuracy*100))
print("    correlation     : {0:7.4f}".format(meanabscorr))
print("    TCL_score(test) : {0:7.4f}".format(tcl_score))
print("    PCA_score(test) : {0:7.4f} ".format(pca_score))
print(f"    TCL_intercep(test) :{reg1.intercept_}")
print(f"    PCA_intercep(test) :{reg2.intercept_}")



print("done.")


