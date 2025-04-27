import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
from torch_geometric.nn import NNConv, global_max_pool
# ------------------------------
# 1. Data Featurization Functions
# ------------------------------
def featurize_atom(atom):
    """Extract feature vector for one atom."""
    features = []
    # (1) Atomic number (one-hot or integer)
    features.append(atom.GetAtomicNum())
    # (2) Degree
    features.append(atom.GetDegree())
    # (3) Formal charge
    features.append(atom.GetFormalCharge())
    
    features.append(atom.GetTotalValence())
    
    features.append(atom.GetNumRadicalElectrons())
    # (4) Hybridization
    hybridization = atom.GetHybridization()
    # Create a simple one-hot encoding for some common hybridizations:
    # [SP, SP2, SP3, OTHER]
    hybrid_map = {
        Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0],
        Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0],
        Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0]
    }
    if hybridization in hybrid_map:
        features.extend(hybrid_map[hybridization])
    else:
        features.extend([0, 0, 0, 1])
    # (5) Is aromatic
    features.append(int(atom.GetIsAromatic()))
    # Add more features as needed...

    return features

def featurize_bond(bond):
    """
    Returns a list of bond features, including bond type, conjugation, ring membership.
    """
    bt = bond.GetBondType()
    bond_type = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0
    ]
    return bond_type + [
        1 if bond.GetIsConjugated() else 0,
        1 if bond.IsInRing() else 0
    ]

def get_global_features(mol):
    """Extract global features for the molecule such as MW, LogP, etc."""
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
    logp = Descriptors.MolLogP(mol)
    num_val = Descriptors.NumValenceElectrons(mol)
    #lip_HBA = Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)
    #lip_HBD = Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)
    #rot_bond = Descriptors.NumRotatableBonds(mol)
    #volume = AllChem.ComputeMolVolume(mol)
    return [mw, tpsa, num_rings, logp, num_val, #lip_HBA, lip_HBD, 
            #rot_bond, 
            #volume
            ]

# ------------------------------
# 2. Custom PyTorch Geometric Dataset
# ------------------------------
class SMILESDataset(Dataset):
    def __init__(self, dataframe, transform=None, pre_transform=None):
        """
        dataframe: pandas DataFrame containing 'SMILES' and 'pAffinity' columns.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.dataframe)

    def get(self, idx):
        smiles = self.dataframe.loc[idx, 'SMILES']
        y_value = self.dataframe.loc[idx, 'pAffinity']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # In case of invalid SMILES, return an empty Data object
            # or handle accordingly
            return Data()

        # Add Hs for a better atom environment (optional)
        mol = Chem.AddHs(mol)
        # AllChem.EmbedMolecule(mol, randomSeed=42)
        # AllChem.UFFOptimizeMolecule(mol)

        # Atom features
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(featurize_atom(atom))
        x = torch.tensor(atom_features_list, dtype=torch.float)

        # Edge (bond) features
        edge_indices = []
        edge_features = []
        #num_atoms = mol.GetNumAtoms()
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            bond_feature = featurize_bond(bond)

            edge_indices.append([start, end])
            edge_indices.append([end, start])
            edge_features.append(bond_feature)
            edge_features.append(bond_feature)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Molecules with no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)

        # Global features
        global_feat = get_global_features(mol)
        u = torch.tensor(global_feat, dtype=torch.float).view(1, -1)  # shape [1, num_global_features]

        y = torch.tensor([y_value], dtype=torch.float)

        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y)

        # Attach global features
        data.u = u  # shape [1, c]

        return data

# ------------------------------
# 3. Define the GNN Model
# ------------------------------
from torch_geometric.nn import NNConv, global_max_pool

class EdgeNet(nn.Module):
    """
    A simple GNN model that incorporates edge attributes via a learned transformation (NNConv).
    """
    def __init__(self, in_channels, edge_in_channels, global_in_channels,
                 hidden_channels=64, num_layers=4):
        super(EdgeNet, self).__init__()

        # We'll build a small feedforward that processes edge attributes
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels * hidden_channels)
        )

        # GNN Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(NNConv(in_channels, hidden_channels, self.edge_nn))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            # Need a new instance of edge_nn for each layer or a universal approach
            edge_nn = nn.Sequential(
                nn.Linear(edge_in_channels, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_channels * hidden_channels)
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, edge_nn))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final linear layers after pooling
        # We incorporate global features by concatenating them after the pooling step
        self.post_pool = nn.Sequential(
            nn.Linear(hidden_channels + global_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output 1 for regression
        )
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, edge_attr, batch, u):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_in_channels]
        # u: [num_graphs, global_in_channels] but used after pooling

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        # global pooling
        x = global_max_pool(x, batch)  # [num_graphs, hidden_channels]
        # x = self.dropout(x)
        # concat with global features
        # u has shape [num_graphs, global_in_channels], so matching dimension
        x = torch.cat([x, u], dim=-1)
        x = self.dropout(x)
        # final FFN
        out = self.post_pool(x)
        return out.view(-1)
    
# ------------------------------
# 4. Training and Evaluation
# ------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.u)
        loss = F.mse_loss(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.u)
            loss = F.mse_loss(out, data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            preds.append(out.cpu().numpy())
            trues.append(data.y.view(-1).cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Compute R², slope, intercept
    r2 = r2_score(trues, preds)
    slope, intercept, r_value, p_value, std_err = stats.linregress(trues, preds)
    return avg_loss, r2, slope, intercept, preds, trues

# ------------------------------
# 5. Residual Analysis
# ------------------------------
def plot_residuals_vs_predicted(preds, trues, idx, output_dir):
    residuals = trues - preds
    std_residuals = np.std(residuals)
    plt.figure(figsize=(6, 5))
    plt.scatter(preds, residuals, alpha=0.7)
    for i in range(len(preds)):
        if residuals[i]> 3*std_residuals or residuals[i]<-3*std_residuals:
            plt.annotate(idx[i], (preds[i], residuals[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize = 12)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted')
    plt.savefig(os.path.join(output_dir, "Residuals_vs_Predicted.png"), dpi=300)
    plt.show()

def normal_probability_plot(preds, trues, output_dir):
    residuals = trues - preds
    plt.figure(figsize=(6,5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Probability Plot of Residuals')
    plt.savefig(os.path.join(output_dir, "NPP.png"), dpi=300)
    plt.show()

def compute_hat_matrix_approx(features,preds, trues, idx):
    """
    Approximate the Hat matrix diagonal by using features in a linear sense.
    features: N x D (where N is the number of samples, D is dimension).
    Returns the approximate leverage for each sample (diagonal of the Hat matrix).
    """
    # In conventional linear regression, H = X (X'X)^-1 X'.
    # We'll do that with 'features' here, though GNN usage is more complex.
    X = np.array(features)
    n_vars=X.shape[0]
    n_samples=X.shape[1]
    # Add a bias column if you want to incorporate an intercept in the design
    # Example:
    # X = np.hstack([X, np.ones((X.shape[0], 1))])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # In case of singular matrix, use pseudo-inverse
        XtX_inv = np.linalg.pinv(XtX)
    H = X @ XtX_inv @ X.T
    leverages = np.diag(H)
    # The diagonal of H is the leverage for each sample
    critical_leverages =  3 * (n_vars + 1) / n_samples
    
    residuals = trues - preds
    std_resid = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-9)
    plt.figure(figsize=(6,5))
    plt.scatter(leverages, std_resid, alpha=0.7)

    plt.figure()
    plt.scatter(leverages, std_resid, alpha=0.6)
    plt.axhline(y=3, color='r', linestyle='--', label='±3 Std Dev Limit')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.axvline(x=critical_leverages, color='g', linestyle='--', label='Leverage Limit')
    for i in range(len(leverages)):
        plt.annotate(idx[i], (leverages[i], std_resid[i]),textcoords= "offset points", xytext= (5,5), ha='center', fontsize=12)
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    plt.title('Williams Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

    return 
    

def williams_plot(train, val, test, train_pred, val_pred, test_pred, train_nom, val_nom, test_nom, train_idx, val_idx, test_idx, output_dir):
    """
    plot all three data sets on one williams plot
    """
    #Notation
    #def notation(h, h_crit, std_residuals, idx):
        #for i in range(len(h)):
            #if h[i]>h_crit or std_residuals[i]>3 or std_residuals[i]<-3:
                #plt.annotate(idx[i], (h[i],std_residuals[i]),textcoords= "offset points", xytext= (5,5), ha='center', fontsize=12)
              
    #Calculate leverage values
    try:
        scores_dict = {
            'train' : train,
            'val' : val,
            'test' : test
            }
        train_inv = np.linalg.pinv(train.T @ train)
        leverage_dict = {}
        for dset, scores in scores_dict.items():
            leverage_dict[dset] = np.diag(scores @ train_inv @ scores.T)
              
        train_lev = leverage_dict['train']
        val_lev = leverage_dict['val']
        test_lev = leverage_dict['test']
        
    except Exception as e:
        print(f'the error in leverage calculation is: {e}')
    
    #Calculate std residuals
    try:
        pred_dict = {
            'train':{'pred': train_pred, 'nom': train_nom},
            'val':{'pred':val_pred, 'nom': val_nom},
            'test':{'pred':test_pred, 'nom': test_nom}
            }

        std_residual_dict = {}  
        for name, data in pred_dict.items():
            residual = data['nom'] - data['pred']
            std_residual_dict[name] = (residual - np.mean(residual))/(np.std(residual) + 1e-9)
            
        train_std_residuals = std_residual_dict['train']
        val_std_residuals = std_residual_dict['val']
        test_std_residuals = std_residual_dict['test']
    except Exception as e:
        print(f"The error in std_residual calculation is: {e}")
        
    # Critical leverage threshold
    p = train.shape[1]  # Number of variables
    n = len(train_pred)
    h_crit = 3 * (p + 1) / n
    
    plt.figure(figsize=(10,8))
    plt.scatter(train_lev, train_std_residuals, label= 'Train', edgecolors='k', color = 'blue', s = 60)
    plt.scatter(val_lev, val_std_residuals, label= 'Validation', edgecolors= 'k', color = 'green', s = 60)
    plt.scatter(test_lev, test_std_residuals, label = 'Test', marker= 'D',edgecolors='k', color = 'orange', s = 60)
    #notation(train_lev, h_crit, train_std_residuals, train_idx)
    #notation(val_lev, h_crit, val_std_residuals, val_idx)
    #notation(test_lev, h_crit, test_std_residuals, test_idx)
    plt.axhline(y=3, color='r' , linestyle = '--')
    plt.axhline(y=-3, color= 'r', linestyle = '--')
    plt.axvline(x = h_crit, color= 'g', linestyle = '--')
    plt.annotate(f' h* = {h_crit:0.2f}', (h_crit, np.min(train_std_residuals)), fontsize = 16)
    plt.legend(fontsize = 14, loc = 'upper right')
    plt.xlabel('Leverage', fontsize = 14)
    plt.ylabel('std residuals', fontsize = 14)
    plt.title('Williams plot', fontsize = 16)
    plt.savefig(os.path.join(output_dir,'Williams_Plot.png'), dpi=300)
    plt.show()
    
def evaluate_statistics_for_datasets(model, train_loader, val_loader, test_loader, device):
    """
    Computes typical regression statistics (MSE, R², slope, intercept) 
    for three datasets: train, val, test.
    Returns a dictionary of results.
    """
    results = {}
    for name, loader in zip(['Train', 'Val', 'Test'], [train_loader, val_loader, test_loader]):
        avg_loss, r2, slope, intercept, preds, trues = evaluate(model, loader, device)
        mae = mean_absolute_error(trues, preds)
        results[name] = {
            'MSE': avg_loss,
            'MAE': mae,
            'R2': r2,
            'Slope': slope,
            'Intercept': intercept
        }
    return results
import statsmodels.api as sm
###########################################
#           plot_true_vs_predicted
###########################################
def plot_true_vs_predicted(
    true_values,
    predicted_values,
    output_dir,
    idx,
    dataset_label='Training',
    color='blue',
):
    # Ensure inputs are numpy arrays
    x = np.array(true_values, dtype=float)
    y = np.array(predicted_values, dtype=float)
    
    # Add a constant term for intercept
    X = sm.add_constant(x)
    
    # Fit the regression model: y = intercept + slope*x
    model = sm.OLS(y, X).fit()
    
    # Extract slope, intercept, standard errors, R²
    intercept = model.params[0]
    slope = model.params[1]
    intercept_se = model.bse[0]
    slope_se = model.bse[1]
    r_squared = model.rsquared
    
    # Generate points for the regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    X_line = sm.add_constant(x_line)
    predictions = model.get_prediction(X_line)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
    lower_bound_regression = pred_summary["mean_ci_lower"]
    upper_bound_regression = pred_summary["mean_ci_upper"]
    
    pred_summary_99 = predictions.summary_frame(alpha=0.01)  # 99% CI
    lower_bound_regression_99 = pred_summary_99["mean_ci_lower"]
    upper_bound_regression_99 = pred_summary_99["mean_ci_upper"]

    
    # --- Confidence Interval Around Predictions ---
    # Calculate 95% confidence interval
    residuals = y - x
    std_res = np.std(residuals)
    n = len(x)
    standard_error = std_res * np.sqrt(1 + 1/n + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
    y_fit = model.predict(X)
    upper_bound_prediction = y_fit + t_val * standard_error  # This should be 1D
    lower_bound_prediction = y_fit - t_val * standard_error  # This should be 1D
    
    t_val_99 = stats.t.ppf(0.999, n - 2)  # 95% confidence
    upper_bound_prediction_99 = y_fit + t_val_99 * standard_error  # This should be 1D
    lower_bound_prediction_99 = y_fit - t_val_99 * standard_error  # This should be 1D

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Fill the confidence interval around the regression line
    ax.fill_between(
        x_line, lower_bound_regression_99, upper_bound_regression_99,
        color='darkgray', alpha=0.9, label='99% CI (Regression Line)'
    )
    
    ax.fill_between(
        x_line, lower_bound_regression, upper_bound_regression,
        color='yellow', alpha=0.8, label='95% CI (Regression Line)'
    )

    # Fill the confidence interval
    ax.fill_between(x, lower_bound_prediction, upper_bound_prediction, 
                    color='red', alpha=0.2, 
                    label='95% Confidence Interval')
    
    ax.fill_between(x, lower_bound_prediction_99, upper_bound_prediction_99, 
                    color='purple', alpha=0.2, 
                    label='99% Confidence Interval')
    
    # Scatter plot of true vs. predicted
    ax.scatter(x, y, color=color, alpha=0.5, label=f'{dataset_label} Data')
    
    # Plot the regression line
    ax.plot(x_line, predictions.predicted_mean, color=color, linewidth=2, label='Best-fit line')
    
    # Highlight points outside the prediction confidence interval
    #for i, (x_point, y_point, lb_pred, ub_pred) in enumerate(zip(x, y, lower_bound_prediction, upper_bound_prediction)):
        #if y_point > ub_pred or y_point < lb_pred:
            #ax.annotate(idx[i], (x_point, y_point), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=12)
    
    # Define the text for the annotation
    eq_text = (
        f"y = {intercept:.3f} (±{intercept_se:.3f}) + "
        f"{slope:.3f} (±{slope_se:.3f}) × x"
        f"\nR² = {r_squared:.3f}"
    )
    
    # Place annotation inside the plot
    ax.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, backgroundcolor='white',
                verticalalignment='top')
    # plt.plot([min(x_line)-1,max(predictions.predicted_mean)+2],
    #          [min(x_line)-1,max(predictions.predicted_mean)+2], 
    #          color='black', linestyle='--')
    
    # Adjust axes to start at (0, 0)
    ax.set_xlim(0, max(x.max(), y.max()) + 7)
    ax.set_ylim(0, max(x.max(), y.max()) + 7)
    ax.set_xlabel('True pAffinity', fontsize=12)
    ax.set_ylabel('Predicted pAffinity', fontsize=12)

    ax.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f"True_vs_Predicted_{dataset_label}_set.png"), dpi=300)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

###############################
#Save HTML
###############################
def plot_true_vs_predicted_plotly(
    true_values,
    predicted_values,
    output_dir,
    idx,
    dataset_label='Training',
    color='blue',
):
    # Ensure inputs are numpy arrays
    x = np.array(true_values, dtype=float)
    y = np.array(predicted_values, dtype=float)

    # Add a constant term for intercept
    X = sm.add_constant(x)

    # Fit the regression model: y = intercept + slope*x
    model = sm.OLS(y, X).fit()

    # Generate points for the regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    X_line = sm.add_constant(x_line)
    predictions = model.get_prediction(X_line)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
    lower_bound_regression = pred_summary["mean_ci_lower"]
    upper_bound_regression = pred_summary["mean_ci_upper"]

    # --- Confidence Interval Around Predictions ---
    residuals = y - x
    std_res = np.std(residuals)
    n = len(x)
    standard_error = std_res * np.sqrt(
        1 + 1/n + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2)
    )
    t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
    y_fit = model.predict(X)
    upper_bound_prediction = y_fit + t_val * standard_error
    lower_bound_prediction = y_fit - t_val * standard_error

    # Create the interactive Plotly figure
    fig = go.Figure()

    # Scatter plot of true vs. predicted
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        text=[str(i) for i in idx],  
        textposition="top center", 
        hoverinfo='text',
        marker=dict(size=8, color=color),
        name=f'{dataset_label} Data'
    ))

    # Regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=predictions.predicted_mean,
        mode='lines',
        line=dict(color='blue'),
        name='Best-fit Line',
        hoverinfo='skip'
    ))

    # Confidence intervals (regression line)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([upper_bound_regression, lower_bound_regression[::-1]]),
        fill='toself',
        fillcolor='rgba(173,216,230,0.2)',  # Light blue
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI (Regression Line)',
        hoverinfo='skip'
    ))

    # Confidence intervals (predictions)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper_bound_prediction, lower_bound_prediction[::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',  # Light red
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI (Predictions)',
        hoverinfo='skip'
    ))

    # Highlight points outside the prediction CI
    for i, (x_point, y_point, lb_pred, ub_pred) in enumerate(zip(x, y, lower_bound_prediction, upper_bound_prediction)):
        if y_point > ub_pred or y_point < lb_pred:
            fig.add_trace(go.Scatter(
                x=[x_point],
                y=[y_point],
                mode='markers+text',
                marker=dict(size=10, color=color),
                text=[str(idx[i])],
                textposition='top center',
                name='Outliers'
            ))

    # Set axis labels and layout
    fig.update_layout(
        title="True vs Predicted pAffinity",
        xaxis_title="True pAffinity",
        yaxis_title="Predicted pAffinity",
        xaxis=dict(range=[0, x.max() + 1]),  
        yaxis=dict(range=[0, y.max() + 1]),  
        template="plotly_white",
        showlegend=True
    )

    # Save the interactive plot as an HTML file
    output_path = os.path.join(output_dir, f"True_vs_Predicted_{dataset_label}_set.html")
    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")

    return fig

def external_set_prediction(model, loader, device, output_dir):
    #External Test set Prediction
    model.eval()
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.u)
            preds.append(out.cpu().numpy())
            
    predictions = np.concatenate(preds, axis=0)
    preds_df = pd.DataFrame(predictions)
    preds_df.to_excel(os.path.join(output_dir,"Predicted_values.xlsx"))
    
    return 

def plot_residuals_histogram_with_density(true_values, predicted_values, output_dir):
    residuals = np.array(true_values) - np.array(predicted_values)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Residuals with Density Curve')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "Residuals_Histogram_Density.png"), dpi=300)
    plt.show()

# ------------------------------
# 6. Main Execution
# ------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_file_path = "C:/Users/Asus/Desktop/shuffled_df.d1.xlsx"
    output_dir = "C:/Users/Asus/Desktop/Cleaned_8irr.new.histogram.added.1.2"
    # Check if file exists
    if not os.path.exists(input_file_path):
        print(f"Dataset file not found in {input_file_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load data from Excel
    # Adjust your path/filename accordingly
    df = pd.read_excel(input_file_path)  # expects columns named "SMILES" and "pAffinity"

    # # Shuffle data
    #df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    #df.to_excel(os.path.join(output_dir,"shuffled_df.xlsx"))
    # Example split into train, validation, test
    train_ratio = 0.7
    val_ratio = 0.15
    #test_ratio = 0.15

    n_samples = len(df)
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n_samples))
    
    train_df = df.iloc[:train_end]

    #number_of_train_rows=train_df
    val_df = df.iloc[train_end:val_end]
    #number_of_val_rows=val_df
    test_df = df.iloc[val_end:]
    #number_of_test_rows=test_df

    # Create datasets
    train_dataset = SMILESDataset(train_df)
    #number_of_train_columns=len(train_dataset)
    val_dataset = SMILESDataset(val_df)
    #number_of_val_columns=len(val_dataset)
    test_dataset = SMILESDataset(test_df)
    #number_of_test_columns=len(test_dataset)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get sample data to see input feature dimensions
    sample_data = train_dataset[0]
    num_node_features = sample_data.x.shape[1]  # e.g., 5 from featurize_atom
    num_edge_features = sample_data.edge_attr.shape[1] if sample_data.edge_attr.shape[0] > 0 else 6
    num_global_features = sample_data.u.shape[1]  # e.g., 2 for (mw, tpsa)
    #print(num_node_features+num_edge_features+num_global_features)
    # Define model
    model = EdgeNet(in_channels=num_node_features,
                    edge_in_channels=num_edge_features,
                    global_in_channels=num_global_features,
                    hidden_channels=64,
                    num_layers=4).to(device)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Training
    #epochs = 500
    #best_val_loss = float('inf')
    #PATIENCE = 500
    #for epoch in range(1, epochs+1):
        #train_loss = train_one_epoch(model, train_loader, optimizer, device)
        #val_loss, val_r2, _, _, _, _ = evaluate(model, val_loader, device)
        #scheduler.step(val_loss)
        
        #print(f"Epoch [{epoch}/{epochs}] - "
              #f"Train Loss: {train_loss:.4f}, "
              #f"Val Loss: {val_loss:.4f}, "
              #f"Val R2: {val_r2:.4f}")

        # Early stopping
        #if val_loss < best_val_loss:
            #best_val_loss = val_loss
            #torch.save(model.state_dict(), 'best_model.d2.pt')
            #patience_counter = 0
        #else:
            #patience_counter += 1
            #if patience_counter >= PATIENCE:
                #print("Early stopping triggered.")
                #break

    # Load best model
    model.load_state_dict(torch.load("best_model.D1.pt"))
    # Final evaluation on test set
    test_loss, test_r2, test_slope, test_intercept, preds, trues = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_loss:.4f}, Test R2: {test_r2:.4f}, "
          f"Slope: {test_slope:.4f}, Intercept: {test_intercept:.4f}")
    
    # # #External set Prediction
    # external_set_prediction(model, test_loader, device, output_dir)
    
    # 7. Residual Analysis
    plot_residuals_vs_predicted(preds, trues, test_idx, output_dir)
    normal_probability_plot(preds, trues, output_dir)

    # Approximate Hat matrix (leverage) for the test set 
    # We'll use the GNN's final hidden features for each graph as 'X'
    # For demonstration, we extract them by a helper function
    def extract_graph_features(loader):
        model.eval()
        outputs = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                # Pass data through GNN up to before final layer
                # We'll replicate the logic in the forward pass but omit final linear layer
                x = data.x
                for i, (conv, bn) in enumerate(zip(model.convs, model.bns)):
                    x = conv(x, data.edge_index, data.edge_attr)
                    x = bn(x)
                    x = F.relu(x)
                # Global pooling
                x = global_max_pool(x, data.batch)
                # Concat global features
                x = torch.cat([x, data.u], dim=-1)
                outputs.append(x.cpu().numpy())
        return np.vstack(outputs)

    # Extract approximate "design matrix" from test loader
    avg_loss, r2, slope, intercept, test_pred, test_true = evaluate(model, test_loader, device)
    test_features = extract_graph_features(test_loader)
    leverages = compute_hat_matrix_approx(test_features,test_pred, test_true, test_idx)
    #williams_plot(preds_test, trues_test, leverages,number_of_test_rows,number_of_test_columns)
    avg_loss, r2, slope, intercept, validation_pred, validation_true = evaluate(model, val_loader, device)
    val_features = extract_graph_features(val_loader)
    leverages = compute_hat_matrix_approx(val_features,validation_pred, validation_true, val_idx)
    #williams_plot(preds_val, trues_val, leverages,number_of_val_rows,number_of_val_columns)
    avg_loss, r2, slope, intercept, training_pred, training_true = evaluate(model, train_loader, device)
    train_features = extract_graph_features(train_loader)
    leverages = compute_hat_matrix_approx(train_features,training_pred, training_true, train_idx)
    #williams_plot(preds_train, trues_train, leverages,number_of_train_rows,number_of_train_columns)
    williams_plot(train_features,
                  val_features, 
                  test_features, 
                  training_pred, 
                  validation_pred, 
                  test_pred, 
                  training_true, 
                  validation_true,
                  test_true, 
                  train_idx, 
                  val_idx, 
                  test_idx,
                  output_dir)
 
    
    # ---------------------
    # Generate the plots
    # ---------------------

    
    fig_train, ax_train = plot_true_vs_predicted(
        training_true,
        training_pred,
        output_dir,
        train_idx,
        dataset_label='Training',
        color='blue'
    )

    fig_val, ax_val = plot_true_vs_predicted(
        validation_true,
        validation_pred,
        output_dir,
        val_idx,
        dataset_label='Validation',
        color='green'
    )

    fig_test, ax_test = plot_true_vs_predicted(
        test_true,
        test_pred,
        output_dir,
        test_idx,
        dataset_label='Test',
        color='brown'
    )
    
    plot_true_vs_predicted_plotly(
        training_true,
        training_pred,
        output_dir,
        train_idx,
        dataset_label='Training',
        color='blue'
    )
    
    plot_true_vs_predicted_plotly(
        validation_true,
        validation_pred,
        output_dir,
        val_idx,
        dataset_label='Validation',
        color='green'
    )
    
    plot_true_vs_predicted_plotly(
         test_true,
         test_pred,
         output_dir,
         test_idx,
         dataset_label='Test',
         color='brown'
    )



    # Show all plots
    plt.show()
    # Evaluate statistics for train, val, and test
    results = evaluate_statistics_for_datasets(model, train_loader, val_loader, test_loader, device)
    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(output_dir, "Results.xlsx"))
    print("The code has completed successfully!")

            

if __name__ == "__main__":
    main()



