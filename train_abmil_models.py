import argparse
import random
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from torch.nn.functional import softmax

from utils import device, MultiClassificationModel, H5Dataset

# Training settings
parser = argparse.ArgumentParser(description='Training 3-class ABMIL models')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory (path to 20x_dir/40x_dir obtained with TRIDENT)')
parser.add_argument('--results_dir', type=str, default='./eval_metrics')
parser.add_argument('--model_dir', type=str, default='./trained_models', help='path to trained models')
parser.add_argument('--splits_dir', type=str, default='./splits')
parser.add_argument('--n_splits', type=int, default=len(os.listdir('./splits')))
parser.add_argument('--df_path', type=str, default='./dummy_template.csv')
args = parser.parse_args()

model_dir = args.model_dir

def fit(model, train_loader, val_loader, n_split, lr=2e-4, num_epochs=20, patience=5):
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_val_loss = np.inf
    early_stop_steps = 0
    
    #Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        for features, labels, _ in train_loader:
            features, labels = {'features': features.to(device)}, labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        #Validation 
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for features, labels, _ in val_loader:
                features, labels = {'features': features.to(device)}, labels.to(device)
                outputs = model(features)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
            #Incrementing early stopping if validation loss has not improved
            if total_val_loss > min_val_loss:
                early_stop_steps += 1
                print(f'Early stopping steps: {early_stop_steps}')
            else:
                #Resetting early stopping
                early_stop_steps = 0
                min_val_loss = total_val_loss
                #Saving best model
                torch.save(model.state_dict(), os.path.join(model_dir, f's_checkpoint_{model_name}_abmil_{n_split}.pt'))
            if early_stop_steps > patience:
                #Interrupting training loop
                print('Early stopping')
                break
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val loss: {total_val_loss/len(val_loader)}")


def compute_eval_metrics(model, loader):
    all_outputs, all_labels, all_predicted = [], [], []
    
    for features, labels, _ in loader:
        features, labels = {'features': features.to(device)}, labels.to(device)
        outputs = model(features)
        
        predicted = torch.argmax(outputs, dim=1) 

        all_outputs.append(softmax(outputs).cpu().numpy())  
        all_labels.append(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().tolist())
        
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_outputs, multi_class='ovo')
    
    # Compute eval metrics
    accuracy = accuracy_score(y_true=all_labels, y_pred=all_predicted)
    balanced_accuracy = balanced_accuracy_score(y_true=all_labels, y_pred=all_predicted)
    f1 = f1_score(y_true=all_labels, y_pred=all_predicted, average='weighted')
    recall = recall_score(y_true=all_labels, y_pred=all_predicted, average='weighted')
    precision = precision_score(y_true=all_labels, y_pred=all_predicted, average='weighted')
    
    return auc, accuracy, balanced_accuracy, f1, recall, precision


if __name__ == '__main__':
    #Path to csv file with slide names, case ids and split
    df = pd.read_csv(args.df_path)
    results_dir = args.results_dir
    
    #Path to .h5 features extracted with TRIDENT
    features_dir = args.data_root_dir
    path_splits = args.splits_dir
    n_splits = args.n_splits
    batch_size = 64
    SEED = 1234

    # Set deterministic behavior
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_names = ['uni_v2', 'virchow2']
    input_dict = {'uni_v2': 1536, 'virchow2': 2560}
    
    for model_name in model_names:
        print(model_name)
        if os.path.exists(os.path.join(results_dir, f'{model_name}_abmil_test.csv')):
            print(f'Skipping {model_name}')
            continue

        col_names = ['AUC', 'Accuracy', 'Balanced Accuracy', 'F1', 'Recall', 'Precision']
        all_metrics_train = []
        all_metrics_val = []
        all_metrics_test = []
        
        input_dim = input_dict[model_name]
            
        feats_path = os.path.join(features_dir, f'features_{model_name}_vanilla')
        
        #Loading test set
        test_loader = DataLoader(H5Dataset(feats_path, df, "test"),
                                batch_size=1,
                                shuffle=False,
                                worker_init_fn=lambda _: np.random.seed(SEED))

        train_loaders, val_loaders = [], []
        
        #Loading training and validation sets
        for n_split in range(n_splits):
            df_split = pd.read_csv(os.path.join(path_splits, f'splits_{n_split}.csv'))
            train_slides = df_split['train'].to_list()
            val_slides = df_split['val'].to_list()
            
            df.loc[df['slide_id'].isin(train_slides), 'fold_0'] = 'train'
            df.loc[df['slide_id'].isin(val_slides), 'fold_0'] = 'val'
            
            train_loader = DataLoader(H5Dataset(feats_path, df, "train", seed=SEED), batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
            val_loader = DataLoader(H5Dataset(feats_path, df, "val", seed=SEED), batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
            
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        #Training model
        for n_split in range(n_splits):
            if os.path.exists(os.path.join(model_dir, f's_checkpoint_{model_name}_abmil_{n_split}.pt')):
                print(f'Skipping split {n_split}')
                continue
            print(f'Split: {n_split}')
            train_loader = train_loaders[n_split]
            val_loader = val_loaders[n_split]
            
            model = MultiClassificationModel(input_feature_dim=input_dim, dropout=0.2, num_classes=3).to(device)
            fit(model, train_loader, val_loader, n_split)

        #Evaluating previously trained models
        for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
            model = MultiClassificationModel(input_feature_dim=input_dim, dropout=0.2, num_classes=3).to(device)
            model.load_state_dict(torch.load(os.path.join(model_dir, f's_checkpoint_{model_name}_abmil_{i}.pt')))
            model.eval()
            with torch.no_grad():
                metrics_train = compute_eval_metrics(model, train_loader)
                metrics_val = compute_eval_metrics(model, val_loader)
                metrics_test = compute_eval_metrics(model, test_loader)
                all_metrics_train.append(metrics_train)
                all_metrics_val.append(metrics_val)
                all_metrics_test.append(metrics_test)
            
        all_metrics_train = np.array(all_metrics_train)
        all_metrics_val = np.array(all_metrics_val)
        all_metrics_test = np.array(all_metrics_test)

        df_train = pd.DataFrame(all_metrics_train, columns=col_names)
        df_val = pd.DataFrame(all_metrics_val, columns=col_names)
        df_test = pd.DataFrame(all_metrics_test, columns=col_names)

        df_train.to_csv(os.path.join(results_dir, f'{model_name}_abmil_train.csv'), index=False)
        df_val.to_csv(os.path.join(results_dir, f'{model_name}_abmil_val.csv'), index=False)
        df_test.to_csv(os.path.join(results_dir, f'{model_name}_abmil_test.csv'), index=False)