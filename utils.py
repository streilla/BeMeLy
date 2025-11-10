import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from slide_encoder_models import ABMILSlideEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, n_heads=1, head_dim=512, dropout=0., gated=True, hidden_dim=256, num_classes=3):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim, 
            n_heads=n_heads, 
            head_dim=head_dim, 
            dropout=dropout, 
            gated=gated,
            freeze=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features)
        
        if return_raw_attention:
            return logits, attn
        
        return logits
    
    def extract_embedding(self, x):
        """Return the slide-level embedding before classifier."""
        features = self.feature_encoder(x)  # shape: [batch, embedding_dim]
        return features
    

# Custom dataset
class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split, num_features=512, seed=None):
        self.df = df[df["fold_0"] == split]
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        self.seed = seed
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with h5py.File(os.path.join(self.feats_path, row['slide_id'] + '.h5'), "r") as f:
            features = torch.from_numpy(f["features"][:])

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(self.seed))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(self.seed))  # Oversampling
            features = features[indices]

        label = torch.tensor(row["label"], dtype=torch.long)
        slide = row['slide_id']
        return features, label, slide
