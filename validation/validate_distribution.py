# @title Distribution-level metrics
# deps: torch torchvision numpy scipy scikit-learn pillow tqdm

import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from scipy import linalg
from sklearn.metrics import pairwise_distances

from config import DATASETS_FOLDER_NAME, OUTPUT_FOLDER_NAME
from utils.argparser import parse_distribution_args

args = parse_distribution_args()
# ===============================
# CONFIGURE PATHS HERE
# ===============================
output_dir = os.path.join(OUTPUT_FOLDER_NAME,args.output_dir)
real_path = os.path.join(DATASETS_FOLDER_NAME,args.dataset,"images_512")

import os

def list_folders(path, exclude, exclude_substrings):
    """
    Return a list of immediate subfolder names under `path`,
    excluding:
      - any names in the `exclude` list
      - any names containing one of the substrings in `exclude_substrings`
    """
    if exclude is None:
        exclude = []
    if exclude_substrings is None:
        exclude_substrings = []

    folders = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if not os.path.isdir(full):
            continue

        if name in exclude:
            continue

        # skip if folder name contains any excluded substring
        if any(sub in name for sub in exclude_substrings):
            continue

        folders.append(name)

    return folders

exclude_list = ["old", "depth", "canny","seg", "baseline"]
exclude_substrings = [] #["norotate"]
# TODO ONLY IF NECESSARY
fake_paths = list_folders(output_dir,exclude_list,exclude_substrings)

print(f"found: {len(fake_paths)} folders")

# ===============================
# Loop over fake folders
# ===============================
results_list = []
for fake_path in tqdm(fake_paths, "Calculating Distribution Level"):
    csv_path_name=fake_path
    #print(f"running {fake_path}")
    fake_path=f"{output_dir}/{fake_path}"
    batch_size = 64
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===============================
    # Dataset
    # ===============================

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),  # Inception input
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]
    )

    class ImageFolderDataset(Dataset):
        def __init__(self, root, exts=(".png",".jpg",".jpeg",".bmp",".webp")):
            self.paths = []
            for e in exts:
                self.paths.extend(glob(os.path.join(root, f"**/*{e}"), recursive=True))
            if not self.paths:
                raise ValueError(f"No images found in {root}")
            self.tf = image_transforms

        def __len__(self): return len(self.paths)

        def __getitem__(self, idx):
            with Image.open(self.paths[idx]) as im:
                return self.tf(im.convert("RGB"))

    # ===============================
    # Feature extractors
    # ===============================

    class InceptionPool3(nn.Module):
        def __init__(self):
            super().__init__()
            net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            net.eval()
            for p in net.parameters(): p.requires_grad = False
            self.features = nn.Sequential(
                net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
                nn.MaxPool2d(3,2),
                net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
                nn.MaxPool2d(3,2),
                net.Mixed_5b,net.Mixed_5c,net.Mixed_5d,
                net.Mixed_6a,net.Mixed_6b,net.Mixed_6c,net.Mixed_6d,net.Mixed_6e,
                net.Mixed_7a,net.Mixed_7b,net.Mixed_7c,
                nn.AdaptiveAvgPool2d((1,1))
            )
        def forward(self,x):
            with torch.no_grad():
                return torch.flatten(self.features(x),1)

    def get_features(loader, model, device):
        feats=[]
        for x in loader:
            feats.append(model(x.to(device)).cpu().numpy())
        return np.concatenate(feats,0)

    def get_logits(loader, model, device):
        probs=[]
        with torch.no_grad():
            for x in loader:
                logits = model(x.to(device))
                if isinstance(logits,tuple): logits=logits[0]
                probs.append(F.softmax(logits,dim=1).cpu().numpy())
        return np.concatenate(probs,0)

    def compute_stats(feats): return feats.mean(0), np.cov(feats,rowvar=False)

    def _sqrtm(c1,c2,eps=1e-6):
        c1=c1.copy(); c2=c2.copy()
        c1.flat[::c1.shape[0]+1]+=eps
        c2.flat[::c2.shape[0]+1]+=eps
        cov,info = linalg.sqrtm(c1.dot(c2),disp=False)
        cov = cov.real if np.iscomplexobj(cov) else cov
        return cov

    # ===============================
    # Metrics
    # ===============================

    def inception_score(probs,splits=10):
        N=probs.shape[0]; split=N//splits; scores=[]
        for i in range(splits):
            part=probs[i*split:(i+1)*split]
            py=part.mean(0,keepdims=True)
            kl=part*(np.log(part+1e-10)-np.log(py+1e-10))
            scores.append(np.exp(kl.sum(1).mean()))
        return float(np.mean(scores)), float(np.std(scores))

    def fid(mu1,s1,mu2,s2):
        diff=mu1-mu2; cov=_sqrtm(s1,s2)
        return float(diff.dot(diff)+np.trace(s1+s2-2*cov))

    def kid_poly(X,Y,deg=3,gamma=None,coef0=1.0,subsets=100,subsize=1000):
        rng=np.random.default_rng(123); n=min(len(X),subsize); m=min(len(Y),subsize)
        if gamma is None: gamma=1.0/X.shape[1]
        vals=[]
        for _ in range(subsets):
            Xs=X[rng.choice(len(X),n,False)]; Ys=Y[rng.choice(len(Y),m,False)]
            Kxx=(gamma*Xs@Xs.T+coef0)**deg; Kyy=(gamma*Ys@Ys.T+coef0)**deg
            Kxy=(gamma*Xs@Ys.T+coef0)**deg
            np.fill_diagonal(Kxx,0); np.fill_diagonal(Kyy,0)
            vals.append(Kxx.sum()/(n*(n-1))+Kyy.sum()/(m*(m-1))-2*Kxy.mean())
        vals=np.array(vals)
        return float(vals.mean()),float(vals.std())

    def mmd_rbf(X,Y,sigma='median'):
        Z=np.vstack([X,Y])
        if sigma=='median':
            sample=Z if len(Z)<2000 else Z[np.random.choice(len(Z),2000,False)]
            D=pairwise_distances(sample); sigma=np.median(D[D>0])
        gamma=1/(2*sigma*sigma)
        Kxx=np.exp(-gamma*pairwise_distances(X,X,squared=True))
        Kyy=np.exp(-gamma*pairwise_distances(Y,Y,squared=True))
        Kxy=np.exp(-gamma*pairwise_distances(X,Y,squared=True))
        return float(Kxx.mean()+Kyy.mean()-2*Kxy.mean())

    def _knn_radii(X, k=3, metric='euclidean', eps=1e-8):
        """
        Distance from each point to its k-th nearest neighbor (excluding self).
        Handles small n by using k_eff = min(k, n-1) and floors radii by eps.
        """
        n = len(X)
        if n < 2:
            # No neighbors: return eps to avoid zero-division / empty balls
            return np.full(n, eps, dtype=np.float64)

        k_eff = min(k, n - 1)

        D = pairwise_distances(X, X, metric=metric)
        # include self (0), so the k_eff-th neighbor excluding self is at index k_eff
        # after partitioning by kth=k_eff
        radii = np.partition(D, kth=k_eff, axis=1)[:, k_eff]

        # floor to avoid zero-radius balls due to duplicates
        radii = np.maximum(radii, eps).astype(np.float64)
        return radii

    def precision_recall_density_coverage(real_feats, fake_feats, k=3, metric='euclidean', eps=1e-8):
        """
        Precision/Recall (Kynkäänniemi et al. 2019) and Density/Coverage (Naeem et al. 2020)
        with stability fixes: k_eff, epsilon floor, configurable metric.
        """
        # (optional) L2 normalize features for cosine or Euclidean stability
        # real_feats = real_feats / (np.linalg.norm(real_feats, axis=1, keepdims=True) + 1e-12)
        # fake_feats = fake_feats / (np.linalg.norm(fake_feats, axis=1, keepdims=True) + 1e-12)

        r_r = _knn_radii(real_feats, k=k, metric=metric, eps=eps)
        f_r = _knn_radii(fake_feats, k=k, metric=metric, eps=eps)

        # Precision / Density: fake -> real
        D_fr = pairwise_distances(fake_feats, real_feats, metric=metric)
        # precision: does each fake fall inside at least one real ball?
        precision = (D_fr <= r_r).any(axis=1).mean().item()

        # density: number of real neighbors within each fake's distance to its k-th real neighbor (normalized by k)
        k_eff_r = min(k, max(1, len(real_feats) - 1))
        kth_fr = np.partition(D_fr, kth=k_eff_r, axis=1)[:, k_eff_r]
        density = ((D_fr <= kth_fr[:, None]).sum(axis=1) / max(1, k_eff_r)).mean().item()

        # Recall / Coverage: real -> fake
        D_rf = pairwise_distances(real_feats, fake_feats, metric=metric)
        recall = (D_rf <= f_r).any(axis=1).mean().item()

        # coverage: real point is covered if nearest fake is within its real k-NN radius
        nearest_rf = D_rf.min(axis=1)
        coverage = (nearest_rf <= r_r).mean().item()

        return float(precision), float(recall), float(density), float(coverage)

    # ===============================
    # Run
    # ===============================

    real_ds=ImageFolderDataset(real_path)
    fake_ds=ImageFolderDataset(fake_path)
    real_loader=DataLoader(real_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    fake_loader=DataLoader(fake_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    feat_net=InceptionPool3().to(device).eval()
    cls_net=models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()

    real_feats=get_features(real_loader,feat_net,device)
    fake_feats=get_features(fake_loader,feat_net,device)
    #print("running IS")
    # IS
    fake_probs=get_logits(fake_loader,cls_net,device)
    is_mean,is_std=inception_score(fake_probs)
    #print("running FID")
    # FID
    mu_r,s_r=compute_stats(real_feats); mu_f,s_f=compute_stats(fake_feats)
    fid_val=fid(mu_r,s_r,mu_f,s_f)
    #print("running KID")
    # KID
    kid_mean,kid_std=kid_poly(real_feats,fake_feats)
    #print("running MMD-RBF")
    # MMD-RBF
    mmd_val=mmd_rbf(real_feats,fake_feats)
    #print("running P/R/D/Cover")
    # Precision / Recall / Density / Coverage
    prec,rec,dens,cov=precision_recall_density_coverage(real_feats,fake_feats)

    # ===============================
    # Save results as CSV
    # ===============================
    results = {
        "Folder-Name": csv_path_name,
        "IS_mean": is_mean,
        "IS_std": is_std,
        "FID": fid_val,
        "KID_mean": kid_mean,
        "KID_std": kid_std,
        "MMD_RBF": mmd_val,
        "Precision": prec,
        "Recall": rec,
        "Density": dens,
        "Coverage": cov,
    }

    results_list.append(results)

df = pd.DataFrame(results_list)
csv_path = os.path.join(output_dir, f"distribution.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Saved results to {csv_path}")
