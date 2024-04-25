import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import pickle

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.transforms as transforms
 
# Random state.
RS = 0
 
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

from dnnlib.util import open_url
from resnet9 import ResNet9
 
# We import seaborn to make nice plots.
import seaborn as sns


def get_activations(dl, model):
    pred_arr = []
    label_arr = []
    print('Starting to sample.')
    for batch, label in dl:
        batch = (batch * 255.).to(torch.uint8)
        # ignore labels
        if isinstance(batch, list):
            batch = batch[0]

        batch = batch.cuda()
        if batch.shape[1] == 1:  # if image is gray scale
            batch = batch.repeat(1, 3, 1, 1)
        elif len(batch.shape) == 3:  # if image is gray scale
            batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

        with torch.no_grad():
            pred = model(batch, return_features=True).unsqueeze(-1).unsqueeze(-1)

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        label = label.numpy()
        pred_arr.append(pred)
        label_arr.append(label)
    
    pred_arr = np.concatenate(pred_arr, axis=0)
    label_arr = np.concatenate(label_arr, axis=0)

    return pred_arr, label_arr

def get_activations_2(dl, model):
    pred_arr = []
    label_arr = []
    print('Starting to sample.')
    for batch, label in dl:
        batch = batch.cuda()

        with torch.no_grad():
            # output
            pred = torch.nn.functional.normalize(model(batch))
            # last layer
            for i in range(len(model)):
                batch = model[i](batch)
                if i == len(model)-3:
                    pred = batch.detach()
                    print(pred.shape)

        pred = pred.cpu().numpy()
        label = label.numpy()
        pred_arr.append(pred)
        label_arr.append(label)
    
    pred_arr = np.concatenate(pred_arr, axis=0)
    label_arr = np.concatenate(label_arr, axis=0)

    return pred_arr, label_arr


sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

data_dir = '/bigtemp/fzv6en/kecen0923/DPDM/cifar10/e10/cifar10_32_s_e10_all_lb/sample5000/samples/'
weight_dir = '/bigtemp/fzv6en/kecen0923/DPDM/cifar10/e10/cifar10_32_s_top5_e9.9_lb/trained_cnn_weight.pth'
transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
#dataset = ImageFolder(root=data_dir, transform=transform)
dataset = CIFAR10(root="/bigtemp/fzv6en/datasets/", train=False, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=1024, drop_last=False)

#with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
#        inception_model = pickle.load(f).cuda()
cls_model = ResNet9(10)
cls_model.load_state_dict(torch.load(weight_dir))
cls_model.eval()

#X, y = get_activations(dataloader, inception_model)
X, y = get_activations_2(dataloader, cls_model)

digits_proj = TSNE(random_state=RS).fit_transform(X)
 
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))
 
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    ax.axis('off')
    ax.axis('tight')
    
    return 
    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
 
    return f, ax, sc, txts
 
scatter(digits_proj, y)
plt.savefig('cifar10_ours_d_128_tsne.pdf', dpi=120)
#plt.show()