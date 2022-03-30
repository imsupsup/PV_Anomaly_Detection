from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def plot_heat(arr):
    plt.pcolor(arr)
    plt.colorbar()
    plt.show()

def flatten_avg(arr):
    avg_arr = np.average(arr,axis =1)
    return avg_arr.reshape(len(avg_arr),-1)

def flatten(arr): 
    return arr.reshape(len(arr),-1)


def scree_plot(pca):
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_ 
    
    ax = plt.subplot()
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals, color = ['#00da75', '#f1c40f',  '#ff6f15', '#3498db']) # Bar plot
    ax.plot(ind, cumvals, color = '#c0392b') # Line plot 
    
    for i in range(num_components): #라벨링(바 위에 텍스트(annotation) 쓰기)
        ax.annotate(r"%s" % ((str(vals[i]*100)[:3])), (ind[i], vals[i]), va = "bottom", ha = "center", fontsize = 13)
     
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance")
    plt.title('Scree plot')


def visualize_PCA(vector,bell,dim=2):
    # PCA로 latent vector transformation(차원축소)
    pca = PCA(n_components = dim)
    pca_Component = pca.fit_transform(flatten_avg(vector))
    pca_bell = [pca_Component[bell], pca_Component[bell ==False]]

    # 
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = ["Bell", "Non-Bell"]
    colors = ['b', 'r']

    for i, color in enumerate(colors):
        ax.scatter(pca_bell[i][:,0],pca_bell[i][:,1],c=color,s=50)
        ax.grid()

    ax.legend(targets)
    ax.grid()
    return (pca_Component,pca)
    

def visualize_PCA_hidden(vector,bell,dim = 2):
    # PCA로 latent vector transformation(차원축소)
    pca = PCA(n_components = dim)
    pca_Component = pca.fit_transform(vector)
    pca_bell = [pca_Component[bell], pca_Component[bell ==False]]

    # 
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = ["Bell", "Non-Bell"]
    colors = ['b', 'r']

    for i, color in enumerate(colors):
        ax.scatter(pca_bell[i][:,0],pca_bell[i][:,1],c=color,s=50)
        ax.grid()

    ax.legend(targets)
    ax.grid()
    return pca_Component, pca


def visualize_PCA_3d(vector,bell,dim = 3):
    # PCA로 latent vector transformation(차원축소)
    pca = PCA(n_components = dim)
    pca_Component = pca.fit_transform(flatten_avg(vector))
    pca_bell = [pca_Component[bell], pca_Component[bell ==False]]

    # 
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1,projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 component PCA', fontsize=20)

    targets = ["Bell", "Non-Bell"]
    colors = ['b', 'r']

    for i, color in enumerate(colors):
        ax.scatter(pca_bell[i][:,0],pca_bell[i][:,1],pca_bell[i][:,2],c=color,s=50)
        ax.grid()

    ax.legend(targets)
    ax.grid()
    return pca_Component, pca


def visualize_PCA_1d(vector,bell,dim = 3):
    # PCA로 latent vector transformation(차원축소)
    pca = PCA(n_components = dim)
    pca_Component = pca.fit_transform(flatten_avg(vector))
    pca_bell = [pca_Component[bell], pca_Component[bell ==False]]

    # 
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_title('1 component PCA', fontsize=20)

    targets = ["Bell", "Non-Bell"]
    colors = ['b', 'r']

    for i, color in enumerate(colors):
        ax.scatter(pca_bell[i][:,0],c=color,s=50)
        ax.grid()

    ax.legend(targets)
    ax.grid()
    return pca_Component, pca