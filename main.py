import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import append
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
from time import time
from lsh import *
from gen import gen
from brute_force import *

#permet d'afficher les graphiques des inerties par apport au choix de K
def ACP_display(data):
    pca=PCA(n_components=768)
    data=pca.fit_transform(data)
    inertia=pca.explained_variance_ratio_
    y=np.append(np.array([0]),np.cumsum(inertia))
    x=np.arange(769)
    plt.plot(x,y)
    plt.show()  
   
    x=np.arange(768)
    plt.plot(x,inertia)
    plt.show()  

#transforme via l'ACP les données et les queries
def ACP_gen(K,data,query):
    pca=PCA(n_components=K)
    data=pca.fit_transform(data)
    query=np.subtract(query,pca.mean_)
    query=np.dot(query,pca.components_.T)
    return data, query

#permet de générer la vérité terrain sur des données et des queries
def ground_truth(data,query,dist='euclide'):
    res=[]
    for q in query:
        index,distance=bruteForce(data,q,dist)
        res.append(index)
    np.save('ground_truth30k',res)

#fonction qui permet d'afficher la précision pour les différéntes méthodes du projet
def check_precision(data,query,truth,method,K=None,nb_projection=None,nb_tables=None,W=None,hist_count=None,dist='euclide'):
    count=0
    if method=="ACP":
        acp_data,acp_queries=ACP_gen(K,data,query)
        
        loop=tqdm(total=100,position=0,leave=False)
        for i,t in enumerate(truth):
            loop.set_description("Loading...".format(i))
            loop.update(1)
            ori,res=bruteForce(acp_data,acp_queries[i],dist)
            if(t==ori):
                count+=1
        loop.close()
        return count/len(query)


    if method=="LSH":
        min_dist=np.full((len(query)),None)
        best=np.empty((len(query)))
        visit=0
        bucket=[]
        for t in range(nb_tables):
            print("préparation de la table",t+1)
            bucket.append(LSH(data,query,nb_projection,W))
        loop=tqdm(total=nb_tables,position=0,leave=False)
        for t in range(nb_tables):
            loop.set_description("Loading...".format(t))
            loop.update(1)
            for i,q in enumerate(query):
                if bucket[t][i] is None:
                    res,distance= None,None
                else:
                    res,distance=bruteForce(data[bucket[t][i]],q,dist)
                    buck=np.array(bucket[t][i])
                    visit+=len(bucket[t][i])
                    res=buck[res]
                
                    if min_dist[i] is None or distance < min_dist[i]:
                        min_dist[i]=distance
                        best[i]=res
        
            
        loop.close()
        for i,t in enumerate(truth):
            if(t==best[i]):
                count+=1
        return count/len(query),visit/(len(data)*len(query))

    if method=="concatenation":
        N_query=np.load(f'imgGen/queriesRGB{hist_count}.npy')
        N_img=np.load(f'imgGen/imgRGB{hist_count}.npy')
        
        for i,t in enumerate(truth):
            res,distance=bruteForce(N_img,N_query[i],dist)
            if(t==res):
                count+=1
        return count/len(query)

#on réalise LSH en itérant la valeur de W
def study_LSH_W(data,query,truth,nb_proj,nb_table):
    weight=[0.01 * i for i in range(1,30)]
    precision=[]
    visit_ratio=[]
    for W in weight:
        print("W=",W)
        preci,visit=check_precision(data,query,truth,"LSH",nb_projection=nb_table,nb_tables=nb_proj,W=W)
        precision.append(preci)
        visit_ratio.append(visit)
    plt.plot(weight,precision)
    plt.plot(weight,visit_ratio)
    plt.show()

#on réalise LSH en itérant le nombre de projections
def study_LSH_proj(data,query,truth,W,nb_table):
    nb_proj=[1 * i for i in range(1,50)]
    precision=[]
    visit_ratio=[]
    for nb in range(1,75):
        print("nb projection=",nb)
        preci,visit=check_precision(data,query,truth,"LSH",nb_projection=nb,nb_tables=nb_table,W=W)
        precision.append(preci)
        visit_ratio.append(visit)
    plt.plot(nb_proj,precision)
    plt.plot(nb_proj,visit_ratio)
    plt.show()

#on réalise LSH en itérant le nombre de tables
def study_LSH_table(data,query,truth,W,nb_proj):
    nb_tables=[1 * i for i in range(1,30)]
    precision=[]
    visit_ratio=[]
    for nb in range(1,30):
        print("nb tables=",nb)
        preci,visit=check_precision(data,query,truth,"LSH",nb_projection=nb_proj,nb_tables=nb,W=W)
        precision.append(preci)
        visit_ratio.append(visit)
    plt.plot(nb_tables,precision)
    plt.plot(nb_tables,visit_ratio)
    plt.show()    

#on étudie le temps de résolution des données avec l'ACP selon les valeurs de K 
def study_ACP(data,query,truth):
    time_arr=[]
    preci=[]
    K_arr=[1 * i for i in range(1,10)]
    for K in range(1,10):
        ti=time()
        preci.append(check_precision(data,query,truth,"ACP",K=K))
        time_arr.append(time()-ti)
    plt.plot(K_arr,preci)
    plt.plot(K_arr,time_arr)
    plt.show()  

#on étudie le temps de résolution des données la méthode de concatenation pour les puissances de 2
def study_concat(data,query,truth):
    time_arr=[0]
    preci=[0]
    K_arr=[1 * i for i in range(9)]
    for i in range(8):
        it=2**(i+1)
        ti=time()
        preci.append(check_precision(data,query,truth,method="concatenation",hist_count=it))
        time_arr.append(time()-ti)
    plt.plot(K_arr,preci)
    plt.plot(K_arr,time_arr)
    plt.show()  

#génération des données
dir='./Flickr8k/'
images=np.sort(os.listdir(f'{dir}images'))
queries=np.sort(os.listdir(f'{dir}queries'))
if os.path.isdir('./imgGen')==False:
    os.makedirs('./imgGen')
# Appel de fonction de génération pour les images et les query
gen(256,images,'images')
gen(256,queries,'queries')
#gen(128,images,'images')
#gen(128,queries,'queries')
#gen(64,images,'images')
#gen(64,queries,'queries')
#gen(32,images,'images')
#gen(32,queries,'queries')
#gen(16,images,'images')
#gen(16,queries,'queries')
#gen(8,images,'images')
#gen(8,queries,'queries')
#gen(4,images,'images')
#gen(4,queries,'queries')
#gen(2,images,'images')
#gen(2,queries,'queries')
#... ajouter les données manquante

#chargement des queries
name_q=np.load('imgGen/queriesName.npy')
rgb_q=np.load('imgGen/queriesRGB256.npy')

#pour les données avec 30k
'''
name_i=np.load('imgGen/imgName.npy')
rgb_i=np.load('imgGen/imgRGB256.npy')
ground_truth(rgb_i,rgb_q,'euclide')
ground_truth=np.load('ground_truth30k.npy')
'''

#pour les données avec 8k
name_i=np.load('imgGen/imgName.npy')
rgb_i=np.load('imgGen/imgRGB256.npy')
#ground_truth(rgb_i,rgb_q,'euclide')
ground_truth=np.load('ground_truth.npy')

#study_ACP(rgb_i,rgb_q,ground_truth)

#study_concat(rgb_i,rgb_q,ground_truth)

#Etudes pour les 3 différents paramètres
#study_LSH_W(rgb_i,rgb_q,ground_truth,15,15)
#study_LSH_proj(rgb_i,rgb_q,ground_truth,0.15,15)
#study_LSH_table(rgb_i,rgb_q,ground_truth,0.15,15)

#vérification de la bonne utilisation des 
preci,visit=check_precision(rgb_i,rgb_q,ground_truth,"LSH",nb_projection=15,nb_tables=15,W=0.15)
print(preci)
print(visit)