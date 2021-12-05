from tqdm import tqdm
import numpy as np

#fonction calcule la distance entre un nuage de point et une query
def distance(data,query,dist):
    res = np.zeros((len(data),), dtype=np.float32)
    if(dist=='inter'):
        res=np.sum(np.min([data,query]))
    elif(dist=='chi'):
        res=np.sum(((data-query)**2)/(data+query))
    elif(dist=='euclide'):
        res = np.sqrt(np.sum((data - query)**2, axis=1))
    
    return res

#trouve le point le plus proche pour chaque query
def bruteForce(data,query,dist):
    distances = distance(data, query, dist)
    min_idx = np.argmin(distances)
    return [min_idx], [distances[min_idx]]
   
#trouve tous les points ce trouvant en dessous d'une distance 'rad'
def radiusSearch(images, query,rad ,type):
    loop=tqdm(total=7991,position=0,leave=False)
    index=[]
    for i,image in enumerate(images):
        loop.set_description("Loading...".format(i))
        loop.update(1)
        dist=distance(image,query,type)
        if (dist<rad):
            index.append(i)
    loop.close()
    return index
        
#trouve les K points les plus proches
def KNN(images, query, k,type):
    loop=tqdm(total=7991,position=0,leave=False)
    res=[]
    index=[]
    for i,image in enumerate(images):
        loop.set_description("Loading...".format(i))
        loop.update(1)
        dist=distance(image,query,type)
        if (len(res)<k):
            res.append(dist)
            index.append(i)
        elif(dist<np.max(res)):
            index.pop(np.argmax(res))
            res.pop(np.argmax(res))
            res.append(dist)
            index.append(i)
    loop.close()
    return index

