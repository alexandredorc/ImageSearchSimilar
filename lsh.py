import numpy as np

class Coord:
    def __init__(self,level):
        self.nodes=[]
        self.coord=[]
        self.level=level

    def add(self,data,index):#si le level est égale à la distance alors 
        if(len(data)==self.level):
            self.nodes.append(index)
        else:
            for i,cor in enumerate(self.coord):
                if(cor==data[self.level]):      #(0,1) -->   [0]-->[1]
                    self.nodes[i].add(data,index) #0-->1-->index
                    return False
            self.coord.append(data[self.level])
            self.nodes.append(Coord(self.level+1))
            self.nodes[len(self.nodes)-1].add(data,index)
    
            
    def find(self,coor,level=0):
        if(len(coor)==self.level):
            return(self.nodes)
        for i,cor in enumerate(self.coord):
            if cor==coor[level]:
                return self.nodes[i].find(coor,level+1)

def hash_func(data,a,b,W):
    return np.floor((np.dot(data,a)+b)/W)

def gen_lsh_sign(size):
    a=np.random.normal(0, 1,(size))
    a=a/np.linalg.norm(a)
    b=np.random.normal(0, 1,(1))
    return a, b
        

def LSH(image,query,nb_projection,W):
    values=[]
    coor=[]
    project_im=[]    
    project_q=[]
    #application des fonction de hachage sur les images et les queries
    for p in range(nb_projection):
        a,b=gen_lsh_sign(np.shape(image)[1])
        project_im.append(hash_func(image,a,b,W))
        project_q.append(hash_func(query,a,b,W))
    H_images=np.array(project_im).T  
    H_query=np.array(project_q).T
    #creation des matrices values et coordonées
    tree=Coord(0)
    
    for img in range(np.shape(image)[0]):
        
        tree.add(H_images[img],img)
    res=[]
    for i,q in enumerate(H_query):
        res.append(tree.find(q))
    return res
                
    
