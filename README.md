# Image Search by Similarity (by A.Dorcival & G.Faure)

This project was made in the frame of the course "Tools for Data Analysis". 

Search by similarity goal is to find in a dataset the closest item compared to a given query item. 

This Project is about finding similar image given a query image, to do so we turn the images into their colorimetry graph which are 3 vectors of 256 dimensions or one vector of 768 dimension. then we can search for the closest vector from the images of the dataset;

in this case we have 2 datasets : 

* https://www.kaggle.com/adityajn105/flickr8k
* https://www.kaggle.com/adityajn105/flickr30k

We can make the search using brute force and using different distance such as euclidian, intersection ,or Chi-2.

```python
def distance(data,query,dist):
    res = np.zeros((len(data),), dtype=np.float32)
    if(dist=='inter'):
        res=np.sum(np.min([data,query]))
    elif(dist=='chi'):
        res=np.sum(((data-query)**2)/(data+query))
    elif(dist=='euclide'):
        res = np.sqrt(np.sum((data - query)**2, axis=1)) 
    return res
```

We can also use other method to make the research faster: 

* [Simple Dimension reduction](#Simple-dimension-reduction)
* [PCA (Principal Component Analysis)](#PCA)
* [LSH (Locality Sensitive Hashing)](#LSH)

## Simple dimension reduction

This dimension reduction is base on the concatenation of the color histogram, instead of having a dimension for each value of each color we can divide them into groups of 2,4,8,16,32,64,152 color values, giving for each a dimension of 384,192,96,48,24,12.

```python 
        img=cv.imread(f'{dir}{varName}/{file}' ,cv.IMREAD_COLOR)
        red_histo = cv.calcHist([img], [0], None, [k], [0,256])
        green_histo = cv.calcHist([img], [1], None, [k], [0,256])
        blue_histo = cv.calcHist([img], [2], None, [k], [0,256])
        cv.normalize(green_histo,green_histo)
        cv.normalize(red_histo,red_histo)
        cv.normalize(blue_histo,blue_histo)
```

in this case the variable **k** is showing how many color values there is in each dimension

## PCA

PCA "***is the process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest.***" (Wikipedia)

The goal of PCA is to make the query faster by reducing the space dimension of the data vectors, but it is way more efficient than a simple dimension reduction, because it is only keeping the main component of the vectorial space. 

We use the library ScikitLearn to make the PCA, it generate for the dataset the the best principals components and we can select the k greater to fit our query vectors to the reduced dataset.

```python 
    pca=PCA(n_components=K)
    data=pca.fit_transform(data)
    query=np.subtract(query,pca.mean_)
    query=np.dot(query,pca.components_.T)
```

## LSH

LSH "***is an algorithmic technique that hashes similar input items into the same "buckets" with high probability.***"(Wikipedia)

the goal of LSH is to reduce the query time by using hashing function to put close data in *buckets* and then make a brute force query in the bucket corresponding to the query.

To make this hashing function we use a tree like structure that enable to quickly find the corresponding bucket.

we can either add a data to a bucket or find the corresponding bucket from a vector. ***Look for the lsh.py file for the code***

