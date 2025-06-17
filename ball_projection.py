from sklearn.decomposition import PCA
import umap
import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds = np.load( "tech_news_embeddings.npy" )
print( "PCA" )
pca = PCA( n_components=3 )
data = pca.fit_transform( ds )

print( "UMAP" )
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, unique=True)
umap_data = reducer.fit_transform(ds)
print( "HBDSCAN" )
clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=10)
hdbscan_labels = clusterer.fit_predict(umap_data)

print( data )

print( "Normalize" )

norm_data = data.copy()
for i in range( data.shape[ 0 ] ):
    norm_data[ i ] = data[ i ] / np.linalg.norm( data[ i ] )
print( norm_data )
print( np.linalg.norm( norm_data ) )
print( np.linalg.norm( data ) )
print( np.linalg.norm( data ).shape )
print( data.shape )
print( np.linalg.norm( data, axis = 1 ) )
print( np.linalg.norm( data, axis = 1 ).shape )


fig = plt.figure()
# ax = fig.add_subplot( projection = "3d" )
norm_ax = fig.add_subplot( projection="3d" )


clusters = np.unique( hdbscan_labels )
for cluster in clusters:
    if cluster == -1:
        continue
    idxs = np.where( hdbscan_labels == cluster )[ 0 ]
    np.random.shuffle( idxs )
    idxs = idxs[ :idxs.shape[ 0 ] // 2 ]
 #   ax.scatter( data[ idxs ][:, 0 ], data[ idxs ][ :,1 ], data[ idxs ][ :,2 ] )
    norm_ax.scatter( norm_data[ idxs ][:, 0 ], norm_data[ idxs ][ :,1 ], norm_data[ idxs ][ :,2 ] )


plt.show()
