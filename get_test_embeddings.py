from datasets import load_dataset
import pandas as pd
import numpy as np
import datetime

ds = load_dataset( "MongoDB/tech-news-embeddings", streaming=True )

metadata = pd.DataFrame( columns=[ "title", "description", "company", "publishedAt", "intTime" ] )
data = np.zeros( ( 100_000, 256 ) )


for i in ds[ "train" ].take( 100_000 ):
    data[ len( metadata ) ] = i[ "embedding" ]
    print( len( metadata ) )
    metadata.loc[ len( metadata ) ] = [ i["title"], i[ "description" ], i[ "companyName" ], i[ "published_at" ], int( datetime.datetime.strptime( i[ "published_at" ], "%Y-%m-%d %H:%M:%S" ).timestamp() ) ]


print( metadata )
print( data )

metadata.to_csv( "metadata.csv" )
np.save( "embeddings.npy", data )
