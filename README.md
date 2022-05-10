# QdrantIndexer

`QdrantIndexer` indexes Documents into a `DocumentArray`  using `storage='qdrant'`. Underneath, the `DocumentArray`  uses 
 [qdrant](https://github.com/qdrant/qdrant) to store and search Documents efficiently. 
The indexer relies on `DocumentArray` as a client for Qdrant, you can read more about the integration here: 
https://docarray.jina.ai/advanced/document-store/qdrant/

## Setup
`QdrantIndexer` requires a running Qdrant server. Make sure a server is up and running and your indexer is configured 
to use it before starting to index documents. For quick testing, you can run a containerized version locally using 
docker-compose :

```shell
docker-compose -f tests/docker-compose.yml up -d
```

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
from docarray import Document
import numpy as np
	
f = Flow().add(
    uses='jinahub+docker://QdrantIndexer',
    uses_with={
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'collection_name',
        'distance': 'cosine',
        'n_dim': 256,
    }
)

with f:
    f.post('/index', inputs=[Document(embedding=np.random.rand(256)) for _ in range(3)])
```

#### via source code

```python
from jina import Flow
from docarray import Document
	
f = Flow().add(uses='jinahub://QdrantIndexer',
    uses_with={
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'collection_name',
        'distance': 'cosine',
        'n_dim': 256,
    }
)

with f:
    f.post('/index', inputs=[Document(embedding=np.random.rand(256)) for _ in range(3)])
```



## CRUD Operations

You can perform CRUD operations (create, read, update and delete) using the respective endpoints:

- `/index`: Add new data to Qdrant. 
- `/search`: Query the Qdrant index (created in `/index`) with your Documents.
- `/update`: Update Documents in Qdrant.
- `/delete`: Delete Documents in Qdrant.


## Vector Search

The following example shows how to perfom vector search using`f.post(on='/search', inputs=[Document(embedding=np.array([1,1]))])`.


```python
from jina import Flow
from docarray import Document

f = Flow().add(
         uses='jinahub://QdrantIndexer',
         uses_with={'collection_name': 'test', 'n_dim': 2},
     )

with f:
    f.post(
        on='/index',
        inputs=[
            Document(id='a', embedding=np.array([1, 3])),
            Document(id='b', embedding=np.array([1, 1])),
        ],
    )

    docs = f.post(
        on='/search',
        inputs=[Document(embedding=np.array([1, 1]))],
    )

# will print "The ID of the best match of [1,1] is: b"
print('The ID of the best match of [1,1] is: ', docs[0].matches[0].id)
```