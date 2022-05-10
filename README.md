# executor-qdrant-indexer


`QdrantIndexer` indexes Documents into a `DocumentArray`  using `storage='qdrant'`. Underneath, the `DocumentArray`  uses 
 [qdrant](https://github.com/qdrant/qdrant) to store and search Documents efficiently. 

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://executor-qdrant-indexer')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://executor-qdrant-indexer')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
