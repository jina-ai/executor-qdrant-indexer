import pytest
from executor import QdrantIndexer


def test_init():

    qindex = QdrantIndexer(collection_name='test', port=6333, distance='euclidean')

    assert qindex._index.storage == 'qdrant'
    assert qindex._index.collection_name == 'test'
    assert qindex._index.port == 6333
    assert qindex._index.distance == 'euclidean'
