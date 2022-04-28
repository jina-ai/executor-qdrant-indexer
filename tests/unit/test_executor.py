import os
import time

import pytest
from docarray.array.qdrant import DocumentArrayQdrant
from docarray import Document, DocumentArray

import numpy as np

from executor import QdrantIndexer

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.array([1, 0, 0, 0])),
            Document(id='doc2', embedding=np.array([0, 1, 0, 0])),
            Document(id='doc3', embedding=np.array([0, 0, 1, 0])),
            Document(id='doc4', embedding=np.array([0, 0, 0, 1])),
            Document(id='doc5', embedding=np.array([1, 0, 1, 0])),
            Document(id='doc6', embedding=np.array([0, 1, 0, 1])),
        ]
    )


@pytest.fixture()
def docker_compose():
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . down --remove-orphans"
    )


def test_init(docker_compose):
    qindex = QdrantIndexer(collection_name='test', port=6333, distance='euclidean')

    assert isinstance(qindex._index, DocumentArrayQdrant)
    assert qindex._index.collection_name == 'test'
    assert qindex._index._config.port == 6333


def test_index(docs):
    qindex = QdrantIndexer(collection_name='test', port=6333, distance='euclidean')
    qindex.index(docs)

    assert len(qindex._index) == len(docs)


def test_delete(docs):
    qindex = QdrantIndexer(collection_name='test', port=6333, distance='euclidean')
    qindex.index(docs)

    qindex.delete({'ids': [1, 2, 3]})
    assert len(qindex._index) == len(docs) - 3
