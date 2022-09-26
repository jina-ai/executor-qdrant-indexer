import os

import pytest
from docarray.array.qdrant import DocumentArrayQdrant
from docarray import Document, DocumentArray

import numpy as np

from executor import QdrantIndexer
from helper import numeric_operators_qdrant

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '../docker-compose.yml'))


def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.random.rand(128)),
            Document(id='doc2', embedding=np.random.rand(128)),
            Document(id='doc3', embedding=np.random.rand(128)),
            Document(id='doc4', embedding=np.random.rand(128)),
            Document(id='doc5', embedding=np.random.rand(128)),
            Document(id='doc6', embedding=np.random.rand(128)),
        ]
    )


@pytest.fixture
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', text='modified', embedding=np.random.rand(128)),
        ]
    )


def test_init(docker_compose):
    qindex = QdrantIndexer(collection_name='test')

    assert isinstance(qindex._index, DocumentArrayQdrant)
    assert qindex._index.collection_name == 'test'
    assert qindex._index._config.port == 6333


def test_index(docs, docker_compose):
    qindex = QdrantIndexer(collection_name='test')
    qindex.index(docs)

    assert len(qindex._index) == len(docs)


def test_delete(docs, docker_compose):
    qindex = QdrantIndexer(collection_name='test')
    qindex.index(docs)

    ids = ['doc1', 'doc2', 'doc3']
    qindex.delete({'ids': ids})
    assert len(qindex._index) == len(docs) - 3
    for doc_id in ids:
        assert doc_id not in qindex._index


def test_update(docs, update_docs, docker_compose):
    # index docs first
    qindex = QdrantIndexer(collection_name='test')
    qindex.index(docs)
    assert_document_arrays_equal(qindex._index, docs)

    # update first doc
    qindex.update(update_docs)
    assert qindex._index[0].id == 'doc1'
    assert qindex._index['doc1'].text == 'modified'


def test_fill_embeddings(docker_compose):
    qindex = QdrantIndexer(collection_name='test', distance='euclidean', n_dim=1)

    qindex.index(DocumentArray([Document(id='a', embedding=np.array([1]))]))
    search_docs = DocumentArray([Document(id='a')])
    qindex.fill_embedding(search_docs)
    assert search_docs['a'].embedding is not None
    assert (search_docs['a'].embedding == np.array([1])).all()

    with pytest.raises(KeyError, match='b'):
        qindex.fill_embedding(DocumentArray([Document(id='b')]))


def test_filter(docker_compose):
    docs = DocumentArray.empty(5)
    docs[0].text = 'hello'
    docs[1].text = 'world'
    docs[2].tags['x'] = 0.3
    docs[2].tags['y'] = 0.6
    docs[3].tags['x'] = 0.8

    qindex = QdrantIndexer(collection_name='test')
    qindex.index(docs)

    result = qindex.filter(parameters={'query': {'text': {'$eq': 'hello'}}})
    assert len(result) == 1
    assert result[0].text == 'hello'

    result = docs.find({'tags__x': {'$gte': 0.5}})
    assert len(result) == 1
    assert result[0].tags['x'] == 0.8

@pytest.mark.parametrize('limit', [1, 2, 3])
def test_search_with_match_args(docs, limit, docker_compose):
    indexer1 = QdrantIndexer(collection_name='test1', match_args={'limit': limit})
    indexer1.index(docs)
    assert 'limit' in indexer1._match_args.keys()
    assert indexer1._match_args['limit'] == limit

    query = DocumentArray([Document(embedding=np.random.rand(128))])
    indexer1.search(query)

    assert len(query[0].matches) == limit

    docs[0].text = 'hello'
    docs[1].text = 'world'
    docs[2].text = 'hello'

    indexer2 = QdrantIndexer(
        collection_name='test2',
        match_args={'filter': {'text': {'$eq': 'hello'}}, 'limit': 1},
    )
    indexer2.index(docs)

    indexer2.search(query)
    print(query[0].summary())
    assert len(query[0].matches) == 1
    assert query[0].matches[0].text == 'hello'


def test_persistence(docs, docker_compose):
    qindex1 = QdrantIndexer(collection_name='test', distance='euclidean')
    qindex1.index(docs)
    qindex2 = QdrantIndexer(collection_name='test', distance='euclidean')
    assert_document_arrays_equal(qindex2._index, docs)


@pytest.mark.parametrize(
    'metric, metric_name',
    [('euclidean', 'euclid_similarity'), ('cosine', 'cosine_similarity')],
)
def test_search(metric, metric_name, docs, docker_compose):
    # test general/normal case
    indexer = QdrantIndexer(collection_name='test', distance=metric)
    indexer.index(docs)
    query = DocumentArray([Document(embedding=np.random.rand(128)) for _ in range(10)])
    indexer.search(query)

    for doc in query:
        similarities = [t[metric_name].value for t in doc.matches[:, 'scores']]
        assert sorted(similarities, reverse=True) == similarities


def test_clear(docs, docker_compose):
    indexer = QdrantIndexer(collection_name='test')
    indexer.index(docs)
    assert len(indexer._index) == 6
    indexer.clear()
    assert len(indexer._index) == 0


def test_columns(docker_compose):
    n_dim = 3
    indexer = QdrantIndexer(
        collection_name='test', n_dim=n_dim, columns=[('price', 'float')]
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )
    indexer.index(docs)
    assert len(indexer._index) == 10


@pytest.mark.parametrize('operator', list(numeric_operators_qdrant.keys()))
def test_filtering(docker_compose, operator: str):
    n_dim = 3
    indexer = QdrantIndexer(
        collection_name='test', n_dim=n_dim, columns=[('price', 'int')]
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    indexer.index(docs)

    for threshold in [10, 20, 30]:

        filter_ = {'must': [{'key': 'price', 'range': {operator: threshold}}]}

        doc_query = DocumentArray([Document(embedding=np.random.rand(n_dim))])
        indexer.search(doc_query, parameters={'filter': filter_})

        assert len(doc_query[0].matches)

        assert all(
            [
                numeric_operators_qdrant[operator](r.tags['price'], threshold)
                for r in doc_query[0].matches
            ]
        )

