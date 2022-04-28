import os
import time

import pytest
from docarray.array.qdrant import DocumentArrayQdrant

from executor import QdrantIndexer

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


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
