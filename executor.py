import inspect

from jina import Executor, requests
from typing import Optional, Dict, List, Tuple, Union
from docarray import DocumentArray
from jina.logging.logger import JinaLogger


class QdrantIndexer(Executor):
    """QdrantIndexer indexes Documents into a Qdrant server using DocumentArray  with `storage='qdrant'`"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6333,
        collection_name: str = 'Persisted',
        distance: str = 'cosine',
        n_dim: int = 128,
        match_args: Optional[Dict] = None,
        ef_construct: Optional[int] = None,
        full_scan_threshold: Optional[int] = None,
        m: Optional[int] = None,
        scroll_batch_size: int = 64,
        serialize_config: Optional[Dict] = None,
        columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None,
        **kwargs,
    ):
        """
        :param host: Hostname of the Qdrant server
        :param port: port of the Qdrant server
        :param collection_name: Qdrant Collection name used for the storage
        :param distance: The distance metric used for the vector index and vector search
        :param n_dim: number of dimensions
        :param match_args: the arguments to `DocumentArray`'s match function
        :param ef_construct: The size of the dynamic list for the nearest neighbors (used during the construction).
            Controls index search speed/build speed tradeoff. Defaults to the default `ef_construct` in the Qdrant
            server.
        :param full_scan_threshold: Minimal amount of points for additional payload-based indexing. Defaults to the
            default `full_scan_threshold` in the Qdrant server.
        :param scroll_batch_size: batch size used when scrolling over the storage.
        :param serialize_config: DocumentArray serialize configuration.
        :param m: The maximum number of connections per element in all layers. Defaults to the default
            `m` in the Qdrant server.
        :param columns: precise columns for the Indexer (used for filtering).
        """
        super().__init__(**kwargs)
        self._match_args = match_args or {}

        self._index = DocumentArray(
            storage='qdrant',
            config={
                'collection_name': collection_name,
                'host': host,
                'port': port,
                'n_dim': n_dim,
                'distance': distance,
                'ef_construct': ef_construct,
                'm': m,
                'scroll_batch_size': scroll_batch_size,
                'full_scan_threshold': full_scan_threshold,
                'serialize_config': serialize_config or {},
                'columns': columns,
            },
        )

        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        """Index new documents
        :param docs: the Documents to index
        """
        self._index.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Dict = {},
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: Dictionary to define the `filter` that you want to use.
        :param kwargs: additional kwargs for the endpoint

        """
        
        match_args = (
                {**self._match_args, **parameters}
                if parameters is not None
                else self._match_args
            )
        match_args = QdrantIndexer._filter_match_params(docs, match_args)
        docs.match(self._index, filter=parameters.get('filter', None), **match_args)


    @staticmethod
    def _filter_match_params(docs, match_args):
        # get only those arguments that exist in .match
        args = set(inspect.getfullargspec(docs.match).args)
        args.discard('self')
        match_args = {k: v for k, v in match_args.items() if k in args}
        return match_args

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters of the request

        Keys accepted:
            - 'ids': List of Document IDs to be deleted
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update existing documents
        :param docs: the Documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/filter')
    def filter(self, parameters: Dict, **kwargs):
        """
        Query documents from the indexer by the filter `query` object in parameters. The `query` object must follow the
        specifications in the `find` method of `DocumentArray` in the docs https://docarray.jina.ai/fundamentals/documentarray/find/#filter-with-query-operators
        :param parameters: parameters of the request
        """
        return self._index.find(parameters['query'])

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """Fill embedding of Documents by id

        :param docs: DocumentArray to be filled with Embeddings from the index
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index"""
        self._index.clear()

    def close(self) -> None:
        super().close()
        del self._index
