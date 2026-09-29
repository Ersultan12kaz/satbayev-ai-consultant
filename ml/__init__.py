"""ML/RAG core of the Satbayev University virtual assistant.

Every model in this package is trained by us from scratch on the official
university documents (no external LLM API is required):

* ``langdetect``  - character n-gram language identifier (RU / KZ / EN)
* ``scope``       - university-related vs. off-topic question classifier
* ``embeddings``  - subword LSA encoder + contrastively fine-tuned projection
* ``bm25``        - keyword (sparse) retriever
* ``retriever``   - dense / BM25 / hybrid retrieval
* ``reranker``    - learning-to-rank model on retrieval features
* ``generator``   - grounded answer generator with citations
* ``rag``         - the end-to-end RAG pipeline
"""

__version__ = "1.0.0"
