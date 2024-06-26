import logging
import os
from tqdm import tqdm

from typing import Callable, Union, List
import chromadb
from chromadb import Documents, Embeddings, EmbeddingFunction
import dspy
from BCEmbedding import EmbeddingModel, RerankerModel


# # method 0: calculate scores of sentence pairs
# scores = model.compute_score(sentence_pairs)

# # method 1: rerank passages
# rerank_results = model.rerank(query, passages)
# print(rerank_results)


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model) -> None:
        super().__init__()
        self.embedding_model = embedding_model

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = [self.embedding_model.encode(x).tolist()[0] for x in input]
        return embeddings


def get_collection(texts=None, adding=False, embed_model=None):
    client = chromadb.Client()
    # 也可以采用持久化存储client = chromadb.PersistentClient(path="/Users/yourname/xxxx")

    collection = client.get_or_create_collection(
        name="md_collection", embedding_function=MyEmbeddingFunction(embed_model)
    )

    if adding and texts is not None:

        texts_content = [doc.page_content for doc in texts]

        collection.add(
            documents=texts_content,  # 处理后数据,因为检索，先使用大纲试试
            metadatas=[doc.metadata for doc in texts],
            ids=[str(i) for i in range(len(texts))],
        )

    return collection


# 改成FAISS


class LocalRM(dspy.Retrieve):
    def __init__(self, reranker, embed_func, collection, k=3):
        super().__init__(k=k)
        self.reranker = reranker
        self.embedding_function = embed_func
        self.collection = collection

    def forward(self, query_or_queries):
        """
        returns:
            list of dicts: each dict has keys of 'description', 'snippets' (list of strings which is the whole content of the method statement),
            'title' (the name of the local method statement), 'url' (the dir of the local method statement), "description" ( "no desc" )
        """

        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        collected_results = []
        for query in queries:
            out = self.collection.query(
                query_embeddings=self.embedding_function(query),
                n_results=16,
            )

            passages = out["documents"][0]
            metadatas = out["metadatas"][0]

            rerank_results = self.reranker.rerank(query, passages)
            rerank_ids = rerank_results["rerank_ids"]
            top_k_result = [metadatas[i] for i in rerank_ids[: self.k]]
            collected_results.extend(top_k_result)

        return collected_results
