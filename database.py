
import abc
import sqlite3

import numpy as np
import pandas as pd

from pymongo import MongoClient
from tqdm import tqdm


def chunks(lst, n):
    """Chunks.

    Parameters
    ----------
    lst : list or dict
        The list or dictionary to return chunks from.
    n : int
        The number of records in each chunk.

    """
    if isinstance(lst, list):
        for i in tqdm(range(0, len(lst), n)):
            yield lst[i:i + n]
    elif isinstance(lst, dict):
        keys = list(lst.keys())
        for i in tqdm(range(0, len(keys), n)):
            yield {k: lst[k] for k in keys[i:i + n]}


class BaseWordEmbeddingDB(abc.ABC):

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def write_embeddings(self):
        pass

    @abc.abstractmethod
    def read_embeddings(self):
        pass


class WordEmbeddingSQLiteDB(BaseWordEmbeddingDB):
    """Word embedding SQLite DB.

    Parameters
    ----------
    db : str
        Name of the database.

    Attributes
    ----------
    db : str
        Name of the database.
    conn : sqlite3.Connection instance
        A SQLite database connection.

    """
    def __init__(self, db):
        self.name = db

    def __enter__(self):
        self.conn = sqlite3.connect(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def write_embeddings(self, embedding, table, chunk_size=100000):
        """Write embeddings.

        Parameters
        ----------
        embedding : gensim.models.KeyedVectors instance
            A KeyedVectors instance with the word embeddings.
        table : str
            Name of the database table.
        chunk_size : int
            Chunk size to be used when writting to database. Default is
            100,000 records.

        """
        word2vec = {}
        for k in tqdm(embedding.vocab.keys()):
            word2vec[k] = embedding[k]

        for c in chunks(word2vec,  chunk_size):
            df = pd.DataFrame.from_dict(c, orient='index')
            df.to_sql(table, self.conn, if_exists='append',
                      index_label='word')

    def read_embeddings(self, words, table):
        """Read embeddings.

        Parameters
        ----------
        words : list of str
            A list of words for which word embeddings will be fetched.
        table : str
            Name of database table to read word embeddings from.

        Returns
        -------
        word_vec : dict
            The word embeddings for `words` in the format {str: ndarray}.

        """
        placeholder = ','.join(['?'] * len(words))
        query = f"""SELECT * 
                    FROM {table} 
                    WHERE word IN ({placeholder})"""
        df = pd.read_sql(query, self.conn, params=tuple(words),
                         index_col='word')
        return df.T.to_dict(orient='list')


class WordEmbeddingMongoDB(BaseWordEmbeddingDB):
    """Word embedding MongoDB.

    Parameters
    ----------
    db : str
        Name of the database.
    host : str
        Hostname.
    port : int
        Port number.

    Attributes
    ----------
    db : str
        Name of the database.
    host : str
        Hostname.
    port : int
        Port number.
    client : pymongo.mongo_client.MongoClient instance
        Client for a MongoDB instance.

    """
    def __init__(self, db, host, port):
        self.db = db
        self.host = host
        self.port = port

    def __enter__(self):
        self.client = MongoClient(host=self.host, port=self.port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def write_embeddings(self, embedding, collection, chunk_size):
        """Write embeddings.

        Parameters
        ----------
        embedding : gensim.models.KeyedVectors instance
            A KeyedVectors instance with the word embeddings.
        table : str
            Name of the database table.
        chunk_size : int
            Chunk size to be used when writting to database. Default is
            100,000 records.

        """
        chunk = []
        for word in tqdm(embedding.vocab, desc=f'Writing data to '
                                               f'{self.db}.'
                                               f'{collection}'):
            chunk.append({'word': word,
                          'vec': embedding[word].tolist()})
            if len(chunk) == chunk_size:
                print(f'Writing chunk to {self.db}.{collection}...')
                self.client[self.db][collection].insert_many(chunk)
                chunk = []
        if len(chunk) > 0:
            self.client[self.db][collection].insert_many(chunk)

    def read_embeddings(self, words, collection):
        """Read embeddings.

        Parameters
        ----------
        words : list of str
            A list of words for which word embeddings will be fetched.
        collection : str
            Name of collection to read word embeddings from.

        Returns
        -------
        word_vec : dict
            The word embeddings for `words` in the format {str: ndarray}.

        """
        word_vec = {}
        cursor = self.client[self.db][collection].find({'word':
                                                            {'$in': words}})
        try:
            for res in cursor:
                word_vec[res['word']] = np.array(res['vec'])
        finally:
            cursor.close()
        return word_vec
