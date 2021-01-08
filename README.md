# nlp-embedding-db

## Purpose
Pre-trained word embeddings are often very large in terms of size and therefore the whole file
might not fit into memory. The purpose of this project is therefore to compare different options for storing word
embeddings and compare the performance of these options in terms of speed.

## Word embeddings
The word embedding used so far is the pre-trained Google News word2vec model [1] which was trained on about 100 billion
words and consists of 300 dimensional vectors for 3 million words. Please refer to [1] for a more detailed description
of the word2vec model. Although this pre-trained model can fit in memory, I think it provides a good start when
exploring and comparing database options for word embeddings.

## Databases
The main task that has been evaluated is to given a list of words, e.g. 
`['hi', 'what', 'you', 'apple', 'cat']` retrieve the associated word vectors in the form
`{'hi': ndarray, 'what': ndarray, 'you': ndarray, 'apple': ndarray, 'cat': ndarry}`. The following storing 
options have so far been considered and compared. 

* MongoDB
* SQLite
* Read in the whole binary file and perform the task in memory

## Results
The main result is presented in the figure below. Please refer to [this](./performance_comparison.ipynb) notebook for a more detailed description regarding 
how the databases were evaluated.

![alt text](./img/comparison_speed.png)

## Reproduce results
Follow these steps in order to reproduce the results:

### Setup
1. Make sure that all the required python packages in `requirement.txt` is installed.
2. Download the pre-trained word2vec model `GoogleNews-vectors-negative300.bin.gz`
   from https://code.google.com/archive/p/word2vec/.
3. Add a configuration file `conf.yaml` to the main folder with the following information:
```yaml
name: "<name-of-db>"
mongodb:
  host: "<hostname-mongodb>"
  port: <port-number-mongodb>
sqlite:
  path: "<path-to-sqlite-db>"
data:
  google_new_vec:
    path: "<path-to-binary-file>"
    name: "<name-of-table-for-storing-word-embedding>"
```

### Create database
To create a database with the word embedding run the script `google_news_vector.py`
by running
```
python google_news_vecs.py '<db_type>'
```
Where `<db_type>` is the database that should be used. Currently 'mongodb' and 'sqlite' are supported.
The script will then write all the content from the binary file to the database using the configuration
information in `conf.yaml`.

If db_type='mongodb' the server needs to be up and running before
running the script.

If db_type='sqlite' the database file will be created in the path given in `conf.yaml`.

### Reading from database
When the database has been created it is possible to run the experiment by running this
[notebook](./performance_comparison.ipynb). The two classes `WordEmbeddingSQLiteDB` and `WordEmbeddingMongoDB`
in `database.py` provide the method `read_embeddings` to make it easy to read the word embeddings
from each of the databases.

__Example MongoDB__
```python
with WordEmbeddingMongoDB(db_name, host, port) as db:
    embeddings = db.read_embeddings(words, table_name)
```

__Example SQLite__
```python
with WordEmbeddingSQLiteDB(f"{conf['sqlite']['path']}/{DB_NAME}.db") as db:
    embeddings = db.read_embeddings(words, table_name)
```

## Reference
[1] word2vec. https://code.google.com/archive/p/word2vec/
