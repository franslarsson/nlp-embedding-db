
import argparse
import yaml

from gensim.models import KeyedVectors

from database import WordEmbeddingMongoDB, WordEmbeddingSQLiteDB


def create_table(conf, db_type):
    data_path = conf['data']['google_news_vec']['path']
    print("Reading data from source...")
    embedding = KeyedVectors.load_word2vec_format(data_path, binary=True)
    db_name = conf['name']
    table_name = conf['data']['google_news_vec']['name']

    if db_type == 'mongodb':
        host = conf['mongodb']['host']
        port = conf['mongodb']['port']
        with WordEmbeddingMongoDB(db_name, host, port) as db:
            print("Start writing data to DB...")
            db.write_embeddings(embedding, table_name, chunk_size=100000)
    elif db_type == 'sqlite':
        sql_db = f"{conf['sqlite']['path']}/{db_name}.db"
        print(f"Using database file: {sql_db}")
        with WordEmbeddingSQLiteDB(sql_db) as db:
            db.write_embeddings(embedding, table_name, chunk_size=100000)
    else:
        raise ValueError(f'`{db_type}` is not a valid option for `db_type`')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database_type', help="Type of data base to use."
                                              "Valid options: "
                                              "{'mongodb', 'sqlite'}")
    args = parser.parse_args()
    with open('db.yaml') as f:
       conf = yaml.safe_load(f)
    print(f"Start script with database_type={args.database_type}")
    create_table(conf, args.database_type)
