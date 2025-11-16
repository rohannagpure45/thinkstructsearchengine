import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

def get_es():
    cloud_id = os.getenv("ES_CLOUD_ID")
    api_key = os.getenv("ES_API_KEY")
    host = os.getenv("ES_HOST", "http://localhost:9200")
    user = os.getenv("ES_USERNAME", "elastic")
    pwd  = os.getenv("ES_PASSWORD", "changeme")

    if cloud_id and api_key:
        return Elasticsearch(cloud_id=cloud_id, api_key=api_key)
    elif api_key:
        return Elasticsearch(hosts=[host], api_key=api_key)
    else:
        return Elasticsearch(hosts=[host], basic_auth=(user, pwd))
