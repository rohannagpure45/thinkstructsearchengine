import argparse, os, glob, json, uuid
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

def es_client():
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

def iter_patents(path):
    files = []
    if os.path.isdir(path):
        for ext in ("*.json",):
            files.extend(sorted(glob.glob(os.path.join(path, ext))))
    else:
        files = [path]
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "patents" in data:
                    data = data["patents"]
                if not isinstance(data, list):
                    continue
                for p in data:
                    yield p
            except Exception as e:
                print(f"[WARN] skipping {fp}: {e}")

def normalize_text(x):
    if not x:
        return None
    if isinstance(x, list):
        x = " ".join([seg for seg in x if isinstance(seg, str)])
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSON file or directory")
    ap.add_argument("--index", default=os.getenv("ES_INDEX_NAME", "patents"))
    ap.add_argument("--elser-pipeline", default=None, help="Optional ingest pipeline for ELSER")
    args = ap.parse_args()

    es = es_client()
    index = args.index
    pipeline = args.elser_pipeline

    from elasticsearch.helpers import streaming_bulk

    actions = []
    for p in iter_patents(args.input):
        doc_number = p.get("Document Number") or p.get("doc_number") or str(uuid.uuid4())
        title = normalize_text(p.get("Title"))
        abstract = normalize_text(p.get("Abstract"))
        claims = p.get("Claims") or []
        if isinstance(claims, str):
            claims = [claims]

        if abstract:
            actions.append({
                "_op_type": "index",
                "_index": index,
                "_id": f"{doc_number}#ABSTRACT",
                "_source": {
                    "doc_number": doc_number,
                    "section_type": "abstract",
                    "title": title,
                    "abstract": abstract
                },
                "pipeline": pipeline
            })

        for i, claim_text in enumerate(claims, start=1):
            if not isinstance(claim_text, str):
                continue
            actions.append({
                "_op_type": "index",
                "_index": index,
                "_id": f"{doc_number}#CLAIM#{i}",
                "_source": {
                    "doc_number": doc_number,
                    "section_type": "claim",
                    "title": title,
                    "claims_text": claim_text
                },
                "pipeline": pipeline
            })

    for ok, resp in streaming_bulk(es, actions, expand_action_callback=lambda a: a):
        if not ok:
            print("[WARN] failed action:", resp)

    es.indices.refresh(index=index)
    print("Done ingest.")

if __name__ == "__main__":
    main()
