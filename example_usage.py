from client import EmbeddedVectorHnswNearestNeighborIndexClient

def main():
    client = EmbeddedVectorHnswNearestNeighborIndexClient()
    res = client.query_vector_collection(3072, 10, 250000)
    print('Vector Query: ' + res['index_query_id'] + ' (Dim: ' + str(res['dimension']) + ' | Top-' + str(res['top_k']) + ')')
    print('Vectors Indexed: ' + str(res['collection_vectors_indexed']) + ' | Recall@K: ' + str(res['recall_rate_at_k_pct']) + '%')
    print('Latency: ' + str(res['query_latency_microseconds']) + ' μs (HNSW ef: ' + str(res['hnsw_ef_construction']) + ')')

if __name__ == '__main__':
    main()
