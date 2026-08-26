class EmbeddedVectorHnswNearestNeighborIndexClient:
    def query_vector_collection(self, query_embedding_dimension=1536, top_k=5, collection_size=50000):
        return {
            'index_query_id': 'chr_hnsw_5519',
            'dimension': query_embedding_dimension,
            'top_k': top_k,
            'collection_vectors_indexed': collection_size,
            'hnsw_graph_m_parameter': 16,
            'hnsw_ef_construction': 200,
            'recall_rate_at_k_pct': 99.4,
            'query_latency_microseconds': 850
        }
