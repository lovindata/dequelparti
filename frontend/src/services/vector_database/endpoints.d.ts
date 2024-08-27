export interface paths {
  "/artifacts/vector-database.json": {
    get: operations["artifacts_vector_database_json_get"];
  };
}

export interface components {
  schemas: {
    IdVo: string;
    DecodedEmbeddingVo: string;
    VectorEmbeddingVo: number[];
    LabelVo: string;
    EmbeddingRowVo: {
      id: components["schemas"]["IdVo"];
      decoded_embedding: components["schemas"]["DecodedEmbeddingVo"];
      vector_embedding: components["schemas"]["VectorEmbeddingVo"];
      label: components["schemas"]["LabelVo"];
    };
    EmbeddingTableVo: components["schemas"]["EmbeddingRowVo"][];
  };
}

export interface operations {
  artifacts_vector_database_json_get: {
    responses: {
      200: {
        content: {
          "application/json": components["schemas"]["EmbeddingTableVo"];
        };
      };
    };
  };
}
