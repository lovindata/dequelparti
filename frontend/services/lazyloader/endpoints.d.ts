export interface paths {
  "/artifacts/vector-database.json": {
    get: operations["artifacts_vector_database_json_get"];
  };
}

export interface components {
  schemas: {
    EmbeddingTableVo: components["schemas"]["EmbeddingRowVo"][];
    EmbeddingRowVo: {
      id: string;
      decoded_embedding: str;
      vector_embedding: number[];
      label: str;
    };
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
