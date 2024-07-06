# dequelparti

- Fix limited vocabulary issue
  - Keep all words (raw + lemmatized) in PDFs -> Raw vocabulary
  - Do not change the json lemmatized programs
  - Create an association hash table to convert raw to lemmatized -> Lemmatized vocabulary
- Fix issue with LCS algorithm results not good enough ("expulser migrant")
  - Word embeddings for lemmatized words
  - Cosine similarity between user input and aggregate window sliding
