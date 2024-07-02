# dequelparti

- Fix limited vocabulary issue
  - Keep all words (raw + lemmatized) in PDFs -> Raw vocabulary
  - Do not change the json lemmatized programs
  - Create an association hash table to convert raw to lemmatized -> Lemmatized vocabulary
- Fix issue with LCS algorithm results not good enough ("expulser migrant")
  - Upper triangular matrix lemmatized vocabulary similarity using Spacy for the lemmatized vocabulary
  - Implement "Smith-Waterman" https://youtu.be/lu9ScxSejSE + The match value is equal to the Spacy similarity score
