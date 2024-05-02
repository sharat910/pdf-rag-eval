# PDF RAG Pipeline

This repo contains code to build and evaluate RAG pipeline. 

Edit the running configs in `main.py` and run using `python main.py` after you install packages `pip install -r requirements.txt` into your python environment.

A basic RAG pipeline consists of the following steps:

## Parsing PDFs
PDF parsing is inherently complex due to its structure and placement of info entities like tables and custom design. This repo utilizes `llama-parse` library to parse PDFs accurately (compared to other libraries like `pypdf` etc.)

The parsed PDFs are stored in markdown formats:
- `./files/parsed_md/nrma.md`
- `./files/parsed_md/allianz.md`

## Chunking
The markdown files are split into chunks using parent-child splitters. This method first splits the document into large chunks and then into small child chunks. The small child chunks are used for embeddings to find close meaning to the question asked but large parent chunks are passed to llms to answer the question which results in a larger context.

## Retrieval
The retrieval step fetches the document chunks relevant to the query ready to be passed on to llm with a prompt.
Implemented 3 retrieval methods:
- bm25 text retrieval on parent chunks
- parent-child vector search retrieval using embeddings (openai vs cohere)
- Ensemble retrieval using both of the above.

## LLMs
Ran the RAG QA pipeline on:
- gpt-3.5-turbo
- llama3 (ollama)

## Evaluation
Evaluated the performance of the rag pipelines using `ragas` library. As a first step, a test set is generated using `gpt-4` with a set of questions and answers on the document: `files/testset.csv`. The questions vary in difficulty and comprehension. Then, RAG pipelines are evaluated using the following metrics:
- context_precision: How much of the context is used to answer the question
- faithfulness: If the answer is present in the context
- answer_relevancy: How relevant the answer is to the question
- context_recall: How well the llm can recall information from the context

## Results
| xpt |   context_precision |   faithfulness |   answer_relevancy |   context_recall | embedding   | retriever_mode   | llm           |
|---:|--------------------:|---------------:|-------------------:|-----------------:|:------------|:-----------------|:--------------|
|  0 |            0.819861 |       0.875    |           0.925629 |         0.966667 | -           | bm25             | ollama        |
|  1 |            0.961111 |       0.866111 |           0.946809 |         1        | openai      | vector           | gpt-3.5-turbo |
|  2 |            0.938889 |       0.95     |           0.954773 |         0.833333 | cohere      | vector           | gpt-3.5-turbo |
|  3 |            0.88171  |       1        |           0.956301 |         1        | openai      | ensemble         | gpt-3.5-turbo |

## Next Steps
From here a few next steps are possible:
- Try agentic rag with multiple steps -- e.g. first ask if info is required from nrma doc or allianz doc and then apply metadata filtering before rag
- Evaluate more retrieval strategies -- multi-query retrieval etc.
- Evaluate better chunking strategies -- semantic chunking, table parsing, etc.
- Manually create a hard evaluation question set and check the answers. 
