import os
import json
from src.pdf_parser import PDFParser
from src.retriever import Retriever
from src.qa_chain import QAChain
from src.evaluator import Evaluator
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Parse PDF into markdown if file doesn't exist
p = PDFParser()
if not os.path.exists("./files/parsed_md/nrma.md"):
    p.parse("./files/pdfs/nrma.pdf")

if not os.path.exists("./files/parsed_md/allianz.md"):
    p.parse("./files/pdfs/allianz.pdf")

# Run evaluation with different configurations
configs = [
    {"embedding": "-", "retriever_mode": "bm25", "llm": "llama3"},
    {"embedding": "openai", "retriever_mode": "vector", "llm": "gpt-3.5-turbo"},
    {"embedding": "cohere", "retriever_mode": "vector", "llm": "gpt-3.5-turbo"},
    {"embedding": "openai", "retriever_mode": "ensemble", "llm": "gpt-3.5-turbo"},
]


for config in configs:
    logging.info(f"Running with config: {config}")
    retriever = Retriever(
        files=["./files/parsed_md/nrma.md", "./files/parsed_md/allianz.md"],
        retriever_mode=config["retriever_mode"],
        embedding=config["embedding"],
    )
    logging.info("Retriever loaded")
    qa_chain = QAChain(retriever=retriever.get_langchain_retriever(), llm=config["llm"])
    logging.info("QA Chain loaded")
    evaluator = Evaluator(
        qa_chain=qa_chain,
        llm="gpt-3.5-turbo",
        testset_path="./files/testset.csv",
        questions_limit=10,
    )
    logging.info("Evaluator loaded")
    evaluation_results = evaluator.evaluate()
    evaluation_results.update(config)
    with open("./files/evaluation_results.jsonl", "a") as f:
        json.dump(evaluation_results, f)
        f.write("\n")
