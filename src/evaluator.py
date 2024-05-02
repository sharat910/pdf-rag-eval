import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from src.helper import get_llm
from src.qa_chain import QAChain
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


class Evaluator:
    """
    Class to evaluate the performance of a QA system.

    Args:
        qa_chain (QAChain): The QAChain object representing the QA system.
        llm (str): The language model used by evaluator.
        testset_path (str): The path to the test set file.
        questions_limit (int, optional): The maximum number of questions to evaluate. Defaults to 20.
    """

    def __init__(
        self, qa_chain: QAChain, llm: str, testset_path: str, questions_limit: int = 20
    ):
        self.qa_chain = qa_chain.get_langchain_qa_chain()
        self.testset = self._load_test_set(testset_path)
        self.llm = get_llm(llm)
        self.questions_limit = questions_limit

    def _load_test_set(self, testset_path):
        """
        Load the test set from a CSV file.

        Parameters:
        testset_path (str): The path to the CSV file containing the test set.

        Returns:
        pandas.DataFrame: The loaded test set as a pandas DataFrame.
        """
        return pd.read_csv(testset_path)

    def _get_evaluation_dataset(self):
        """
        Get the evaluation dataset for testing the model.

        Returns:
            Dataset: The evaluation dataset containing questions, answers, contexts, and ground truths.
        """
        questions = self.testset["questions"].tolist()[: self.questions_limit]
        ground_truths = self.testset["ground_truths"].tolist()[: self.questions_limit]

        answers = []
        contexts = []
        print("Getting answers for test questions...")
        for question in tqdm(questions[: self.questions_limit]):
            response = self.qa_chain.invoke({"question": question})
            contexts.append([context.page_content for context in response["context"]])

            if type(response["response"]) == str:
                answers.append(response["response"])
            else:
                answers.append(response["response"].content)
        return Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

    def evaluate(self):
        """
        Evaluate the performance of the QA system.

        Returns:
            dict: A dictionary containing the evaluation metrics and their values.
        """
        return evaluate(
            self._get_evaluation_dataset(),
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
            llm=self.llm,
        )
