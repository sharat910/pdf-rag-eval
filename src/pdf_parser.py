import os
from llama_parse import LlamaParse


class PDFParser:
    """
    A class that represents a parser for processing input files using LlamaParse.

    Attributes:
        parser (LlamaParse): An instance of the LlamaParse class.
    """

    def __init__(self):
        self.parser = LlamaParse(
            api_key=self._get_api_key(),
            result_type="markdown",
            verbose=True,
            language="en",
        )

    def _get_api_key(self):
        """
        Retrieves the LlamaParse API key from the environment variables.

        Returns:
            str: The LlamaParse API key.

        Raises:
            ValueError: If the LlamaParse API key is not provided.
        """
        k = os.getenv("LLAMAPARSE_API_KEY")
        if k is None:
            raise ValueError("Please provide your LlamaParse API key")
        return k

    def parse(self, input_file, output_file):
        """
        Parses the input file using LlamaParse and writes the parsed text to the output file.

        Args:
            input_file (str): The path to the input file.
            output_file (str): The path to the output file.
        """
        documents = self.parser.load_data(input_file)
        with open(output_file, "w") as f:
            f.write(documents[0].text)


if __name__ == "__main__":
    # Testing the parser
    p = PDFParser()
    p.parse("../files/pdfs/allianz.pdf", "../files/parsed_md/allianz.md")
    p.parse("../files/pdfs/nrma.pdf", "../files/parsed_md/nrma.md")
