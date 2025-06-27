from langchain.text_splitter import RecursiveCharacterTextSplitter
import yaml

class Chunker:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            length_function=len
        )

    def split_text(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)