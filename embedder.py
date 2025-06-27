import yaml
from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    def _init_(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.embedder = HuggingFaceEmbeddings(
            model_name=config["embedder"]["model_name"],
            model_kwargs={"device": config["embedder"]["device"]}
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)