import os
from groq import Groq
from dotenv import load_dotenv
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Do NOT log the actual API key
        if not os.getenv("GROQ_API_KEY"):
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY is missing")

        # Load configuration
        try:
            with open("config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            logger.error("config.yaml not found in config/ directory")
            raise

        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model_name = config["llm"]["model_name"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.temperature = config["llm"]["temperature"]

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using Groq's LLaMA model.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            str: The generated response.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a legal assistant specializing in Indian law. Provide accurate and concise answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            logger.info("Generated response for prompt preview: %s", prompt[:50].replace("\n", " "))
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Groq API call failed: %s", str(e))
            raise Exception("LLM generation failed. Please try again.")