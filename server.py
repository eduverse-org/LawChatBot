import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify 
import psycopg2
import yaml
from pgvector.psycopg2 import register_vector
from embedder import Embedder
from models.model_loader import ModelLoader
from prompt_templates import basic_prompt
from flask_cors import CORS

# Load .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Embedder and Groq LLM
embedder = Embedder()
llm = ModelLoader()

# Connect to PostgreSQL + pgvector
conn = psycopg2.connect(config["vector_store"]["postgres"]["connection_string"])
register_vector(conn)
cursor = conn.cursor()
table_name = config["vector_store"]["postgres"]["table_name"]

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        question = data.get("question")
        language = data.get("language", "en")
        k = data.get("top_k", 3)

        if not question:
            return jsonify({"error": "Missing 'question' in request"}), 400

        # Step 1: Embed user query
        query_embedding = embedder.embed_texts([question])[0]

        # Step 2: Validate embedding
        if not isinstance(query_embedding, list) or not all(isinstance(v, float) for v in query_embedding):
            return jsonify({"error": "Invalid embedding vector format"}), 500
        if len(query_embedding) != 384:
            return jsonify({"error": f"Expected embedding of 384 dimensions, got {len(query_embedding)}"}), 500

        # Step 3: Perform similarity search
        try:
            cursor.execute(
                f"""
                SELECT content
                FROM {table_name}
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (query_embedding, k)
            )
            rows = cursor.fetchall()
        except Exception as db_error:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(db_error)}"}), 500

        if not rows:
            return jsonify({"error": "No relevant context found in the database."}), 404

        context = "\n".join([row[0] for row in rows])
        prompt = basic_prompt.format(context=context, question=question, language=language)
        answer = llm.generate_response(prompt)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
