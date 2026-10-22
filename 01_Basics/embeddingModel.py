from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", dimension=10)

vector = embeddings.embed_documents("This is statement")
print(len(vector))

print(len(vector[0]))