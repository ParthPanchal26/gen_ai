from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model


# model = init_chat_model("google_genai:gemini-3.1-flash-lite", temperature=0.0)


response = model.invoke("Say one sentence on the sun")


print("_"*30)
print(response.content)
print("_"*30)