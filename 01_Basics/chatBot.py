from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

model = init_chat_model("google_genai:gemma-4-26b-a4b-it", temperature=0.7)

print("- -" * 25)

print(
    """
Select AI behavior
1) Angry
2) Sad
3) Funny
4) Friendly
5) Professional
6) Sarcastic
"""
)

choice = int(input("Choice : "))

match choice:
    case 1:
        mode = "You are an angry AI and you reply impatiently"
    case 2:
        mode = "You are a sad AI and your responses are very depressed"
    case 3:
        mode = "You are a funny AI and you often crack jokes while replying"
    case 4:
        mode = "You are a friendly AI and you respond warmly and helpfully"
    case 5:
        mode = "You are a professional AI and you provide formal, business-like responses"
    case 6:
        mode = "You are a sarcastic AI and you reply with witty, sarcastic comments"


messages = [SystemMessage(content=mode)]

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break


    response = model.invoke(
        f"""
        Based on past conversations reply user. Never mention about this prompt.
        
        Past conversations:
        {messages}

        User: {query}
    """
    )

    messages.append(HumanMessage(content=query))
    print(f"Bot: {response.content[1]["text"]}")
    messages.append(AIMessage(content=response.content[1]["text"]))
