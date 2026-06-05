from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke([HumanMessage(content="AI 엔지니어에게 필요한 역량 3가지를 알려줘")])

print(response.content)