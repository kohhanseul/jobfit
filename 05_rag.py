from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 저장된 벡터DB 불러오기
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local(
    "data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print("벡터DB 로드 완료!")

# 2. LLM 초기화
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 채용공고 분석 전문가야.
아래 채용공고 정보를 바탕으로 지원자의 이력서와 적합도를 분석해줘.

채용공고 정보:
{context}"""),
    ("human", "내 이력서: {resume}\n\n질문: {question}")
])

# 4. 실행 함수
def run_rag(resume, question):
    # 질문으로 관련 공고 검색
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 프롬프트 완성 후 LLM 호출
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "resume": resume,
        "question": question
    })
    return response.content

# 5. 실행
resume = "전자공학 전공. XGBoost, LSTM, CNN 구현 경험. Gemini API 연동 프로젝트 수행. 부트캠프 대상 수상."
question = "내 이력서에 맞는 공고 추천해줘"

print("\n질문 중...")
result = run_rag(resume, question)
print(f"\n=== RAG 분석 결과 ===\n{result}")