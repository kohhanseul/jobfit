from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. 텍스트 파일 읽고 청킹
with open("data/jobs.txt", "r", encoding="utf-8") as f:
    content = f.read()

job_list = [job.strip() for job in content.split("---") if job.strip()]
documents = [Document(page_content=job) for job in job_list]

print(f"문서 수: {len(documents)}개 로드 완료")

# 2. 임베딩 모델 초기화(임베딩 모델은 해당 API가 사용하는 임베딩모델임(따로 확인해야함))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. FAISS 벡터DB 생성
print("벡터DB 생성 중...")
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. 저장
vectorstore.save_local("data/faiss_index")
print("벡터DB 저장 완료!")

# 5. 검색 테스트
query = "딥러닝 경험이 필요한 공고 찾아줘"
results = vectorstore.similarity_search(query, k=2)

print(f"\n=== 검색 결과: '{query}' ===")
for i, doc in enumerate(results):
    print(f"\n[결과 {i+1}]\n{doc.page_content}")