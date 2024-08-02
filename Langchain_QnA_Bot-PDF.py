from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")



doc_path = "/home/elicer/SUJINJEONG/DAY5/Synopsys-Documents/dw_lp_fp_multifunc.pdf"
loader = PyPDFLoader(doc_path)
docs = loader.load()



doc_len = [len(doc.page_content) for doc in docs]
print(doc_len)



vectorstore = FAISS.from_documents(
    docs,
    embedding=embeddings
)


db_retriever = vectorstore.as_retriever()



def get_retrieved_text(docs):
    result = "\n".join([doc.page_content for doc in docs])
    return result

def init_chain():
    messages_with_contexts = [
        ("system", "문서 정보 전달 지원 챗봇입니다. 사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {context}.\n{question}."),
    ]

    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    # 체인 구성
    # context에는 질문과 가장 비슷한 문서를 반환하는 db_retriever에 get_retrieved_text를 적용한 chain의 결과값이 전달됩니다.
    qa_chain = (
        {"context": db_retriever | get_retrieved_text, "question": RunnablePassthrough()}
        | prompt_with_context
        | llm
        | StrOutputParser()
    )
    
    return qa_chain


qa_chain = init_chain()



messages_with_variables = [
    ("system", "당신은 정보 지원 챗봇입니다."),
    ("human", "{question}."),
]
prompt = ChatPromptTemplate.from_messages(messages_with_variables)
parser = StrOutputParser()
chain = prompt | llm | parser



import streamlit as st

# qa_chain 초기화
qa_chain = init_chain()

# Streamlit 애플리케이션 시작
st.title("Real-time Text Generation with OpenAI API")

# 입력 프롬프트 받기
prompt = st.text_input("Enter your prompt:")

if prompt and st.button("Generate Text"):
    # 텍스트 생성 중 표시
    st.write("Generating text...")
    
    # qa_chain을 사용하여 답변 생성
    response = qa_chain.stream(prompt)
    
    generated_text = ""
    
    st.write(response)

else:
    st.warning("Please enter a prompt to generate text.")


# 저희는 이전 챕터에서 구현한 챗봇이 가지고 있는 문제점 중 '문서나 데이터 기반 추론이 불가능하다.'를 완화했습니다.
# 
# 또한, 지금 구성한 챗봇은 UI가 없고 단순 표준 입출력 만을 사용합니다. 5챕터에서 Streamlit을 활용해 ChatGPT와 비슷한 웹 챗봇 어플리케이션을 제작해 볼 것입니다.
