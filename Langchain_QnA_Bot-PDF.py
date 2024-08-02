#!/usr/bin/env python
# coding: utf-8

# # [실습] Langchain으로 시장조사 문서 기반 챗봇 만들기 - PDF

# ## 실습 목표
# ---
# [실습] LangChain을 활용해서 입력된 문서를 요약해서 Context로 활용하는 챗봇을 개발합니다.

# ## 실습 목차
# ---
# 
# 1. **시장조사 문서 벡터화:** RAG 챗봇에서 활용하기 위해 시장조사 파일을 읽어서 벡터화하는 과정을 실습합니다.
# 
# 2. **RAG 체인 구성:** 이전 실습에서 구성한 미니 RAG 체인을 응용해서 간단한 시장 조사 문서 기반 RAG 체인을 구성합니다.
# 
# 3. **챗봇 구현 및 사용:** 구성한 RAG 체인을 활용해서 시장조사 문서 기반 챗봇을 구현하고 사용해봅니다.

# ## 실습 개요
# ---
# RAG 체인을 활용해서 시장조사 문서 기반 챗봇을 구현하고 사용해봅니다.

# ## 0. 환경 설정
# - 필요한 라이브러리를 불러옵니다.

# In[4]:


from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# - Ollama를 통해 Mistral 7B 모델을 불러옵니다.

# In[2]:


#get_ipython().system('ollama pull llama3')


# ## 1. 시장조사 문서 벡터화
# - RAG 챗봇에서 활용하기 위해 시장조사 파일을 읽어서 벡터화하는 과정을 실습합니다.

# 먼저, mistral:7b 모델을 사용하는 ChatOllama 객체와 OllamaEmbeddings 객체를 생성합니다.

# In[5]:


llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")


# 다음으로, 시장조사 PDF 문서를 불러와서 벡터화 해보겠습니다.
# - 한국소비자원의 2022년 키오스크(무인정보단말기) 이용 실태조사 보고서를 활용했습니다
#   - https://www.kca.go.kr/smartconsumer/sub.do?menukey=7301&mode=view&no=1003409523&page=2&cate=00000057
# - 이 실태조사 보고서는 2022년 키오스크의 사용자 경험, 접근성, 후속 조치에 대해 논의하는 보고서입니다. 
# - 이를 활용해서 키오스크를 어떻게 세일즈 할 수 있을지 아이디어를 제공하는 챗봇을 만들어야 하는 상황이라고 가정해 봅시다.

# 먼저, LangChain의 `PyPDFLoader`를 활용해서 시장조사 보고서의 텍스트를 추출하고, 페이지 별로 `Document`를 생성하여 저장합니다.

# In[6]:


doc_path = "/home/elicer/SUJINJEONG/DAY5/Synopsys-Documents/dw_lp_fp_multifunc.pdf"
loader = PyPDFLoader(doc_path)
docs = loader.load()


# 생성된 Document의 수를 확인해봅시다.

# In[7]:


print(len(docs))


# 다음으로, 각 Document의 길이를 확인해봅시다.

# In[8]:


doc_len = [len(doc.page_content) for doc in docs]
print(doc_len)


# 1천자 미만의 문서도 있지만, 6천자가 넘는 문서도 있는 것을 확인할 수 있습니다. 이대로 그냥 사용할 경우, Context가 너무 길어져 오히려 성능이 낮아질 수도 있습니다.
# 
# 우선은 이대로 RAG 체인을 구성해 봅시다.

# ## 2. RAG 체인 구성
# RAG 체인을 구성하기 위해 `Document`를 `OllamaEmbeddings`를 활용해 벡터로 변환하고, FAISS DB를 활용하여 저장합니다.
# - 변환 및 저장 과정은 약 3분 정도 소요됩니다.

# In[9]:


vectorstore = FAISS.from_documents(
    docs,
    embedding=embeddings
)


# In[10]:


db_retriever = vectorstore.as_retriever()


# 이전 실습에서 구성한 미니 RAG Chain과 비슷하게 Chain을 구성해 봅시다.
# - 지난 실습과 달리 이번 챗봇의 역할은 마케터를 위한 챗봇으로 고정했으므로, 역할을 별도로 인자로 전달할 필요가 없습니다.
# - `RunnablePassthrough()`는 Chain의 이전 구성 요소에서 전달된 값을 그대로 전달하는 역할을 수행합니다.

# In[11]:


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


# In[12]:


qa_chain = init_chain()


# Chain 구성이 완료되었습니다.

# ## 3. 챗봇 구현 및 사용
# - 구성한 RAG 체인을 활용해서 시장조사 문서 기반 챗봇을 구현하고 사용해봅니다.

# 방금 구현한 RAG Chain을 사용해서 시장조사 문서 기반 챗봇을 구현해볼 것입니다. 
# 
# 그 전에, 별도로 RAG 기능을 추가하지 않은 LLM과 답변의 퀄리티를 비교해 봅시다.

# In[13]:


messages_with_variables = [
    ("system", "당신은 정보 지원 챗봇입니다."),
    ("human", "{question}."),
]
prompt = ChatPromptTemplate.from_messages(messages_with_variables)
parser = StrOutputParser()
chain = prompt | llm | parser


# In[16]:


#print(chain.invoke("DW_lp_fp_multifunc 에 대한 정보를 알려줘"))


# In[14]:


#print(qa_chain.invoke("DW_lp_fp_multifunc에 대한 정보를 알려줘"))


# 일반 체인은 아무런 출처가 없는 답변을 생성한 반면, RAG 기능을 추가한 챗봇은 데이터를 기반으로 상대적으로 정확한 답변을 하는 것을 확인할 수 있습니다. 
# 
# 이제 챗봇을 한번 사용해 봅시다.

# In[17]:


import streamlit as st
from openai import OpenAI

# OpenAI API 키 설정
api_key = 'sk-proj-u3n3LaECJBbtN1nOGSwiT3BlbkFJOkUc860FfNReBVr8SC5u'
client = OpenAI(api_key=api_key, base_url="https://f08a-14-35-173-251.ngrok-free.app/v1")

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
