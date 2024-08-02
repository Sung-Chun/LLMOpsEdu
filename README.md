# LLMOpsEdu 과제

## 개요
* 날짜: 2024.8.1

* Members: 전성민, 박성언, 정수진

## 프로젝트 아이템 설명
* 서비스 아이디어: 삼성전자 파운드리 반도체 설계 지원 챗봇 (Design Support Chatbot for Samsung Foundry Customers)

* 서비스 내용:
EDA(전자설계자동화) 도구를 사용하는 삼성전자 파운드리 고객사를 대상으로, 삼성전자 파운드리 공정에 최적화된 회로 설계 과정을 지원하는 AI 기반 챗봇입니다.

## 개발
### Skeleton 코드
* Langchain_QnA_Bot-PDF.ipynb: 랭체인으로 PDF 파일을 읽어서 FAISS vector DB에 vector화 하여 저장하고 mistal:7b 모델을 기반으로 챗봇 서비스

* Langchain_QnA_Bot-PDF.py: streamlit으로 실행하는 코드

* app.py: 챗봇 서버에 API를 통해 질문을 던지고 답변을 받아서 그 결과를 사용자에게 보여주는 streamlit 기반 webapp


## TODO
1. 현재 1가지 PDF 파일을 읽어서 vector DB화 하였는데, 이것을 Synopsys-Document 내의 모든 PDF 파일을 읽어서 vector DB화 하기 (DONE)
2. Foundation 모델을 mistral:7b에서 Llama3.1 8B 로 변경하기 (DONE)
3. Llama3.1 70B 로 해보고 8B 모델과 비교해보기 (TBD)

## 실행방법
ollama 설치
```
$ ollama pull mistral:7b
$ ollama pull llama3
$ ollama serve
```

streamlit 실행
```
$ streamlit run Langchain_QnA_Bot-PDF.py
```
