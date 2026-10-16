# 2026년 [HYUNDAI] - AI Insight Campus_AI 서비스 개발과정
## **LangChain으로 만드는 웹 기반 챗봇 실습**

LLM 기반 챗봇의 구조를 이해하고, LangChain·Streamlit을 활용해 실질적인 웹 챗봇을 구현하는 것을 목표로 합니다.


## 교육 내용 요약

-과정소개 (목적) : 과정 목표 및 결과물 소개, LLM 기반 챗봇 개요, LangChain 특징
-개발 환경 준비 : Python 환경, OpenAI API 키 세팅, LangChain 설치 및 기본 테스트
- LangChain 기본 구조: LCEL(LangChain Expression Language) 개념, PromptTemplate/Model/OutputParser 실습 
- LCEL 실습: 단순 Q&A 챗봇, 다양한 프롬프트 실습, 구조화된 출력(JSON), ConversationMemory 기반 대화형 챗봇 구현 
- 문서 기반 챗봇 : 문서 로딩·분할, 임베딩 생성, 벡터DB(Chroma/FAISS), RetrievalQA 구성 
- 웹 UI 연동 및 배포 : Streamlit/Gradio UI 제작, 로컬 테스트, Streamlit Cloud 배포 
- 개인 프로젝트 : “나만의 챗봇” 설계·구현·시연 (전공 Q&A, 문서 검색 등) 

---


## 실습 환경 준비

본 과정의 모든 실습은 **Python 3.11 기반 Conda 가상환경**에서 진행됩니다.  
원활한 실습 진행을 위해 아래 절차를 **사전에 확인**해 주시기 바랍니다.

### 1. Anaconda 설치

본 실습은 **Conda 기반 Python 가상환경**을 사용합니다.  
아래 공식 페이지에서 운영체제(Windows / macOS / Linux)에 맞는 **Anaconda**를 먼저 설치해 주세요.

Anaconda 공식 다운로드 페이지:  
https://www.anaconda.com/download/success

### 2. Conda 가상환경 생성 (Python 3.11)

```bash
conda create -n hyundai01 python=3.11
```

### 3. 가상환경 활성화

```bash
conda activate "지정된 이름"(hyundai01)
```

### 4. 실습에 필요한 패키지 설치

루트 디렉터리에 포함된 `requirements.txt` 파일을 사용하여  
실습에 필요한 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 5. VS Code 설치 

과정의 실습은 Jupyter Notebook 또는 Visual Studio Code(VS Code)를 활용하여 진행됩니다.  
원활한 실습과 코드 관리, 디버깅을 위해 **VS Code 사용을 권장**합니다.

VS Code가 설치되어 있지 않은 경우, 아래 공식 홈페이지를 통해 설치해 주세요.

https://code.visualstudio.com/



> © 2026. LangChain으로 만드는 웹 기반 챗봇 실습  
> All rights reserved.
