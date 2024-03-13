---
title: "Langchain을 통한 GPT활용 톺아보기"
publishedAt: 2024-03-13
description: "노마드코더 풀스택 GPT 강의를 듣고 정리해보았습니다."
slug: "gpt-with-langchain"
isPublish: true
---

## 개발에서 GPT를 활용해 어떤 일들을 할 수 있을까?

GPT가 코딩도 하고 그림도 그리고 영상도 만든다는데, 정작 나의 GPT활용도는 구글 검색의 확장판이었다. 개발자로서는 GPT를 이용해 어떤 부분 활용까지 접근할 수 있을까 호기심이 생겨 공부를 해봐야겠다고 생각하던 찰나, 나에게 html과 css, React에 입문 시켜주었던 노마드 코더에 [Full stack GPT 강의]((https://nomadcoders.co/fullstack-gpt))가 있다는 것을 알게 되었다. 이 강의를 통해 **LLM모델 활용에 대한 기본적인 개념**과, **LangChain, Hugging Face와 같은 LLM모델을 사용할 수 있도록 도와주는 툴 이용법**, 그리고 이를 활용해 **응용해볼 수 있는 것**들이 무엇인지 알게 되었고, 따라서 이를 정리해보고자한다. 

## LangChain, 개발에서 LLM모델에 접근하는 방법 

개발에서 LLM을 이용한다는 것은, ChatGPT를 이용하듯 프롬프트를 통해 LLM 모델에 접근해, 얻은 결과를 메모리에 저장하고, 이를 캐싱해서 사용할 수 있는 것을 뜻한다. 다만 프롬프트를 어떻게 효율적으로 넣을 것인가, 또는 어떤 LLM 모델을 사용할 것인가(*Hugging Face 이용*), 그리고 이 결과값을 원하는 형태로 저장해서 활용할 수 있다는 점에서 ChatGPT를 활용하는 것 보다 훨씬 더 높은 활용도를 가질 수 있고, 그 거의 모든 과정을 LangChain이 지원을 해주고있다.

### RAG (Retrieval-Augmented Generation)

LLM모델을 다루는 방법에는, 프롬프트를 넣는 단계에서 여러 기법을 활용해 모델을 제어해서 활용하는 방법과(답변에 템플릿을 제공한다던지, 예제를 넣어서 알려준다던지 등등), 만들어진 모델에 특정 데이터를 학습시켜 조정하는 방법(fine-tuning) 등등 여러가지 방법이 있는데, 이 강의에서 주로 활용하는 방식은 **RAG** 이라는 기법을 활용하는 방법이다. RAG란 정보 검색과 생성을 결합한 기술로, 개인이 가지고 있는 데이터베이스 또는 문서 등을 활용해 검색에 대한 답변을 생성하는 방법이다.

<img
  src="https://image.samsungsds.com/kr/insights/what-is-langchain_img01.jpg"
  width="800"
  height="600"
/>

> 출처: https://www.samsungsds.com/kr/insights/what-is-langchain.html

RAG는, 활용하고자 하는 외부 데이터를 LLM모델을 활용해 검색할 수 있는 데이터 형태로 변환해, 데이터베이스에 저장을 하고, 프롬프트를 활용해 이를 뽑아낼 수 있도록 하는 과정이다. 개발자는 어떤 데이터를 로드할 것인지, 검색 효율화를 위해 데이터를 어느 token단위로 나눌 것인지, 분할한 데이터를 어떤 임베딩 모델을 활용해 벡터 형태로 전환할 것인지, 어떤 vector store에 저장할 것인지, 프롬프트를 어떻게 뽑아낼 것인지 등등의 세부 디테일 조정을 하면 된다. 

<img
  src="https://python.langchain.com/assets/images/data_connection-95ff2033a8faa5f3ba41376c0f6dd32a.jpg"
  width="800"
  height="600"
/>

> 출처: https://python.langchain.com/docs/modules/data_connection/

### 1. 데이터 로드와 분할

LangChain에서 제공하고 있는 data loader는 엄청나게 다양하기 때문에 LLM에 활용할 수 있는 외부 데이터 형태는 무궁무진하다. .pdf, .txt, .doc, .json .csv 등의 다양한 확장자의 외부 파일들도 거의 대응하고 있고, S3, Google Cloud 같은 클라우드 기반 데이터 베이스, 일반 데이터 베이스 뿐만 아니라 Git, Trello, Youtube, Wikipedia 등등 API활용해서 가지고 와야하는 데이터들까지 다 메소드 형식의 툴로 제공해주고 있어서 거의 모든 데이터를 간편하게 로드할 수 있다. 또한 BeautifulSoup이나 HTML to text같은 툴도 제공하기 때문에 크롤링한 데이터도 쉽게 접근이 가능하다. <br />

데이터를 로드하고 저장하기 전에 token단위로 분할하는 과정을 거치는데, LangChain은 데이터 형태에 맞게 분할하는 여러 툴을 제공하고 있다. 분할이 필요한 이유는, LLM이 효율적으로 데이터를 찾고 정확한 정보를 도출해내는 것을 돕기 위함이다. 데이터를 분할하게 되면 LLM모델이 한번에 다뤄야할 데이터 양이 적어져 불필요한 비용을 줄일 수 있고, 각 컨텍스트를 이해하기가 더 쉬워지므로 결과적으로 더 관련성 높은 데이터를 출력할 수 있다. <br />
간단하게 LangChain을 이용해 로컬 파일을 로드해 분할하는 예시 코드를 보면 다음과 같다.

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter

cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
loader = UnstructuredFileLoader(file_path)
docs = loader.load_and_split(text_splitter=splitter)
```

> 출처: 노마드코더 Fullstack GPT 강의 제공 코드


### 2. 데이터 임베딩과 스토어 저장

LLM에서 활용 가능한 형태로 데이터를 사용하려면, 데이터를 벡터로 변환해야하고, 이 과정을 임베딩이라고 한다. 이를 vector store에 저장을 해야 검색이 가능한데, LangChain을 이용해 활용할 수 있는 vector store는 유/무료로 나뉘어져 있고, Cloud공간 활용을 할 것인지 로컬 컴퓨터 저장 공간을 활용할 것인지 선택할 수 있다. 다만 이 과정은 CPU나 GPU와 같은 자원을 많이 활용해야하고, 시간이 오래걸리고, 벡터화를 동일하게 해서 일관된 데이터를 사용해야하기 때문에 보통 캐싱을 활용한다.<br />

이 과정 역시도 LangChain이 제공해주는 메소드를 통해 간단하게 구현할 수 있다.
```python
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS

embeddings = OllamaEmbeddings(
  model="mistral:latest" 
)
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddingss, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embeddings) # 위에서 load한 docs 이용
```
> 출처: 노마드코더 Fullstack GPT 강의 제공 코드

### 3. Retriever + LLM + Prompt 를 활용한 정보 검색

앞서 만들어진 vectorstore에 `as_retriever()`라는 메소드를 호출하면, 관련 정보나 데이터를 검색해내는 retriever를 만들어낸다. prompt를 이용해 유저에게 필요한 정보를 받아, 이 retriever를 이용해 정보 검색을 하는 것이다. LLM은 이 과정에서 사용자 질문에 대한 맥락을 이해하고, retriever에서 받은 결과를 통해 추론/통합/해석을 하는 등의 역할을 함으로써 정확한 정보가 도출되게 된다. 또한 이를 통해 얻어낸 응답을 Output Parser를 이용해 원하는 데이터로 가공하거나 특정 정보를 추출하는 일을 할 수 있다. 이 과정은 `|` 파이프라인 오퍼레이터를 이용해 한줄 pseudo 코드로 나타낼 수 있고 가능하고, 이를 LCEL(LangChain Expression Language) 일컫는다.

```python
chain = prompt | llm | output parser
```

데이터를 효과적으로 도출해내기 위해 prompt를 어떻게 입력할 것인지 제어하는 방법이 여러가지가 있는데, 이 강의에서 소개하고 있는 기본 타입에는 Stuff, Refine, Map Reduce, Map Re-rank 와 같은 방법들이 있다. 간단하게 소개하자면, retriever에서 받은 문서들로 prompt를 그냥 하나로 합쳐 채워넣는(stuff) 방법이 있고, 개별 문서를 읽으면서 답을 찾고 다음 문서들을 읽어내면서 답을 개선(refine)해내는 방법, 각각의 문서를 요약해가면서 LLM에 전달하는 방법(Map Reduce), 각 문서를 통해 답변을 만들어낸 다음 scoring해서 바꾸는 방법(Map Re-rank)이 있다. 어떤 방법을 이용해야할지는 원하는 prompt와 문서 개수에 따라 달라진다.<br />

간단하게 Map Reduce만 Pseudo code로 살펴보자면 다음과 같다.

```python
list of docs

for doc in list of docs | prompt | llm

for response in list of llm response | puth them all together

final doc | prompt | llm
```

이를 실제 구현 코드로 살펴보면 다음과 같다.
```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda # input이 통과하도록 해주는

retriever = vectorstore.as_retriever()

final_prompt = ChatPromptTemplate.from_messages([
  ("system",
    """
    Given the following extracted parts of a long document and a question, create final answer. 
    If you don't know the answer just say you don't know, 
    don't make it up.
    ------
    {context}
    """
  ),
  ("human", "{question}")
])

map_doc_prompt = ChatPromptTemplate.from_messages([
  ("system",
    """
    Use the following portion of a long document to see if any of the
    text is relevant to answer the question. Return any relevant text
    verbatim.
    ------
    {context}
    """
  ),
  ("human", "{question}")
]) # final prompt와 다르게 글자 그대로 전달하라고 명령

map_doc_chain = map_doc_prompt | llm

def map_docs(inputs):
  documents = inputs['documents']
  question = inputs['question']

  return "\n\n".join(
    map_doc_chain.invoke({
      "context": doc.page_content,
      "question": question
    }).content 
    for doc in documents
  ) 

map_chain = { 
  "documents": retriever, # 모든 document를 가져와
  "question": RunnablePassthrough() 
} | RunnableLambda(map_docs)

chain = {"context": map_chain, "question": RunnablePassthrough()} | final_prompt | llm 

chain.invoke("Describe Victory Mansions")
```
> 출처: 노마드코더 Fullstack GPT 강의 제공 코드

위 예제에서도 볼 수 있듯이, 정확한 결과를 도출해내기 위해서 ChatGPT를 사용하는 것처럼 prompt에도 가이드를 주어야하고, prompt를 넣는 방식에서도 제어가 필요하다.


## 그래서 어떻게 활용할 수 있을까?

위에서 설명한 RAG를 사용하면 Document GPT, 즉 외부 문서를 읽고 요약하거나 필요한 데이터를 포맷에 맞춰 뽑을 수 있는 GPT를 만들 수 있다는 것을 알 수 있다. 다른 예제는 어떤 것들이 있는지 강의 내용을 토대로 정리해보면 다음과 같다.

### Private GPT

강의에서 소개하고 있는 Private GPT는 새로운 기능을 이야기하는 것은 아니고, 외부 서버를 이용한 LLM모델을 이용하면 정보 유출의 우려가 있기 때문에 이를 private하게 돌릴 수 있는 방법을 소개하고 있다. 정보 유출이 우려된다면, LangChain에서 지원하는 Ollama같은 모델을 로컬에서 다운받아 돌리거나, hugging face에서 다른 로컬에서 활용할 수 있는 모델을 붙여서 이렇게 로컬 LLM모델을 사용할 수도 있다.

### Quiz GPT

Document GPT에서 응용할 수 있는 형태의 퀴즈 GPT이다. LLM에서 답변을 원하는 형태로 포맷팅할 수 있다는 아이디어에서 착안해 퀴즈 형태로 뽑아낼 수 있도록 하는 것이다. 강의에서 활용하는 예제에서는, 유저가 질문을 하면 객관식으로 보기를 던지도록 LLM에게 프롬프트로 요청을 하고, 이를 특정 텍스트 포맷으로 응답을 받아, 해당 응답을 json화 포맷으로 받도록 하는 포맷팅 프롬프트를 붙여서, 정/오답에 해당하는 데이터를 뽑을 수 있도록 하고 있다. 문서를 통한 퀴즈 GPT 뿐만 아니라 LangChain에서 제공하는 `WikipediaRetriever`를 이용해 위키피디아 글을 이용해 퀴즈를 생성하는 예도 보여주고 잇다.

### Site GPT

LangChain을 이용하면 크롤링도 비교적 단순하게 구현할 수 있다. `SiteMapLoader`를 제공하고 있기 때문에 .xml 형태의 페이지를 로드하고, 해당 페이지를 원하는 데이터 형태에 맞춰 parsing해서 크롤링하고, 이를 똑같이 RAG 형태로 vectorStore에 저장해 LLM모델에 넣어 원하는 정보를 쉽게 추출할 수 있다.

### Meeting GPT

OpenAI에서 오디오파일을 문서로 변환시켜주는 `Whisper api`도 제공하고 있기 때문에, 예를 들어 미팅 내용에서 어떤 이야기가 오고 갔는지 살펴볼 수 있는 Meeting GPT도 만들 수 있다.

### Investor GPT

어떤 회사의 주식을 사야하는지 말아야하는지도 대답해주는 GPT도 만들 수 있다. 이때는 단순히 LLM에 prompt연결을 하는 것이 아니라 `Agent`라는 개념이 나온다. 해당 회사를 사야하는지 물어보는 프롬프트를 주고, 그 프롬프트를 도출해 내는 과정을 list로 넣어서 LLM이 알아서 답을 도출하는데 List의 툴들을 이용하도록 하는 것이다. 코드 예시로 보자면 다음과 같다.

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager. 
            
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            
            Consider the performance of a stock, the company overview and the income statement.
            
            Be assertive in your judgement and recommend the stock or advise the user against it.
        """
        )
    },
)

```

이렇게 `initialize_agent` 를 활용해 정답을 도출해 내는 툴을 만들어 넣어주면 - *ticker 찾는 툴(StockMarketSymbolSearchTool), 주식 정보 api를 이용해 회사의 전반적인 정보(CompanyOverviewTool)/손익계산서(CompanyIncomeStatementTool)/주간보고서(CompanyStockPerformanceTool) 등등을 가져오는 툴* - LLM이 알아서 툴들을 활용해 해당 주식을 사는게 좋은지 추론을 해준다.


## 어떻게 활용할 것인가

위 강의를 통해서 아주 간략하게 GPT를 개발에서 기본적으로 사용하는 방법을 알아봤는데, 배운 툴을 가지고서는 딱 떠오르는 만들고 싶은 것이 생각이 나지는 않는다. 하지만 예제를 보면 볼수록 LangChain이 제공하고 있는 툴이 이 강의 하나로는 알기 어려울 정도로 너무나 많다는 것이다. 노마드코더 덕분에 기본적인 개념정도는 쉽고 빠르게 살펴볼 수 있었고, LangChain 문서 등을 더 살펴보면서 일 혹은 개인적으로 어디까지 활용할 수 있을지 고민해봐야겠다는 생각이 들었다.

<br/>
<br/>
<br/>

---

### 출처
- [참고한 fullstack-gpt 강의] https://nomadcoders.co/fullstack-gpt
- [LangChain Docs Python] https://python.langchain.com/docs


