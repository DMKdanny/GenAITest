import streamlit as st
import openai
import tiktoken  # 텍스트를 여러 개의 청크로 나눌 때 문자 개수를 무엇으로 산정할 것인가 -> 토큰 수
from loguru import logger  # streamlit 웹사이트 상 구동했던 이력이 로그로 남게하기 위한 로거 라이브러리

from langchain.chains import ConversationalRetrievalChain  # 메모리를 가지고 있는 체인이기에 Q&A가 아닌 Conversational 사용
from langchain.chat_models import ChatOpenAI  # OpenAI 라이브러리 호출

# 여러 유형의 문서들을 인풋해도 이해하도록 하기 위해 각각의 파일형태 로더 호출
from langchain.document_loaders import PyPDFLoader  # PDF 파일 로더
from langchain.document_loaders import Docx2txtLoader  # 워드 파일 로더
from langchain.document_loaders import UnstructuredPowerPointLoader  # ppt 로더

from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트를 백터로 쪼갤 때
from langchain.embeddings import HuggingFaceEmbeddings  # 한국어에 특화된 임베딩 모델 채택

from langchain.memory import ConversationBufferMemory  # 이전 대화를 몇 개까지 기억할지 정함
from langchain.vectorstores import FAISS  # 임시 백터 스토어

# 메모리를 구현하기 위한 추가적 라이브러리
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
        page_title="IRMED-GenAI Test",
        page_icon=":rocket:")

    st.title("_IRMED:red[보도자료 검색]_ :rocket:")

    # Streamlit의 session_state.conversation 변수를 사용하기 위한 선행 정의 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # 위의 세션과 마찬가지
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("파일을 업로드해주세요", type=['pdf', 'docx', 'ppt'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("승인요청")
        
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstores(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 업로드 된 보도자료에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 채팅창 로직
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI의 토크나이저
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstores(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )  
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
