import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from Bio import Entrez
from Bio import Medline
import numpy as np
from tqdm import tqdm
from google.auth import credentials, exceptions
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import mypubmed_utils
from mypubmed_utils import retrieve_pubmed_abstracts
from langchain_core.output_parsers import PydanticOutputParser
import uuid

#from IPython.display import Image, display
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from google.cloud import translate_v2 as translate # Import the translate module
from langchain_google_genai import ChatGoogleGenerativeAI

import gradio as gr
import logging

# VARIÁVEIS

# Caminho para o arquivo JSON da chave
key_path = 'bimaster-googlecloud.json'

# Define a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS com o caminho do arquivo
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

#Carrega as variáveis de ambiente conforme o arquivo env
#load_dotenv('env')

#Variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
OPENAI_API_CHAT_MODEL = os.getenv("OPENAI_API_CHAT_MODEL")
OPENAI_API_EMBEDDING_MODEL = os.getenv("OPENAI_API_EMBEDDING_MODEL")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME")

#Variáveis constantes que serão utilizadas para definir o tipo de pesquisa
CANCER_RESEARCH = "CANCER_RESEARCH"
GENERAL_MEDICAL_RESEARCH = "GENERAL_MEDICAL_RESEARCH"
GENERAL_QUESTION = "GENERAL_QUESTION"

#Variaváveis constantes que serão utilizadas na interface gráfica do Huggingface
MODEL_OPENAI_LABEL = "OpenAI gpt 3.5 Turbo "
MODEL_GOOGLE_LABEL = "Google Gemini 1.5 flash"

#Variáveis Globais que serão utlizadas

# Instancia o cliente Pinecone, usado para interagir com o serviço de banco de dados vetorial.
# A chave de API é fornecida via variável global PINECONE_API_KEY.
pc = Pinecone(api_key=PINECONE_API_KEY)

# Cria um índice Pinecone, utilizado para armazenar e pesquisar dados vetoriais.
#Foi armazenado anteriormente, cerca de 80 MIL artigos mais recentes completos e relacionados a cancer, da base PMC do NCBI em chunks
# O índice é acessado com base no nome definido na variável global PINECONE_INDEX_NAME.
index = pc.Index(PINECONE_INDEX_NAME)

# Instancia o modelo OpenAIEmbeddings, que gera embeddings para texto.
# O modelo de embeddings é selecionado com base na variável global OPENAI_API_EMBEDDING_MODEL.
embeddings = OpenAIEmbeddings(model=OPENAI_API_EMBEDDING_MODEL)

# Cria uma instância do modelo de linguagem generativo da Google.
# O modelo de IA do Google é configurado com o nome do modelo e a temperatura.
# A temperatura controla a aleatoriedade das respostas geradas (0 significa resposta mais determinística).
llm_google = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL_NAME,
    temperature=0)

# Cria uma instância do modelo de linguagem generativo da OpenAI.
# O modelo de IA da OpenAI é configurado com o nome do modelo e a temperatura.
# A temperatura controla a aleatoriedade das respostas geradas (0 significa resposta mais determinística).
llm_openai = ChatOpenAI(
    model=OPENAI_API_CHAT_MODEL,
    temperature=0)

# Cria uma instância do PineconeVectorStore para armazenar vetores dos artigos PMC relacionados a cancer.
# O parâmetro 'index' se refere ao índice do Pinecone onde os dados vetoriais serão armazenados e pesquisados.
# 'embedding' é o modelo de embeddings da OpenAI, usado para gerar representações vetoriais de texto.
# 'namespace' define namespace dentro do Pinecone onde os artigos foram armazenados.
cancer_vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="cancer")

#Configuração do Logging para o arquivo myapp.log e para o console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    handlers=[logging.FileHandler("myapp.log"),
                              logging.StreamHandler()], 
                    force=True)

#FUNÇÕES


def translate_text(text, target_language='en'):
    """
    Traduz o texto de qualquer idioma para o idioma especificado (inglês por padrão).

    Parâmetros:
    text (str): O texto que será traduzido.
    target_language (str): O idioma de destino (padrão: 'en' para inglês).

    Retorna:
    str: O texto traduzido ou o texto original caso haja exceção ou se já estiver no idioma alvo.
    """
    # Verifica se a chave de autenticação foi configurada
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        print("Erro: A chave de autenticação do Google Cloud não foi configurada corretamente.")
        return text

    try:
        # Cria o cliente de tradução
        translate_client = translate.Client()

        # Detecta o idioma original do texto
        result = translate_client.detect_language(text)
        detected_language = result['language']

        # Se o idioma detectado já for o idioma de destino, não faz nada
        if detected_language == target_language:
            return text

        # Realiza a tradução
        translated = translate_client.translate(text, target_language=target_language)
        return translated['translatedText']

    except DefaultCredentialsError:
        print("Erro: Credenciais do Google Cloud não foram fornecidas corretamente.")
    except Exception as e:
        print(f"Erro: {e}")

    # Em caso de erro, retorna o texto original
    return text

def assign_with_default(value, default_value):
    """
    Atribui o valor fornecido à variável 'value', mas se 'value' for vazio, 'N/A' ou None,
    atribui o valor padrão 'default_value'.

    Parâmetros:
        value (str, None): O valor que será verificado e atribuído.
        default_value (str, None): O valor a ser atribuído caso 'value' seja vazio, 'N/A', ou None.

    Retorna:
        str, None: O valor de 'value' se for válido, caso contrário, retorna 'default_value'.

    Exemplo:
        >>> assign_with_default("", "Valor Padrão")
        "Valor Padrão"
        >>> assign_with_default("Texto válido", "Valor Padrão")
        "Texto válido"
    """
    return value if value not in ["", "N/A", None] else default_value

class ChatAgentState(MessagesState):
    """
    Representa o estado do agente de chat durante a interação com o usuário.
    Essa classe herda de 'MessagesState' e adiciona atributos específicos que mostram o
    tipo de consulta, a consulta do usuário, à versão traduzida dessa consulta e a versão simplificada da consulta.

    Atributos:
        query_type (str): O tipo da consulta realizada pelo usuário (ex: 'pesquisa sobre câncer', 'pesquisa biomédica geral' e 'geral').
        simplified_query (str): A versão simplificada da consulta do usuário, focada em palavras chaves usada para pesquisa no PubMed
        user_query (str): A consulta original do usuário, sem modificações.
        user_query_translated (str): A consulta do usuário traduzida.

    Essa classe mantém as informações sobre o estado atual da interação com o agente de chat.
    """

    query_type: str  # O tipo da consulta do usuário, 'pesquisa sobre câncer', 'pesquisa biomédica geral' e 'geral'
    simplified_query: str  # Versão simplificada da consulta do usuário, focada em palavras chaves, usada para pesquisa no PubMed
    user_query: str  # A consulta original enviada pelo usuário.
    user_query_translated: str  # A consulta do usuário traduzida.
    chat_model: BaseChatModel #Modelo que será utlizado para as consultas ao LLM (usuário pode trocar a qualquer momento)

class Router(BaseModel):
    """
    Classe responsável por estruturar e validar a saída do modelo LLM em relação às consultas do usuário.

    Atributos:
        query_type (str): O tipo da consulta realizada, com os possíveis valores:
            - 'CANCER_RESEARCH': Relacionada a pesquisas sobre câncer.
            - 'GENERAL_MEDICAL_RESEARCH': Relacionada a temas biomédicos gerais, mas não sobre câncer.
            - 'GENERAL_QUESTION': Questões gerais não relacionadas à área biomédica.
        simplified_query (str): Uma versão simplificada e concisa da consulta do usuário em inglês,
            com foco nos termos científicos ou biomédicos principais.
        user_query (str): A consulta original fornecida pelo usuário, sem modificações.
        user_query_translated (str): A consulta original traduzida para o inglês, se necessário.

    Utiliza `pydantic` para validação e descrição dos campos, garantindo consistência nas entradas e saídas.
    """
    query_type: str = Field(description="Query type. It can assume the values: CANCER_RESEARCH, GENERAL_MEDICAL_RESEARCH or GENERAL_QUESTION")
    simplified_query: str = Field(description="Simplified query")
    user_query: str = Field(description="Original user query")
    user_query_translated: str = Field(description="Original user query translated to English")


def route_input(state: ChatAgentState):
    """
    Processa a entrada do usuário para determinar o tipo de consulta, simplificar a consulta
    e traduzi-la para o inglês, se necessário.

    Parâmetros:
        state (ChatAgentState): Estado atual da interação com o agente de chat, contendo mensagens e contexto.

    Retorna:
        dict: Um dicionário contendo os seguintes valores:
            - "query_type" (str): O tipo de consulta classificada.
            - "simplified_query" (str): A versão simplificada da consulta em inglês. Essa versão simplificada será ultizada para realizar buscas diretamente na base do PubMed
            - "user_query" (str): A consulta original do usuário.
            - "user_query_translated" (str): A consulta traduzida para o inglês.

    Descrição:
        1. Extrai a última mensagem enviada pelo usuário.
        2. Utiliza um modelo LLM com saída estruturada baseada na classe `Router` para:
            - Classificar a consulta (`query_type`).
            - Traduzir a consulta para o inglês (`user_query_translated`).
            - Gerar uma versão simplificada da consulta (`simplified_query`), que será ultizada para pesquisa direta na base do PubMed.
        3. Implementa regras de fallback:
            - Se houver erro na chamada do LLM ou no parsing, assume `GENERAL_QUESTION` como tipo de consulta.
        4. Retorna os resultados processados em um dicionário.

    Tratamento de Erros:
        - Em caso de falha ao invocar o modelo ou processar a resposta, a função loga o erro e lança a exceção.
        - O tipo de consulta é assumido como `GENERAL_QUESTION` para garantir que o fluxo continue.

    Exemplo de Uso:
        state = ChatAgentState(messages=[{"content": "Quais são os efeitos da dieta no câncer de pulmão?"}])
        result = route_input(state)
        print(result)
        # Saída esperada:
        # {
        #     "query_type": "CANCER_RESEARCH",
        #     "simplified_query": "Diet and lung cancer risk",
        #     "user_query": "Quais são os efeitos da dieta no câncer de pulmão?",
        #     "user_query_translated": "What are the effects of diet on lung cancer risk?"
        # }
    """
    # Inicializa variáveis
    msg = ""
    user_query = state["messages"][-1].content  # Extrai a última mensagem do usuário
    simplified_query = ""
    query_type = ""
    user_query_translated = ""

    # Define o prompt para o LLM
    prompt_router = PromptTemplate(
    template="""
      You are an expert in biomedical search queries helping a user carry out scientific searches on the PubMed database.
      Your task is to analyze the user query and organize the response into the following 4 categories:

      1) **user_query**: The original user question.
      2) **user_query_translated**: Translate the user query to English if it is not already in English. Always provide this translation.
      3) **query_type**: Classify the query into one of the following categories, ensuring that you select only one value from the provided options. No other value is
no other value is accepted:
        - **CANCER_RESEARCH**: For cancer-related research questions.
        - **GENERAL_MEDICAL_RESEARCH**: For biomedical topics unrelated to cancer.
        - **GENERAL_QUESTION**: For general non-biomedical topics.
      4) **simplified_query**: Create a simplified and concise query in English based on the key concepts extracted from the **user_query_translated**. This should:
        - Focus on essential biomedical or scientific terms relevant to the question.
        - Avoid pronouns, stopwords, and unnecessary details.
        - Be simple, clear, and effective for PubMed database searches.
        - Ensure that **only English terms are used**, even if the original query was in another language.

      **IMPORTANT VALIDATION RULES**:
      - Ensure the **simplified_query** is exclusively in English and focused on the core terms relevant to the search.
      - If the user query is short and clear, the simplified query may closely mirror the translation, but it must still be in English.


      ### Examples

      #### Example 1:
      user_query: "Quero informações sobre câncer de mama."
      user_query_translated: "I want information about breast cancer."
      query_type: CANCER_RESEARCH
      simplified_query: "Breast cancer information"

      #### Example 2:
      user_query: "Quais são os efeitos da dieta no risco de câncer de pulmão?"
      user_query_translated: "What are the effects of diet on lung cancer risk?"
      query_type: CANCER_RESEARCH
      simplified_query: "Diet and lung cancer risk"

      #### Example 3:
      user_query: "Me conte uma piada sobre gatos."
      user_query_translated: "Tell me a joke about cats."
      query_type: GENERAL_QUESTION
      simplified_query: "Tell me a joke about cats"

      #### Example 4:
      user_query: "Quero saber sobre a resistência à insulina no diabetes tipo 2."
      user_query_translated: "I want to know about insulin resistance in type 2 diabetes."
      query_type: GENERAL_MEDICAL_RESEARCH
      simplified_query: "Insulin resistance in type 2 diabetes"

      #### Example 5:
      User Query: Você pode explicar as pesquisas mais recentes sobre a eficácia da imunoterapia para o câncer de pulmão?
      User Query Translated: Can you explain the latest research on immunotherapy effectiveness for lung cancer?
      Is simplification needed here: Yes.
      Query Type: CANCER_RESEARCH
      Simplified Query: Immunotherapy effectiveness lung cancer

      ####Example 6:
      User Query: Houve algum progresso significativo no tratamento da doença de Alzheimer utilizando anticorpos monoclonais nos últimos cinco anos?
      User Query Translated: Has there been any significant progress in Alzheimer's disease treatment using monoclonal antibodies in the last five years?
      Is simplification needed here: Yes.
      Query Type: GENERAL_MEDICAL_RESEARCH
      Simplified Query: Alzheimer's disease monoclonal antibodies treatment progress

      #### Example 7:
      User Query: Você pode fornecer informações detalhadas sobre os avanços recentes na terapia genética para o tratamento da cegueira hereditária?
      User Query Translated: Can you provide detailed insights into the recent advancements in gene therapy for treating hereditary blindness?
      Is simplification needed here: Yes.
      Query Type: GENERAL_MEDICAL_RESEARCH
      Simplified Query: Gene therapy for hereditary blindness advancements

      #### Example 8:
      User Query: Estou interessado em compreender como a tecnologia CRISPR tem sido aplicada no desenvolvimento de terapias contra o câncer nos últimos anos.
      User Query Translated: I am interested in understanding how CRISPR technology has been applied in the development of cancer therapies over the recent years.
      Is simplification needed here: Yes.
      Query Type: CANCER_RESEARCH
      Simplified Query: CRISPR technology in cancer therapy development

      #### Example 9:
      User Query: Doença de Alzheimer e placas amilóides
      User Query Translated: Alzheimer's disease and amyloid plaques
      Is simplification needed here: No.
      Query Type: GENERAL_MEDICAL_RESEARCH
      Simplified Query: Alzheimer's disease and amyloid plaques

      #### Example 10:
      User Query: Fatores de estilo de vida e risco de câncer colorretal
      User Query Translated: Lifestyle factors and colorectal cancer risk
      Is simplification needed here: No.
      Query Type: CANCER_RESEARCH
      Simplified Query: Lifestyle factors and colorectal cancer risk

      #### Example 10:
      User Query: Qual é a montanha mais alta do mundo?
      User Query Translated: What is the tallest mountain in the world?
      Is simplification needed here: No.
      Query Type: GENERAL_QUESTION
      Simplified Query: What is the tallest mountain in the world?

      #### Example 11:
      User Query: Quais são as últimas descobertas sobre o impacto das alterações climáticas na incidência de doenças transmitidas por vetores nas regiões tropicais?
      User Query Translated: What are the latest findings on the impact of climate change on the incidence of vector-borne diseases in tropical regions?
      Is simplification needed here: Yes.
      Query Type: GENERAL_MEDICAL_RESEARCH
      Simplified Query: Climate change and vector-borne diseases in tropics

      ---

      **Attention**:
      1. If any part of the simplified query contains non-English words, you must rewrite it in English before submitting your response.
      2. Double-check all outputs to ensure they follow these rules. Any deviation from these instructions is an error.
      3. If the question is biomedical and verbose, simplify it appropriately for PubMed searches.
      4. Always respond for all attributes. Do not use N/A values.

      Now, process the following user query and provide the outputs:

      User query: {query}
      """,
          input_variables=["query"]
      )


    try:
        #Recupera o modelo setado no estado atual
        chat_model = state["chat_model"]
        
        # Configura o modelo LLM para saída estruturada usando a classe Router
        structured_llm = chat_model.with_structured_output(Router)

        # Combina o prompt e o modelo em uma cadeia
        chain = prompt_router | structured_llm

        # Invoca o modelo com a consulta do usuário
        result = chain.invoke(user_query)
        

        # Processa o resultado se for válido
        if isinstance(result, Router):
            query_type = assign_with_default(result.query_type, GENERAL_QUESTION)
            simplified_query = translate_text(result.simplified_query)
            user_query_translated = result.user_query_translated
            
        logging.info(f"{chat_model.get_name()} | router | {query_type}|")

    except Exception as e:
        # Loga erros e define valores padrão
        print(f"Error in route_input: {e}")
        query_type = GENERAL_QUESTION
        raise e

    # Retorna o resultado processado
    return {
        "query_type": query_type,
        "simplified_query": simplified_query,
        "user_query": user_query,
        "user_query_translated": user_query_translated
    }


def retrieve_cancer_chunks_context(query, k=10):
    """
    Recupera trechos de contexto relacionados a câncer da base de dados vetorial
    que armazena cerca de 80 mil artigos completos e mais recentes relacionados
    a cancer no PMC do NCBI.

    Parâmetros:
        query (str): Consulta do usuário.
        k (int, opcional): Número de documentos mais semelhantes a serem recuperados.
                           O valor padrão é 10.

    Retorna:
        str: Trechos concatenados de documentos relacionados ao câncer, formatados com título e link.

    Descrição:
        - Realiza uma busca de similaridade no `cancer_vector_store` com base na consulta fornecida.
        - Formata os trechos dos documentos recuperados, incluindo o conteúdo, título e link do artigo.
        - Retorna os trechos como uma única string concatenada, pronta para ser usada como contexto em prompts.
    """
    # Realiza a busca de similaridade com a consulta fornecida
    result = cancer_vector_store.similarity_search(query=query, k=k)

    # Lista para armazenar os trechos formatados
    context_chunks = []
    for document in result:
        # Formata cada trecho com conteúdo, título e link
        chunk = f"{document.page_content}\n(Title: '{document.metadata['title']}'\nlink: {document.metadata['link']})\n\n"
        context_chunks.append(chunk)

    # Retorna os trechos concatenados como string
    return "".join(context_chunks)

def perform_cancer_research(state: ChatAgentState):
    """
    Realiza uma pesquisa científica baseada em artigos relacionados
    a câncer utlizados como RAG

    Parâmetros:
        state (ChatAgentState): Estado atual da interação com o agente de chat,
                                contendo a consulta do usuário e sua versão simplificada.

    Retorna:
        dict: Um dicionário contendo a resposta gerada como mensagens estruturadas.

    Descrição:
        - Recupera trechos de contexto relacionados ao câncer usando a função `retrieve_cancer_chunks_context`.
        - Gera um prompt dinâmico com as informações recuperadas e a consulta do usuário.
        - Utiliza o modelo de linguagem (LLM) definido anteriormente para
        responder à consulta, citando artigos e links relevantes.
        - Aplica validações específicas:
            - Responder sempre no idioma da consulta original.
            - Incluir notas de rodapé com as fontes utilizadas.
    """
    # Extrai a consulta original e simplificada do estado
    user_query = state["user_query"]
    simplified_query = state["simplified_query"]

    # Recupera os trechos de contexto relacionados ao câncer usado a query simplificada
    # É esperado que a query simplificada esteja em ingles e com palavras chaves
    retrieved_cancer_chunks_context = retrieve_cancer_chunks_context(simplified_query, k=15)

    # Cria o prompt dinâmico
    prompt = PromptTemplate(
    template="""
      Answer the following scientific cancer related question (user question): {user_query},
      using just the entire following context retrieved from chunks of scientific cancer related articles from PMC:
      {retrieved_cancer_chunks_context}

      The user might refer to the history of your conversation. Please, use the following history of messages for the context as you see fit.
      When the question doesn't mention the type of cancer, try answering the question for cancer in general

      The articles chunks will come formatted in the following way (the content inside <> will be variable).:
      <article chunk 1>\n
      (<article title 1>\n
      link: <link to pubmed article 1>)

      <article chunk 2>\n
      (<article title 2>\n
      link: <link to pubmed article 2>)
      ...

      In your answer, ALWAYS respond on the SAME LANGUAGE from the user question. Cite the abstract(s) title and the link(s) to the article(s) when citing a particular piece of information from that given abstract.

      **IMPORTANT VALIDATION RULES**:

      - ALWAYS respond on the same language from the user question (propably Portuguese).
      - After the response, ALWAYS include the follow footnotes:
      "Answer based on chunks of the following medical articles from PMC:
      <Article Title 1>\n
      <Link to article 1>\n
      <Article Title 2>\n
      <Link to article 2>\n
      ..."
      - Footnotes also MUST be on the same language from the user question (propably Portuguese).

  """,
          input_variables=["user_query", "retrieved_cancer_chunks_context"]
      )

    #Obtem o modelo setado no estado atual
    chat_model = state["chat_model"]
    
    # Encadeia o prompt ao modelo de linguagem
    chain = prompt | chat_model

    # Invoca o modelo com os dados formatados
    result = chain.invoke({"user_query": user_query, "retrieved_cancer_chunks_context": retrieved_cancer_chunks_context})

    logging.info(f"{chat_model.get_name()} | cancer_research | {result.content[:50]}...|")

    messages = state["messages"]
    messages.append(result)

    # Retorna a resposta para ser setada no atributo messages do estado atual
    return {"messages": messages}


def perform_general_medical_research(state: ChatAgentState):
    """
    Realiza uma pesquisa científica baseada em resumos (abstracts) de artigos médicos gerais.

    Parâmetros:
        state (ChatAgentState): Estado atual da interação com o agente de chat,
                                contendo a consulta do usuário e sua versão simplificada.

    Retorna:
        dict: Um dicionário contendo a resposta gerada como mensagens estruturadas.

    Descrição:
        - Recupera resumos de artigos médicos gerais usando a função `retrieve_pubmed_abstracts`.
        - Gera um prompt dinâmico com os resumos e a consulta do usuário.
        - Utiliza um modelo de linguagem (LLM) para responder à consulta, citando artigos e links relevantes.
        - Aplica validações específicas:
            - Responder sempre no idioma da consulta original.
            - Incluir notas de rodapé com as fontes utilizadas.
    """
    # Extrai a consulta original e simplificada do estado
    user_query = state["user_query"]
    simplified_query = state["simplified_query"]

    # Recupera os resumos de artigos médicos gerais. Função definida no arquivo mypubmed_utils
    retrieved_abstracts = retrieve_pubmed_abstracts(simplified_query, retmax=60, num_articles=15)

    # Cria o prompt dinâmico
    prompt = PromptTemplate(
    template="""
      Answer the following scientific question (user question): {user_query},
      using just the entire following context retrieved from scientific articles: {retrieved_abstracts}.

      The user might refer to the history of your conversation. Please, use the following history of messages for the context as you see fit.

      The abstracts will come formatted in the following way:
      #Title: <abstract title>\n##Abstract:\n<abstract content>\nauthors: <authors list>\ndate of publication: <publication date>\nlink: <link to pubmed article> (the content inside <> will be variable).
      In your answer, ALWAYS respond on the SAME LANGUAGE from the user question. Cite the abstract(s) title and the link(s) to the article(s) when citing a particular piece of information from that given abstract.

      **IMPORTANT VALIDATION RULES**:

      - ALWAYS respond on the same language from the user question (propably Portuguese).
      - After the response, ALWAYS include the follow footnotes:
      "Answer based on abstracts of the following medical articles from PubMed:
      <Article Title 1>\n
      <Link to article 1>\n
      <Article Title 2>\n
      <Link to article 2>\n
      ..."
      - Footnotes also MUST be on the same language from the user question (propably Portuguese).

  """,
          input_variables=["user_query", "retrieved_abstracts"]
      )
    
    #Recupera o modelo do estado atual
    chat_model = state["chat_model"]
    
    # Encadeia o prompt ao modelo de linguagem
    chain = prompt | chat_model

    # Invoca o modelo com os dados formatados
    result = chain.invoke({"user_query": user_query, "retrieved_abstracts": retrieved_abstracts})

    logging.info(f"{chat_model.get_name()} | medical_research | {result.content[:50]}...|")

    messages = state["messages"]
    messages.append(result)

    # Retorna a resposta para ser setada no atributo messages do estado atual
    return {"messages": messages}




def call_llm(state: ChatAgentState):
    """
    Chama o modelo de linguagem (LLM) para responder a uma consulta geral.

    Parâmetros:
        state (ChatAgentState): Estado atual da interação com o agente de chat,
                                contendo a consulta do usuário e o histórico de mensagens.

    Retorna:
        dict: Um dicionário contendo a resposta gerada como mensagens estruturadas.

    Descrição:
        - Define um prompt genérico para o assistente responder perguntas gerais.
        - Combina o histórico de mensagens existente com a nova consulta do usuário.
        - Configura o modelo de linguagem com uma etiqueta específica (`final_node`).
        - Retorna a resposta do modelo como uma lista de mensagens.

    Notas:
        - O prompt utilizado orienta o modelo a evitar informações não verificadas e
          a assumir quando não tiver informações suficientes.
        - O histórico de mensagens é mantido para oferecer contexto na interação.
    """
    user_query = state['user_query']


    # Define o prompt inicial para o assistente
    general_prompt = f'''You're a friendly assistant and your goal is to answer general questions.
    Please, don't provide any unchecked information and just tell that you don't know if you don't have enough info.
    Question: {user_query}
    '''

    # Mensagens iniciais a serem enviadas ao modelo, incluindo a consulta do usuário
    messages = [
        HumanMessage(content= general_prompt)
    ]


    #Obtem do modelo que está setado no estado atual do agente
    chat_model = state["chat_model"]
    
    # Combina o histórico de mensagens com a nova interação
    msg_list = state["messages"] + messages

    # Invoca o modelo com as mensagens combinadas
    response = chat_model.invoke(msg_list)

    logging.info(f"{chat_model.get_name()} | call_llm | {response.content[:50]}...|")

    messages.append(response)


    # Retorna a resposta para ser setada no atributo messages do estado atual
    return {"messages": messages}


def get_route(state: ChatAgentState):
    """
    Obtém o tipo de consulta a partir do estado atual do agente.

    Parâmetros:
        state (ChatAgentState): Estado atual contendo informações da consulta do usuário.

    Retorna:
        str: O tipo de consulta identificado no estado (`query_type`).

    Descrição:
        - Retorna o valor da chave `query_type` do estado.

    Observação:
        - O valor de `query_type` deve ser um dos seguintes: `CANCER_RESEARCH`,
          `GENERAL_MEDICAL_RESEARCH` ou `GENERAL_QUESTION`.
    """
    return state["query_type"]


def build_state_graph():
    """
    Constrói e retorna um fluxo de trabalho (state graph) para gerenciar as etapas do processamento de consultas.

    Retorna:
        tuple: O grafo compilado (`app`) e a instância da memória utilizada como checkpoint
        para armazenar o histórico das mensagens (`memory`).

    Descrição:
        - Define um grafo de estados utilizando a classe `StateGraph`.
        - Configura os estados e as transições para processar diferentes tipos de consultas:
            - `router`: Identifica o tipo de consulta.
            - `cancer_research`: Lida com pesquisas relacionadas a câncer.
            - `general_medical_research`: Lida com consultas biomédicas gerais.
            - `general_question`: Lida com perguntas gerais.
        - Adiciona um sistema de memória para salvar o estado entre as etapas.

    Detalhes do Grafo:
        - Início (`START`) -> Roteador (`router`).
        - Roteador (`router`) -> Estado correspondente com base em `query_type`.
        - Estados de processamento (`cancer_research`, `general_medical_research`, `general_question`) -> Fim (`END`).

    Observação:
        - O grafo é compilado com uma funcionalidade de checkpoint para manter o estado em memória.

    Exemplos:
        - Se `query_type` for `CANCER_RESEARCH`, a consulta será direcionada ao nó `cancer_research`.
        - Se `query_type` for `GENERAL_MEDICAL_RESEARCH`, a consulta será direcionada ao nó `general_medical_research`.
        - Para consultas gerais, será utilizado o nó `general_question`.
    """
    # Cria um grafo de estados com o esquema de mensagens
    workflow = StateGraph(state_schema=ChatAgentState)

    # Define as transições do grafo
    workflow.add_edge(START, "router")  # Transição inicial para o nó router

    # Adiciona o nó de roteamento
    workflow.add_node("router", route_input)

    # Define as transições condicionais baseadas no tipo de consulta
    workflow.add_conditional_edges(
        "router",
        get_route,  # Função para determinar o próximo estado
        {
            CANCER_RESEARCH: 'cancer_research',
            GENERAL_MEDICAL_RESEARCH: 'general_medical_research',
            GENERAL_QUESTION: 'general_question'
        }
    )

    # Adiciona os nós de processamento e suas transições
    workflow.add_node("cancer_research", perform_cancer_research)
    workflow.add_edge("cancer_research", END)

    workflow.add_node("general_medical_research", perform_general_medical_research)
    workflow.add_edge("general_medical_research", END)

    workflow.add_node("general_question", call_llm)
    workflow.add_edge("general_question", END)

    # Adiciona um sistema de memória para salvar o estado
    memory = MemorySaver()

    # Compila o fluxo de trabalho com o checkpoint
    app = workflow.compile(checkpointer=memory)

    
    #Estado inicial do Agente
    inital_state = {  "messages" : [],
    "query_type": "",
    "simplified_query": "",
    "user_query": "",
    "user_query_translated": "",
    "chat_model" : llm_openai
      }

    #Atualizando o grafo com o estado inicial
    app.update_state(config=config, values=inital_state)
    
    return app, memory


# def display_graph(app):
#     """
#     Exibe o grafo do fluxo de trabalho (workflow) da aplicação como uma imagem.

#     Args:
#         app: Aplicação compilada que contém o grafo do fluxo de trabalho.

#     Raises:
#         Exception: Caso as dependências necessárias para renderizar a imagem não estejam instaladas.
#     """
#     try:
#         # Obtém o grafo do workflow e renderiza como uma imagem PNG usando Mermaid.js.
#         display(Image(app.get_graph().draw_mermaid_png()))
#     except Exception:
#         # Lança uma exceção se a renderização falhar, geralmente por falta de dependências.
#         raise Exception

def print_memory_messages(memory):
    """
    Itera e imprime as mensagens armazenadas na memória de checkpoint.

    Args:
        memory: Objeto que contém o estado salvo do fluxo de trabalho, incluindo mensagens armazenadas.
    """
    for idx, message in enumerate(memory.get_tuple(config).checkpoint["channel_values"]["messages"], start=1):
        # Itera sobre as mensagens armazenadas e imprime o conteúdo.
       print(f"Mensagem {idx}: {message.content}")


def respond_to_message(message, history, radio_choice):
    """
    Processa a mensagem do usuário, invoca o modelo de linguagem apropriado e retorna a resposta formatada.

    Args:
        message (str): Mensagem enviada pelo usuário.
        history (list): Histórico de mensagens anteriores.
        radio_choice (str): Modelo selecionado (OpenAI ou Google Gemini).

    Returns:
        str: Resposta gerada pelo modelo, formatada com rótulos.
    """
    # Inicializa os rótulos para o modelo e tipo de consulta
    model_sub_label = ""
    query_type_label = ""

    chat_model = llm_openai  # Modelo padrão é OpenAI

    # Seleção do modelo com base na escolha do usuário
    if radio_choice == MODEL_OPENAI_LABEL:
        chat_model = llm_openai  # Modelo OpenAI
        model_sub_label = "by OpenAI"  # Rótulo do modelo usado
    elif radio_choice == MODEL_GOOGLE_LABEL:
        chat_model = llm_google  # Modelo Google Gemini
        model_sub_label = "by Gemini"  # Rótulo do modelo usado

    #Atualiza o modelo do estado atual do grafo para o selecionado pelo usuário
    app.update_state(config=config, values={"chat_model": chat_model})
    
    # Criação da mensagem de entrada para o fluxo de trabalho
    input_message = HumanMessage(content=message)

    logging.info(f"{chat_model.get_name()} | sending | {message[:30]}|")

    # Invoca o fluxo de trabalho da aplicação com a mensagem de entrada
    result = app.invoke({"messages": [input_message]}, config)

    # Obtém o tipo de consulta processada
    query_type = app.get_state(config).values["query_type"]

    # Define o rótulo de tipo de consulta com base no resultado
    if query_type == GENERAL_QUESTION:
        query_type_label = "&#x1F916; resposta gerada pela IA"  # Rótulo para perguntas gerais
    if query_type == CANCER_RESEARCH:
        query_type_label = "&#x1F397;&#xFE0F; resposta sobre cancer"  # Rótulo para pesquisas sobre câncer
    if query_type == GENERAL_MEDICAL_RESEARCH:
        query_type_label = "&#x1F50D; resposta sobre pesquisa biomédica"  # Rótulo para pesquisas biomédicas

    # Monta o subtítulo com os rótulos de tipo de consulta e modelo
    sub_label = f"<sub>{query_type_label} {model_sub_label} \>\>\></sub>\n"

    # Formata a resposta final combinando o subtítulo e o conteúdo gerado
    response = f"{sub_label}{result['messages'][-1].content}"

    # Retorna a resposta formatada
    return response

def fill_example(example):
    """Fill the example textbox with the selected example."""
    return example[0]


thread_id = uuid.uuid4()
logging.info(f"Thead ID: {thread_id}")
   
config = {"configurable": {"thread_id": uuid.uuid4()}}
app, memory = build_state_graph()

description = """ <p style="font-size: 12px;">
                  &#x1F397;&#xFE0F; Realize pesquisas relacionadas a cancer nos artigos mais recentes e completos do PMC<br>
                  &#x1F50D; Realize pesquisas relacionadas questões biomédicas em geral em resumos de artigos do PubMed<br>
                  &#x1F916; Ou utilize o chatbot diretamente no modelo de LLM escolhido para quaisquer outras questões.
                  </p>"""


with gr.Blocks(theme="Ocean") as demo:

  with gr.Row():
    with gr.Column():
      gr.Markdown("<center><h1>ChatBot Biomédico</h1></center>")
      gr.Markdown(description)

      iface = gr.ChatInterface(
          fn=respond_to_message,
          chatbot=gr.Chatbot(height=300),
          additional_inputs=[
              gr.Radio(
                  choices=[MODEL_OPENAI_LABEL, MODEL_GOOGLE_LABEL],
                  label="Escolha um modelo de LLM",
                  value= MODEL_OPENAI_LABEL
              )
          ],
          additional_inputs_accordion="Escolha o tipo de chat aqui:"
       )

  with gr.Row():
    with gr.Column():
      gr.Markdown("<center><h2>Exemplos</h2></center>")
      examples = [
      "Qual a relação entre obesidade e cancer?",
      "Cite fatores de risco para diabetes?",
      "Você pode fornecer informações detalhadas sobre os avanços recentes na terapia genética para o tratamento da cegueira hereditária?",
      "Você pode explicar as pesquisas sobre a eficácia da imunoterapia para o câncer colorretal?",
      "Quais são as últimas descobertas sobre o impacto das alterações climáticas na incidência de doenças transmitidas por vetores nas regiões tropicais?",
      "Quais são os efeitos da dieta no risco de câncer de pulmão?",
      "Qual a montanha mais alta do mundo?",
      "Cite as palavras chave que foram utilizadas em nossa conversa"]
      dataset = gr.Dataset(label="Exemplos de perguntas:",
        components=[iface.textbox],
        samples=[[example] for example in examples],)
      dataset.click(fn=fill_example, inputs=dataset, outputs=iface.textbox)


if __name__ == "__main__":
    logging.info("starting CHATBOT")
    demo.launch(share=True)