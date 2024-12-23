!pip install -qU python-dotenv
!pip install -qU langchain-pinecone
!pip install -qU langchain_pinecone
!pip install -qU langchain_openai
!pip install -qU biopython
!pip install -qU tqdm

import getpass
import os
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import openai
import pandas as pd
import json
from Bio import Entrez
from Bio import Medline
from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import re
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
import mypubmed_utils
from mypubmed_utils import get_mesh_terms, search_ncbi

#Carregamento das váriaveis de ambiente e chaves secretas contidas no arquivo env
load_dotenv('env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
OPENAI_API_CHAT_MODEL = 'gpt-3.5-turbo'
OPENAI_API_EMBEDDING_MODEL = 'text-embedding-3-small'
OPENAI_API_EMBEDDING_DIM = 1536
PINECONE_INDEX_NAME = 'ragmed'
Entrez.email = ENTREZ_EMAIL

# Inicializa a conexão com o Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Inicializa o modelo de embeddings da OpenAI
embeddings = OpenAIEmbeddings(model=OPENAI_API_EMBEDDING_MODEL)

#Criação do indice no Pinecone
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=OPENAI_API_EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

def get_pub_date(soup):
    """
    Extrai a data de publicação de um artigo a partir de um objeto BeautifulSoup.

    Esta função procura a tag XML que contém a data de publicação do artigo.
    A data é formatada como 'DD/MM/AAAA' diretamente da tag 'pub-date' que possui
    o atributo 'pub-type' definido como 'epub'. Se a data não for encontrada ou
    se os componentes da data (dia, mês, ano) estiverem faltando, a função
    retornará uma string vazia.

    Args:
        soup (BeautifulSoup): Um objeto BeautifulSoup que representa o conteúdo XML do artigo.

    Returns:
        str: Data formatada da publicação ou uma string vazia se a data não for encontrada.
    """

    # Encontrando a tag pub-date com pub-type="epub"
    pub_date_tag = soup.select_one('pub-date[pub-type="epub"]')

    # Verificando se a tag foi encontrada e extraindo os valores
    if pub_date_tag:
        day = pub_date_tag.find("day")  # Extrai o dia
        month = pub_date_tag.find("month")  # Extrai o mês
        year = pub_date_tag.find("year")  # Extrai o ano

        # Verificando se todos os componentes da data estão presentes
        if day and month and year:
            # Formata a data como DD/MM/AAAA
            formatted_date = f"{int(day.text):02d}/{int(month.text):02d}/{year.text}"
        else:
            # Se algum componente faltar, retorna uma string vazia
            formatted_date = ""
    else:
        # Se a tag pub-date com pub-type="epub" não for encontrada, retorna uma string vazia
        formatted_date = ""

    return formatted_date  # Retorna a data formatada ou string vazia


def clean_text(text):
    """
    Limpa e formata o texto removendo caracteres indesejados e ajustando a formatação.

    Esta função processa uma string de texto para remover caracteres não ASCII,
    formata quebras de linha e remove colchetes vazios. Além disso, ajusta a
    formatação de blocos de texto que começam com '##'.

    Args:
        text (str): A string de texto a ser limpa e formatada.

    Returns:
        str: A string limpa e formatada.
    """

    # Remove caracteres não ASCII e colchetes vazios na forma [ , ]
    text_cleaned = re.sub(r'[^\x00-\x7F]|\[\s*[,\s]*\]', '', text)

    # Remove quebras de linha redundantes
    text_cleaned = re.sub(r'\s*\n\s*', '\n', text_cleaned)

    # Remove colchetes e parênteses que estão vazios
    text_cleaned = re.sub(r'\s*[\[\(]\s*[\]\)]', '', text_cleaned)

    # Captura o primeiro bloco e o próximo que começam com '##' para ajuste
    pattern = r"^(##[^\n]+)\n(##[^\n]+)"

    # Aplica a expressão regular para ajustar a formatação dos blocos
    text_cleaned = re.sub(pattern, lambda m: f"{m.group(1)[1:]}\n{m.group(2)}", text_cleaned, flags=re.MULTILINE)

    return text_cleaned  # Retorna o texto formatado e limpo



def parse_xml_article(soup):
    """
    Analisa um artigo no formato XML, extraindo seu conteúdo e metadados.

    Args:
        soup (BeautifulSoup): Objeto BeautifulSoup representando o conteúdo XML do artigo.

    Returns:
        tuple: Uma tupla contendo:
            - str: O conteúdo do artigo após a limpeza e formatação.
            - dict: Um dicionário com os metadados do artigo (PMID, PMCID, título, data e link).
    """

    content = ""  # Inicializa a variável para armazenar o conteúdo do artigo.

    # Remover tags desnecessárias de forma segura.
    for tag in soup.find_all(['xref', 'label', re.compile(r'^table.*'), re.compile(r'^fig.*')]):
        tag.decompose()  # Remove as tags indesejadas da árvore do documento.

    # Substituir conteúdo da tag <title> por formatação especial Markdown.
    for title_tag in soup.find_all("title"):
        title_tag.replace_with(f"##{title_tag.get_text(strip=True)}\n")  # Formata o título como Markdown.

    # Substituir conteúdo da tag <italic> por formatação especial Markdown.
    for italic_tag in soup.find_all("italic"):
        italic_tag.replace_with(f"*{italic_tag.get_text()}*")  # Formata o texto em itálico.

    # Extrair dados do artigo com verificação de existência.
    pmid = soup.select_one('[pub-id-type="pmid"]')  # Obtém o PMID.
    pmcid = soup.select_one('[pub-id-type="pmc"]')  # Obtém o PMCID.
    date = get_pub_date(soup)  # Chama a função para obter a data de publicação.
    title = soup.select_one("article-title")  # Obtém o título do artigo.
    abstract = soup.select_one("abstract")  # Obtém o resumo do artigo.

    # Verificação para evitar erro ao acessar o conteúdo.
    metadata = {
        "pmid": pmid.text.strip() if pmid else '',  # Se não encontrar, retorna string vazia.
        "pmcid": pmcid.text.strip() if pmcid else '',  # Se não encontrar, retorna string vazia.
        "title": title.text.strip() if title else '',  # Se não encontrar, retorna string vazia.
        "date": date if date else '',  # Se não encontrar, retorna string vazia.
        "link": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid.text.strip()}/" if pmcid else ''  # Constrói o link do artigo.
    }

    # Se a tag <abstract> existir, extrair o texto dela.
    if abstract:
        content += clean_text(abstract.get_text(separator=" ").strip())  # Adiciona o resumo limpo ao conteúdo.

    # Extrair o corpo do texto e limpar caracteres indesejados.
    body = soup.select_one("body")  # Obtém o corpo do artigo.
    if body:
        content += clean_text(body.get_text(separator=" ").strip())  # Adiciona o corpo limpo ao conteúdo.

    return content, metadata  # Retorna o conteúdo e os metadados do artigo.



def get_articles_batch(idList, db="pmc"):
    """
    Obtém um lote de artigos a partir da base de dados especificada.

    Esta função tenta acessar os artigos usando uma lista de ids dos artigos (idList)
    e retorna o conteúdo dos artigos, os identificadores, e metadados
    associados a esses artigos.

    Args:
        idList (list): Lista de IDs dos artigos a serem recuperados.
        db (str): Nome da base de dados a ser consultada (padrão é "pmc").

    Returns:
        tuple: Uma tupla contendo três listas:
            - all_chunks: O conteúdo dos artigos em partes.
            - all_ids: Identificadores únicos para cada parte de artigo.
            - all_metadatas: Metadados associados a cada parte do artigo.
    """

    all_chunks = []  # Lista para armazenar partes do conteúdo dos artigos.
    all_ids = []     # Lista para armazenar IDs dos artigos.
    all_metadatas = []  # Lista para armazenar metadados dos artigos.

    # Vai tentar por 3 vezes o acesso aos dados.
    for attempt in range(3):
        try:
            # Faz a requisição para obter os artigos no formato XML usando o Entrez
            handle = Entrez.efetch(db=db, id=idList, retmode="xml")
            data = (handle.read()).decode("UTF-8")  # Lê e decodifica os dados obtidos
            soup = BeautifulSoup(data, "xml")  # Analisa os dados XML com BeautifulSoup
            metadata = {}  # Dicionário para armazenar metadados do artigo
            content = ""  # Inicializa a variável de conteúdo

            # Define os separadores que serão usados para divisão do texto
            separators = [
                "#",
                "##",
                "\n"
            ]

            # Inicializa o divisor de texto para dividir o conteúdo em partes menores
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,  # Tamanho máximo de cada parte
                chunk_overlap=256,  # Número de caracteres que se sobrepõem entre partes
                separators=separators  # Separadores definidos anteriormente
            )

            # Procura por todos os artigos dentro do documento XML
            for article in soup.find_all('article'):
                content, metadata = parse_xml_article(article)  # Extrai conteúdo e metadados do artigo
                if content:
                    # Divide o conteúdo em partes menores
                    chunks = text_splitter.split_text(content)
                    # Cria IDs únicos para cada parte
                    ids = [f"{metadata['pmcid']}_{i}" for i in range(len(chunks))]
                    all_chunks.extend(chunks)  # Adiciona as partes ao resultado
                    all_ids.extend(ids)  # Adiciona os IDs ao resultado
                    metadata_template = {'text': 'Texto Padrão'}  # Template de metadado (se necessário)
                    # Duplica os metadados para cada parte
                    all_metadatas.extend([metadata.copy() for _ in range(len(chunks))])

            return all_chunks, all_ids, all_metadatas  # Retorna os resultados

        except Exception as e:
            print(f"Ocorreu um erro: {e}")  # Exibe a mensagem de erro
            if attempt < 2:
                print(f"Tentando novamente em 5 segundos...")  # Indica a tentativa de nova chamada
                time.sleep(5)  # Espera 5 segundos antes da próxima tentativa
            else:
                print(f"Não foi possível obter os dados após três tentativas.\n{idList}")  # Caso falhe após 3 tentativas
                return [], [], []  # Retorna listas vazias

def load_articles_pinecone(query, vector_store, total_articles=10000, batch_size=100, db="pmc"):
    """
    Carrega artigos da base de dados NCBI e os insere no armazenamento vetorial do Pinecone.

    Esta função realiza uma busca pelos artigos relacionados a um termo específico
    utilizando a API NCBI, processa os resultados e os armazena em um vetor
    utilizando o Pinecone.

    Args:
        query (str): Termo de busca para encontrar artigos na base de dados.
        vector_store (PineconeVectorStore): Instância do vetor de armazenamento onde os artigos serão inseridos.
        total_articles (int): Número total de artigos a ser recuperado (padrão é 10.000).
        batch_size (int): Número de artigos a serem processados por lote (padrão é 100).
        db (str): Nome da base de dados a ser consultada, padrão é "pmc".
    """

    # Obtém os termos MeSH relacionados à consulta e a consulta formatada
    qt, q = get_mesh_terms(query)

    # Realiza a busca na base de dados NCBI, limitando o número de artigos retornados
    results = search_ncbi(qt, retmax=total_articles, db=db, sort="pub_date ")

    # Itera sobre os resultados em lotes, atualizando a barra de progresso
    for i in tqdm(range(0, len(results['IdList']), batch_size), desc="Inserindo artigos"):
        # Extrai um lote de artigos da lista de IDs
        batch_articles = results['IdList'][i:i + batch_size]

        # Obtém o conteúdo, IDs e metadados dos artigos utilizando a função get_articles_batch
        chunks, ids, metadatas = get_articles_batch(idList=batch_articles, db=db)

        # Adiciona os artigos processados ao armazenamento vetorial no Pinecone
        vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="cancer"  )

load_articles_pinecone(query="cancer", vector_store=vector_store, total_articles=80000, batch_size=100, db="pmc")