# Importações necessárias
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from Bio import Entrez
from Bio import Medline
from langchain_core.documents import Document

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv('env', override=True)

# Carrega variáveis de ambiente sensíveis
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
OPENAI_API_CHAT_MODEL = os.getenv("OPENAI_API_CHAT_MODEL")
OPENAI_API_EMBEDDING_MODEL = os.getenv("OPENAI_API_EMBEDDING_MODEL")
OPENAI_API_EMBEDDING_DIM = os.getenv("OPENAI_API_EMBEDDING_DIM")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
Entrez.email = ENTREZ_EMAIL

# Função para obter termos MeSH (Medical Subject Headings) a partir de uma consulta
def get_mesh_terms(query):
    """
    Acessa a API BioPython e recupera os termos MeSH relevantes para uma consulta fornecida.

    Args:
        query (str): Consulta de busca.

    Returns:
        tuple: Tradução da consulta em termos MeSH 
        e a consulta formatada contendo somente os termos Mesh.
    """
    handle = Entrez.esearch(db="mesh", term=query)
    record = Entrez.read(handle)
    handle.close()

    mesh_terms = []
    for translation in record['TranslationSet']:
        terms = translation['To'].split(' OR ')
        for term in terms:
            if '[MeSH Terms]' in term:
                mesh_terms.append(term.replace('[MeSH Terms]', '').replace('"', '').strip())

    query_terms = [f"{term}" for term in mesh_terms]
    query = " AND ".join(query_terms)
    query_translation = record['QueryTranslation']
    return query_translation, query

# Função para buscar artigos em bancos de dados NCBI (ex.: PubMed)
def search_ncbi(query, db='pmc', sort='relevance', retmax='20'):
    """
    Realiza uma busca no NCBI com base na consulta fornecida.

    Args:
        query (str): Consulta de busca.
        db (str): Banco de dados a ser usado (ex.: 'pubmed', 'pmc').
        sort (str): Critério de ordenação (ex.: 'relevance').
        retmax (int): Número máximo de resultados.

    Returns:
        dict: Resultados da busca.
    """
    Entrez.email = ENTREZ_EMAIL
    handle = Entrez.esearch(db=db, sort=sort, retmax=retmax, term=query)
    results = Entrez.read(handle)
    return results

# Função para buscar abstracts com base em uma lista de IDs
def fetch_abstracts(idlist, db='pubmed'):
    """
    Busca abstracts no banco de dados fornecido com base em uma lista de IDs.

    Args:
        idlist (list): Lista de IDs dos artigos.
        db (str): Banco de dados a ser usado (padrão: 'pubmed').

    Returns:
        list: Lista de abstracts recuperados.
    """
    ids = ','.join(idlist)
    handle = Entrez.efetch(db=db, id=ids, rettype="medline", retmode="text")
    results = Medline.parse(handle)
    results = list(results)
    handle.close()
    return results

# Função para lidar com listas de autores ou dados semelhantes
def handle_list(value):
    """
    Converte uma lista em uma string separada por vírgulas, se aplicável.

    Args:
        value (list or any): Valor a ser processado.

    Returns:
        str: String concatenada se for uma lista, caso contrário retorna o valor original.
    """
    if isinstance(value, list) and all(x is not None and x == x for x in value):
        return ', '.join(map(str, value))
    return value

# Função para recuperar abstracts do PubMed e criar um contexto baseado em similaridade
def retrieve_pubmed_abstracts(query, retmax=50, num_articles=10):
    """
    Recupera abstracts do PubMed e cria um contexto de busca baseado em similaridade.

    Args:
        query (str): Consulta de busca.
        retmax (int): Número máximo de resultados a recuperar.
        num_articles (int): Número de artigos mais semelhantes a retornar.

    Returns:
        str: Contexto formatado com abstracts e metadados relevantes.
    """
    # Obtém a tradução da consulta em termos MeSH
    query_translation = get_mesh_terms(query)[0]

    # Busca IDs de artigos relacionados
    id_list = search_ncbi(query=query_translation, db='pubmed', sort='relevance', retmax=retmax)['IdList']

    # Recupera abstracts com base nos IDs
    abstract_list = fetch_abstracts(idlist=id_list, db='pubmed')

    # Cria documentos a partir dos abstracts
    document_list = []
    for abstract in abstract_list:
        document = Document(
            id=abstract.get("PMID", "?"),
            page_content=f"{abstract.get('TI', '?')}\n{abstract.get('AB', '?')}",
            metadata={
                "pmid": abstract.get("PMID", "?"),
                "title": abstract.get("TI", "?"),
                "abstract": abstract.get("AB", "?"),
                "author": abstract.get("AU", "?"),
                "date_of_publication": abstract.get("DP", "?"),
                "link": f"https://pubmed.ncbi.nlm.nih.gov/{abstract.get('PMID', '?')}/",
            },
        )
        document_list.append(document)

    # Cria uma loja vetorial em memória usando embeddings
    pubmed_vector_store = InMemoryVectorStore(OpenAIEmbeddings(model=OPENAI_API_EMBEDDING_MODEL))

    # Adiciona os documentos ao vetor
    pubmed_vector_store.add_documents(documents=document_list)

    # Realiza uma busca de similaridade
    results = pubmed_vector_store.similarity_search_with_score(query=query_translation, k=num_articles)

    # Formata o contexto com os resultados
    context_list = []
    for doc, score in results:
        context_list.append(f"#Title: {doc.metadata['title']}\n")
        context_list.append(f"##Abstract\n{doc.metadata['abstract']}\n")
        context_list.append(f"author(s): {handle_list(doc.metadata['author'])}\n")
        context_list.append(f"date of publication: {doc.metadata['date_of_publication']}\n")
        context_list.append(f"link: {doc.metadata['link']}\n\n")

    # Retorna o contexto formatado
    context = "".join(context_list)
    return context
