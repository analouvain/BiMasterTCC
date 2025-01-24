# MedFusion - ChatBot inteligente

#### Aluno: [Ana Paula Louvain Marinho Costa](https://github.com/analouvain/BiMasterTCC/) - Mat: 221100813
#### Orientadora: [Evelyn Batista]([https://github.com/link_do_github](https://github.com/evysb)).
#### Co-orientador(/a/es/as): Leonardo Alfredo Forero Mendoza

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

https://github.com/analouvain/BiMasterTCC/

[app.py](https://github.com/analouvain/BiMasterTCC/blob/main/app.py)

[mypubmed_utils.py](https://github.com/analouvain/BiMasterTCC/blob/main/mypubmed_utils.py)

[load_articles_to_pinecone.py](https://github.com/analouvain/BiMasterTCC/blob/main/load_articles_to_pinecone.py)

Interface:
https://huggingface.co/spaces/analouvain/MedFusion

---

### Resumo

Este trabalho apresenta o desenvolvimento do **MedFusion**, um chatbot biomédico baseado em técnicas de **Retrieval-Augmented Generation (RAG)** para facilitar a recuperação e o uso de informações científicas. O chatbot opera em três modalidades: pesquisas relacionadas ao câncer, consultas biomédicas gerais e questões genéricas, integrando tecnologias como **LangChain, LangGraph, Pinecone** e modelos **LLM (OpenAI e Gemini).** Utilizando bases de dados como **PubMed** e **PMC**, o **MedFusion** permite a recuperação de artigos científicos e geração de respostas contextualizadas, otimizando a pesquisa em grandes volumes de informação. O sistema também inclui uma interface intuitiva desenvolvida em **Gradio**, disponível para testes no **Huggingfaces**. Este trabalho explora não apenas o desenvolvimento da ferramenta, mas também o potencial das técnicas de IA para domínios especializados, demonstrando a viabilidade e eficácia de soluções baseadas em RAG.

### 1. Introdução

A pesquisa em bases de dados científicas, como o PubMed e o PMC do NCBI, é um pilar essencial para o avanço do conhecimento biomédico. No entanto, a complexidade em realizar buscas eficientes em meio a milhões de artigos disponíveis representa um desafio significativo, especialmente para pesquisadores que precisam localizar informações precisas em áreas específicas como oncologia ou biomedicina geral. Essa abordagem motivou o desenvolvimento deste projeto, que propõe o desenvolvimento de  um **chatbot** inovador capaz de realizar pesquisas avançadas em artigos científicos, facilitando o acesso e a utilização de informações relevantes. Nesse contexto, a técnica de **Retrieval-Augmented Generation (RAG)** se destaca como uma solução promissora, ao combinar métodos de recuperação de informações com a geração de respostas contextualizadas e precisas em uma base de informação confiável. 

O chatbot desenvolvido opera em três modalidades principais: (1) pesquisas relacionadas ao câncer, utilizando artigos completos da base PMC, onde cerca de 80 mil artigos do PMC relacionados a cancer foram processados em chunks, armazenados como embeddings no Pinecone e são recuperados por similaridade para fornecer respostas baseadas em contexto; (2) pesquisas biomédicas gerais, que envolvem a extração de palavras-chave da consulta do usuário, mapeamento para termos MeSH, busca de resumos na base PubMed e uso de embeddings e busca por similaridade para contextualizar as respostas; e (3) pesquisas genéricas, nas quais o modelo de linguagem é usado diretamente para responder às consultas sem recuperação de contexto.

O projeto explora e integra frameworks e tecnologias de inteligência artificial como **LangChain, LangGraph, Pinecone, API BioPython, Hugging Face, Gradio,** além de modelos avançados de LLM como **OpenAI e Gemini**, onde é permitido ao usuário escolher o modelo a ser utilizado. A técnica de RAG é aplicada para otimizar a recuperação e a geração de respostas contextualizadas, enquanto o design de prompts eficientes é explorado para garantir que o modelo de linguagem compreenda e responda às consultas de forma relevante e precisa.

A proposta deste trabalho vai além do desenvolvimento de uma ferramenta prática; busca também avaliar e demonstrar o potencial de técnicas de inteligência artificial aplicadas à recuperação de informações em domínios especializados. 

### 2. Modelagem
Nesta seção, apresentarei a modelagem técnica do chatbot **MedFusion** desenvolvido para auxiliar nas pesquisas de artigos científicos. O objetivo será detalhar os aspectos conceituais e técnicos que embasam o funcionamento do chatbot, destacando como as tecnologias e técnicas aplicadas contribuem para a solução

O chatbot foi projetado para lidar com três modalidades principais de pesquisa que utilizam o modelo LLM  **OpenAI (3.5 turbo)** ou **Gemini (1.5 flash)**, escolhidos pelo usuário no momento da pesquisa:

- **Consultas relacionadas à câncer**, utilizando artigos completos da base **PMC**:
  Essa modalidade, envolve o desenvolvimento e execução de um script que realiza a recuperação de artigos do PMC através de uma pesquisa com termos MeSH associados a palavra chave **cancer**, segmenta os textos em chunks e armazena seus embeddings em uma base vetorial para posterior consulta por similaridade pelo chatbot na recuperação de contexto para fundamentar a resposta do LLM.
- **Pesquisas biomédicas gerais**, utilizando abstracts de artigos do **PubMed**:
  Essa modalidade envolve a extração de palavras-chave da pergunta realizada pelo usuário, mapeamento para termos MeSH, busca por resumos de artigos na base PubMed e geração de embeddings desses resumos e armazenamento dos mesmos em memória e consulta por similaridade para gerar contexto para fundamentar a resposta do LLM. 
- **Pesquisas genéricas**, nas quais as consultas são tratadas diretamente por um modelo de linguagem sem recuperação de contexto especializado.

A modelagem faz uso de uma combinação de tecnologias e frameworks, incluindo **LangChain, LangGraph, Pinecone, API BioPython, OpenAI, Gemini, Hugging Face e Gradio**, cada uma desempenhando um papel essencial na implementação da solução.  A técnica de **Retrieval-Augmented Generation (RAG)** é a base da arquitetura do chatbot, permitindo a combinação de recuperação de informações com geração de respostas personalizadas.
Segue abaixo um esquema de como a arquitetura foi projetada. Logo em seguida, irei explicar cada etapa do desenvolvimento do projeto.

![representação da arquitetura do chatbot MedFusion](https://raw.githubusercontent.com/analouvain/BiMasterTCC/refs/heads/main/images/rag_medfusion.png)


#### Acesso aos artigos científicos
Na raiz do repositório encontra-se um arquivo chamado `mypubmed_utils.py` que contém as funções desenvolvidas para acesso aos artigos que serão utilizadas nos demais scripts. 
Para acessar os artigos foi utilizado o pacote  [Bio.Entrez](https://biopython.org/docs/latest/api/Bio.Entrez.html) da  API [Bio Python](https://biopython.org/) que encapsula a API  [E-utilities](https://www.ncbi.nlm.nih.gov/home/develop/api/) do [NCBI](https://www.ncbi.nlm.nih.gov/)
Um dos grandes desafios da pesquisa em artigos biomédicos é o grande volume de artigos e retorno relevante. Para um melhor resultado de busca pelos artigos científicos, sejam eles do **PubMed** (nesse trabalho, utilizado para pesquisas biomédicas gerais) o do **PMC** (nesse trabalho para pesquisas relacionadas a cancer) foi utilizada a busca por termos **MeSH** para pesquisa. O uso de **termos MeSH (Medical Subject Headings)** justifica-se por sua capacidade de padronizar buscas em bases como PubMed e PMC, superando as limitações de variações linguísticas e sinônimos. Eles garantem que os resultados recuperados sejam relevantes e diretamente relacionados ao tema pesquisado, evitando informações irrelevantes. Além disso, sua estrutura hierárquica permite explorar conceitos específicos ou abrangentes, aumentando a precisão e a eficiência das pesquisas científicas. 

#### Recuperação dos Artigos Científicos Relacionados ao Câncer 
A recuperação dos artigos científicos relacionados ao câncer ocorre numa etapa idependente e anterior ao fluxo conversacional, atraves da execução do script `load_articles_to_pinecone.py` com o objetivo de disponibilizar os artigos em uma base de dados vetorial para posterior consumo pelo chatbot. 
 O código para recuperação dos artigos científicos relacionados ao câncer está disponível no arquivo `load_articles_to_pinecone.py`. A recuperação dos artigos ocorre seguindo as etapas descritas abaixo: 
1. **Identificação dos Termos MeSH**: Primeiramente, são identificados os termos MeSH relacionados à palavra "câncer" para realizar a pesquisa no PubMed Central (PMC), utilizando a função `get_mesh_terms(query)` do arquivo `mypubmed_utils.pyc`. 
2. **Consulta ao PMC**: A query, agora composta pelos termos MeSH, é usada para buscar os 80.000 artigos mais recentes no PMC, por meio da função `search_ncbi(qt, retmax=total_articles, db=db, sort="pub_date")`. 
3. **Recuperação dos Artigos**: Com a lista de IDs dos artigos, em lotes de 100, os artigos completos são recuperados utilizando a função `get_articles_batch(idList=batch_articles, db=db)`. 
4. **Tratamento dos Artigos**: Cada artigo é retornado na íntegra em formato XML. Utilizando a biblioteca **BeautifulSoup**, o conteúdo do artigo XML é processado, permitindo a extração das informações relevantes e a limpeza dos dados. Nesse estágio, realiza-se um extenso tratamento do texto para deixá-lo pronto para vetorização e posterior uso como contexto de busca. Para facilitar o particionamento dos dados, o conteúdo dos arquivos é segmentado com marcações **Markdown**. 
5. **Identificação de Metadados**: Durante o processamento, também são identificados os metadados relevantes, como título, autores, resumo, etc. 
6. **Particionamento dos Dados**: Com o conteúdo tratado e limpo, os artigos são particionados em "chunks" usando a função `RecursiveCharacterTextSplitter` do **LangChain**. Esses "chunks" são armazenados junto com os metadados em uma lista. 
7. **Armazenamento no Pinecone**: Finalmente, os "chunks" e os metadados dos artigos de cada lote são inseridos na base de dados vetorial **Pinecone**, tornando-os acessíveis para futuras consultas, funcionando como contexto para o modelo de linguagem (LLM).
- Abaixo a representação gráfica do fluxo descrito acima:
![Recuperação dos artigos relacionados ao cancer](https://raw.githubusercontent.com/analouvain/BiMasterTCC/4ef20dcb553cdc3d217c80cb4783ef0406cafdce/images/recuperacao_artigos_cancer.png)
- Abaixo a configuração do Pinecone com o namespace indexado:
![](https://raw.githubusercontent.com/analouvain/BiMasterTCC/refs/heads/main/images/pinecone_database.png)

#### Fluxo Conversacional do Chatbot 
Conforme já mencionei, chatbot desenvolvido opera em três modalidades: (1) pesquisas relacionadas ao câncer, utilizando artigos completos da base **PMC**, onde cerca de 80 mil artigos do **PMC** relacionados a cancer foram processados em chunks, armazenados como embeddings no **Pinecone** e são recuperados por similaridade para fornecer respostas baseadas em contexto; (2) pesquisas biomédicas gerais, que envolvem a extração de palavras-chave da consulta do usuário, mapeamento para termos **MeSH**, busca de resumos na base **PubMed** e uso de embeddings e busca por similaridade para contextualizar as respostas; e (3) pesquisas genéricas, nas quais o modelo de linguagem é usado diretamente para responder às consultas sem recuperação de contexto. Além disso é permitido que o usuário escolha do modelo de **LLM** que irá utilizar para responder suas questões. 
O código do  fluxo do chatbot, bem como a recuperação dos artigos científicos biomédicos em geral e a interface gráfica estão descritos no arquivo `app.py`. A interface gráfica, desenvolvida em Gradio para Huggingfaces será explicada em um tópico posterior. Irei me concentrar agora a explicar como funciona o backend do fluxo conversacional. 
Para desenvolvimento do fluxo conversasional utilizei o framework **LangChain** que foi projetado para facilitar o desenvolvimento de aplicativos baseados em modelos de linguagem (LLM). Ele fornece ferramentas e integrações para construir fluxos de trabalho complexos, combinando modelos de linguagem com fontes de dados externas, memória e lógica personalizada. Também foi utilizado o **LangGraph** que é uma extensão do LangChain que nos permite criar, manter e gerenciar o fluxo de trabalho a partir de um grafo de estado. 
Antes de explicar o passo a passo das etapas do desenvolvimento, acho pertinente falarmos sobre a arquitetura do nosso grafo de estado. 
A figura abaixo representa o grafo que foi criado no arquivo `app.py` através da função `build_state_graph()`
![](https://raw.githubusercontent.com/analouvain/BiMasterTCC/refs/heads/main/images/grafo.png)
(*imagem gerada pelo próprio LangGraph com o seguinte código (app é a instância do meu grafo compilado):* `display(Image(app.get_graph().draw_mermaid_png()))`)

 1. **Criação do grafo de estado usando LangGraph**: A criação do grafo de estado está implementada na função `build_state_graph()`e ocorre seguindo as seguintes etapas: 
 - *Definição da Classe que representará o estado do grafo*:

    ```python
           class  ChatAgentState(MessagesState):
	            query_type: str  # O tipo da consulta do usuário, 'pesquisa sobre câncer', 'pesquisa biomédica geral' e 'geral'
	            
	            simplified_query: str  # Versão simplificada da consulta do usuário, focada em palavras chaves, usada para pesquisa no PubMed
	            
	            user_query: str  # A consulta original enviada pelo usuário.
	            
	            user_query_translated: str  # A consulta do usuário traduzida.
	            
	            chat_model: BaseChatModel  #modelo de LLM que será utilizadoO
- *O estado do grafo será passado de nó em nó armazenando os valores dos atributos da classe ChatAgentState*
- *Definição dos nós e arestas do grafo*:

  ```python
    workflow = StateGraph(state_schema=ChatAgentState)
    # Define as transições do grafo
    workflow.add_edge(START, "router") # Transição inicial para o nó router
    
    # Adiciona o nó de roteamento
    workflow.add_node("router", route_input)
    # Define as transições condicionais baseadas no tipo de consulta
    workflow.add_conditional_edges("router", get_route, # Função para determinar o próximo estado
    {  CANCER_RESEARCH: 'cancer_research',
       GENERAL_MEDICAL_RESEARCH: 'general_medical_research',
       GENERAL_QUESTION: 'general_question'
    } ) 
    # Adiciona os nós de processamento e suas transições
    workflow.add_node("cancer_research", perform_cancer_research)
    workflow.add_edge("cancer_research", END)
    workflow.add_node("general_medical_research", perform_general_medical_research)
    workflow.add_edge("general_medical_research", END)
    workflow.add_node("general_question", call_llm)
    workflow.add_edge("general_question", END)

- *Explicação*: `workflow = StateGraph(state_schema=ChatAgentState)`  cria o grafo setando o estado como `ChatAgentState`; `workflow.add_edge(<nó1>, <nó2>)`cria uma aresta do nó 1 para o nó 2; `workflow.add_node(<nome do nó>, <função que será executada pelo nó>)` cria um nó; `workflow.add_conditional_edges` cria uma aresta condicional que recebe como parametros: Nó de origem, função de condição que retorna os valores das condições, dicionário que contem pares de valor da condição e o nó de destino. 
- *Persistência*: Para que o histórico conversacional se mantenha durante toda conversa (inclusive quando o usuário muda o modelo de llm no meio da conversa), utlizamos uma instancia da classe `MemorySaver` e passamos ela como `checkpointer` na compilação do grafo. *É importante ressaltar que a cada nó os parametros do estado devem atualizados corretamente*

    ```python
        # Adiciona um sistema de memória para salvar o estado
        memory = MemorySaver()
        # Compila o fluxo de trabalho com o checkpoint
        app = workflow.compile(checkpointer=memory)
 - Inicialiação do estado do grafo: Ainda na função `build_state_graph()`, o grafo é atualizado com os valores default de início, que é essencialmente o modelo de LLM que irá ser utilizado de ínicio, a menos que o usuário mude na interface gráfica. 
   ```python
     #Estado inicial do Agente
    inital_state = { "messages" : [],
    "query_type": "",
    "simplified_query": "",
    "user_query": "",
    "user_query_translated": "",
    "chat_model" : llm_openai
    }
    #Atualizando o grafo com o estado inicial
    app.update_state(config=config, values=inital_state)

2.  **Início do fluxo: Roteamento**: Implementado pela função `route_input(state: ChatAgentState)`. Essa é a etapa inicial do fluxo que recebe a query do usuário e a envia para o modelo de LLM (setado no estado do grafo) com um prompt bastante detalhado e com exemplos, solicitando que a partir da query do usuário o modelo: 1) Traduza a query para o inglês; 2) Simplifique a query utilizando palavras chaves para que a mesma possa ser utilizada para as buscas de termos MeSH do NCBI. 3) Classifique a query em um das 3 categorias:  
- CANCER_RESEARCH: Para pesquisas relacionadas ao câncer.
- GENERAL_MEDICAL_RESEARCH: Para pesquisas biomédicas não relacionadas ao câncer.
- GENERAL_QUESTION: Para pesquisas genéricas não relacionadas a temas biomédicos.
Para facilitar o recebimento das variáveis de retorno solicitadas pelo prompt e garantir que a resposta do modelo **LLM** siga um padrão, utilizei uma **cadeia** do **LangChain** com o prompt e o modelo com **saída estruturada** da seguinte forma:

    ```python
    # Configura o modelo LLM para saída estruturada usando a classe Router
    structured_llm = chat_model.with_structured_output(Router)
    # Combina o prompt e o modelo em uma cadeia
    chain = prompt_router | structured_llm
    # Invoca o modelo com a consulta do usuário
    result = chain.invoke(user_query)

- Dessa forma é garantido que o retorno da execução da cadeia é um objeto estrturado do tipo `Router()`
 - A seguir segue a classe Router que tem exatamente as variáveis que foram solitadas ao modelo LLM. Ela é do tipo `BaseModel` que o que o metódo `with_structured_output` espera receber para fazer as validações necessárias.

     ```python
     class  Router(BaseModel):
    query_type: str = Field(description="Query type. It can assume the values: CANCER_RESEARCH, GENERAL_MEDICAL_RESEARCH or GENERAL_QUESTION")
    simplified_query: str = Field(description="Simplified query")
    user_query: str = Field(description="Original user query")
    user_query_translated: str = Field(description="Original user query translated to English") 

3. **Execução de pesquisa genérica não relacionada a pesquisas biomédicas**: Essa etapa é implementada pela função `call_llm(state: ChatAgentState)`. Aqui o modelo de LLM setado no estado do grafo é recuperado e simplesmente chamado com a query original do usuário com uma pequena instrução, informando que a função do chatbot nesse momento é ser um assistente e responder questões genéricas desde que ele possua certeza sobre sua resposta.

4. **Execução de pesquisas biomédicas não relacionadas ao câncer.**: Essa etapa é implementada pela função `perform_general_medical_research(state: ChatAgentState)`. 
- Nessa etapa, o modelo de LLM e a query simplificada são recuperados do estado do grafo (a query simplificada foi setada no nó anterior, Roteamento) 
- A query simplificada é passada para a etapa **Recuperação dos Artigos Científicos Biomédicos em geral** , detalhada posteriomente, que em linhas gerais obtem os termos **MeSH** da query simplificada, busca os abstracts dos artigos no **PubMed**, vetoriza e os armazena em um **InMemoryVectorStore** , uma vector store em **memória**
- Um busca por similaridade com a query simplificada é realizada sobre o **InMemoryVectorStore**
- Com o resultado da busca anterior o contexto é construído
- O prompt para o LLM é elaborado com o contexto
- O modelo de LLM é chamado e o resultado retornado.

5 . **Execução de pesquisas biomédicas relacionadas ao câncer.**: Essa etapa é implementada pela função `perform_cancer_research(state: ChatAgentState)`. 
- Nessa etapa, o modelo de LLM e a query simplificada são recuperados do estado do grafo (a query simplificada foi setada no nó anterior, Roteamento) 
- Um busca por similaridade com a query simplificada é realizada sobre o `cancer_vector_store` que foi definido como variável global e conectado ao namespace do **Pinecone** que foi populado na etapa de **Recuperação dos Artigos Científicos Relacionados ao Câncer** já detalhada anteriormente. 
- Com o resultado da busca anterior o contexto é construído
- O prompt para o LLM é elaborado com o contexto
- O modelo de LLM é chamado e o resultado retornado.

#### Recuperação dos Artigos Científicos Biomédicos em geral
Diferente da recuperação de artigos médicos relacionados ao câncer que ocorre em um momento anterior a execução do fluxo conversacional do chatbot através da execução de um script, a recuperação dos artigos científicos biomédicos é realizada como parte do fluxo conversacional do chatbot **em tempo de execução**. 
O processo de recuperação ocorre nas seguintes etapas: 
1. **Recepção da Query**: Considerando que já houve um processamento da query anteriormente, vamos detalhar que essa etapa recebe uma query peviamente processada, relacionada a um tema biomédico, em inglês e simplificada em palavras chave, que chamaremos de *query simplificada*. 
2. **Identificação dos Termos MeSH**: São identificados os termos MeSH relacionados à *query simplificada* para realizar a pesquisa no PubMed, utilizando a função  `get_mesh_terms(query)`  do arquivo  `mypubmed_utils.pyc`.
3. **Busca no PubMed**: A query com os termos MeSH é utilizada para buscar os artigos mais relevantes no PubMed. A busca retorna um máximo de 15 artigos, que são filtrados por relevância, utilizando o sistema de classificação do PubMed. 
4. **Recuperação dos Abstracts**: Apenas os resumos (abstracts) dos artigos são recuperados, em texto já limpo, juntamente com metadados relevantes como título, data de publicação, autores, etc. Como os abstracts são pequenos, não há necessidade de particionamento adicional dos textos. 
5. **Armazenamento dos Vetores**: Os abstracts dos artigos são transformados em vetores (embeddings e metadados) Os vetores resultantes da vetorização dos abstracts são armazenados em memória, utilizando o **InMemory Vector Store** de **LangChain**. Isso permite que os artigos e seus resumos fiquem disponíveis de forma eficiente para consultas subsequentes, proporcionando uma resposta imediata durante a interação com o chatbot. 
6. **Uso no Fluxo Conversacional**: Durante o uso do chatbot, esses vetores armazenados são consumidos imediatamente, permitindo que o chatbot forneça ao usuário informações relevantes e atualizadas sobre os temas biomédicos consultados.

#### Interface Gráfica & Execução do Chatbot
O Chatbot MedFusion está hospedado e disponível para uso e testes no Huggingfaces no seguinte endereço:

**https://huggingface.co/spaces/analouvain/MedFusion**


A interface do MedFusion ChatBot é exibida a seguir e é bem intuitiva, típica de aplicações de chatbots, foi desenvolvida com Gradio do Huggingfaces.
As variáveis de ambiente foram setadas no space do Huggingfaces. 
Sempre que uma pesquisa utilizando RAG sobre artigos científicos for realizada, o modelo retorna a informação dos artigos que foram levados em consideração para a resposta e o link para o mesmo em seu repositório original (site do NCBI). 

O usuário pode escolher através dos radio buttons o modelo que ele quer usar no chatbot, e o objetivo dessa implemetação foi testar, aprender e utilizar mais de um modelo. 
![](https://raw.githubusercontent.com/analouvain/BiMasterTCC/refs/heads/main/images/med_fusion_interface.png)


Sempre que o chatbot responde uma pergunta, antes da resposta aparece o tipo da questão (por onde ela foi respondida e qual o modelo de LLM que foi utilizado na resposta. 
![](https://github.com/analouvain/BiMasterTCC/blob/main/images/exemplo_busca.png?raw=true)

Para utilizar algumas queries de exemplo, basta selecionar o exemplo e depois apertar o botão de enviar.

![](https://github.com/analouvain/BiMasterTCC/blob/main/images/exemplo_busca_exemplo.png?raw=true)

Uma boa forma de testar a persistência do histórico é realizar algumas perguntas e depois pedir para o chatbot citar palavras chaves da conversa. ;) 
![](https://github.com/analouvain/BiMasterTCC/blob/main/images/exemplo_busca_memoria.png?raw=true)


### 3. Resultados


1. **Efetividade nas Respostas**: 
- O chatbot respondeu adequadamente a maioria das consultas realizadas em testes, fornecendo informações relevantes e precisas, conforme as orientações dos prompts de comando. 
- Em consultas relacionadas ao câncer, as respostas demonstraram forte embasamento no contexto recuperado da base PMC, porém quando não mencionado o tipo de câncer ele as vezes acabava respondendo sobre um tipo específico. 
- Para pesquisas biomédicas gerais, a busca por abstracts no PubMed resultou em respostas que cobriram os principais tópicos das consultas de forma satisfatória. 

2. **Desempenho Técnico**: 
- O tempo médio para geração de respostas eu ainda considero elevado, de 7 a 10 segundos para os 3 tipos de perguntas.   
- Acredito que o tempo mais elevado para geração das respostas se deve ao fato de uma chamada ao modelo para categorização da pergunta, porém pela automação da categorização e "switch" automático do RAG utilizado eu considero satisfatório. 
- O sistema não demonstrou queda significativa de perfomance das pesquisas que envolviam o RAG para as genéricas sem recuperação de contexto, o que é um excelente resultado. 
- O sistema demonstrou capacidade de armazenar e processar vetores em memória para até 15 artigos simultaneamente sem comprometer a performance. (No caso de pesquisas biomédicas não relacionadas ao câncer)
- O modelo Gemini 1.5 flash me pareceu ter uma perfomance discretamente pior do que o da OpenAI, principalmente para obeceder as instruções do prompt. 

3. **Usabilidade e Experiência do Usuário**: 
- A interface gráfica do Gradio atendeu bem as expectativas sobre a clareza das informações, facilidade de navegação e fácil implementação. 
- A persistência do histórico conversacional funcionou como esperado, permitindo mudanças no modelo de LLM sem perda de contexto. 

### 4. Conclusões

1. **Validação do Projeto**: 
- O MedFusion demonstrou ser uma ferramenta funcional e inovadora para a recuperação de informações biomédicas, combinando técnicas de RAG com modelos avançados de linguagem e acredito de muita valia para pesquisadores ou leigos com interesse na área biomédica. 
- A divisão em três modalidades de pesquisa permitiu um atendimento eficaz para diferentes tipos de consultas e ampliou o uso do chatbot que pode ser multifuncional. 

- 2. **Contribuições Relevantes**: 
- A integração de tecnologias como LangChain, LangGraph, Pinecone e LLMs mostrou-se eficaz na criação de um fluxo de trabalho robusto para recuperação e geração de respostas baseadas em contexto. 
- A adoção de termos MeSH para busca aumentou significativamente a relevância dos artigos recuperados. 

3. **Impacto e uso futuro**: 
- Ferramentas como o MedFusion podem ser adaptadas para áreas além da biomedicina, como direito, engenharia e educação, otimizando o acesso a informações críticas em diferentes domínios. 

4. **Melhorias Futuras**: 
- Implementação de novos modelos de linguagem mais avançados, como GPT-4 ou modelos especializados em biomedicina. 
- Expansão da base de dados para incluir outras fontes confiáveis além de PubMed e PMC. 
- Implementação de mecanismos de avaliação das respostas para identificar relevância e precisão automaticamente. 

5. **Desafios e Aprendizados**: 
- A dificuldade em lidar com consultas excessivamente específicas ressalta a importância de desenvolver métodos para enriquecer os dados de entrada, como expansão de consultas. 
- A experiência reforça o papel crucial da curadoria de dados e do design de prompts na construção de sistemas conversacionais eficazes.
- O meu desafio pessoal com esse projeto, que foi mais do entregar uma solução e sim aprender e testar novas técnicas, frameworks, modelos e tecnologias que são oferecidas para lidar com soluções de LLM foi alcançado. Aprendi a usar várias técnicas e tecnologias como: 
	- LangChain
	- LangGraph
	- Grafos de estado com condicionais
	- Pinecone
	- API Bio Python 
	- Tratamento e limpeza de dados com BeautifulSoup para preparar contexto eficaz para LLM 
	- OpenAI 
	- Gemini 
	- Resultado de LLM com dados estruturados
	- Persistência de histórico conversacional
	- Armazenamento de vetores em memória
	- Huggingfaces
	- Gradio
	- Desenvolvimento de prompts eficazes 
	- Logging

---

Matrícula: 221100813

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*




