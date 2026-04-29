# Atividade Prática - Pipeline RAG com ChromaDB

Este projeto implementa uma aplicação RAG básica para consulta da Resolução nº 001/2024 - CONEPE, que institui a Normatização Acadêmica da Universidade do Estado de Mato Grosso Carlos Alberto Reyes Maldonado - UNEMAT.

## Documento utilizado

O documento escolhido para indexação foi:

- `Res 001-2024-CONEPE - Normatização Acadêmica.pdf`

Esse documento foi escolhido por ter estrutura normativa clara, com títulos, capítulos, artigos, parágrafos e incisos. Essa organização favorece uma estratégia de chunking estrutural, pois cada trecho possui uma unidade semântica bem definida.

## Objetivo

Criar um pipeline RAG básico capaz de:

1. Carregar o documento PDF.
2. Dividir o texto em chunks adequados.
3. Gerar embeddings dos chunks.
4. Armazenar os embeddings no ChromaDB.
5. Receber uma pergunta do usuário.
6. Buscar os trechos mais relevantes por similaridade semântica.
7. Gerar uma resposta fundamentada no contexto recuperado.
8. Citar as fontes utilizadas na resposta.

## Tecnologias escolhidas

### Backend

O backend será desenvolvido em **Python**, seguindo a mesma linguagem usada nos exemplos da aula.

O backend será responsável por:

- carregar e extrair o texto do PDF;
- limpar cabeçalhos e rodapés;
- aplicar o chunking por artigos;
- gerar embeddings;
- armazenar os dados no ChromaDB;
- recuperar os chunks relevantes;
- chamar o modelo gerador para produzir a resposta final.

Essa escolha mantém o projeto simples e alinhado ao conteúdo apresentado pelo professor.

### Frontend

O frontend será desenvolvido com **Streamlit**.

Motivos:

- Usa Python, evitando a necessidade de HTML, CSS e JavaScript manual.
- Permite criar uma interface web simples e funcional rapidamente.
- É adequado para demonstrações acadêmicas e protótipos de IA.
- Integra diretamente com o backend Python.
- Facilita criar uma tela com campo de pergunta, botão de consulta, resposta e fontes.

A interface terá uma estrutura simples:

```text
Título da aplicação
Botão para indexar o documento
Campo para digitar a pergunta
Botão para consultar
Área de resposta
Lista de fontes recuperadas
```

Exemplo de uso esperado:

```text
Pergunta:
O que acontece se o estudante perder o prazo de renovação de matrícula?

Resposta:
O estudante poderá recorrer ao Colegiado de Curso mediante justificativa, no prazo máximo de até 10 dias após o encerramento do último período de matrícula.

Fontes:
- Art. 35, Capítulo III - Da Renovação da Matrícula, página 8.
```

### Banco vetorial

O banco vetorial escolhido foi o **ChromaDB**.

Motivos:

- É gratuito e simples de usar localmente.
- Atende diretamente ao requisito da atividade.
- Permite persistir coleções em disco.
- Armazena texto, embeddings e metadados no mesmo fluxo.
- É adequado para projetos pequenos e demonstrações acadêmicas de RAG.

### Modelo de embeddings

Para gerar embeddings, a escolha recomendada é usar a API do **Gemini Embedding**, especialmente:

- `gemini-embedding-001`.

Também existe a opção:

- `gemini-embedding-2`.

Motivos:

- Possui camada gratuita para uso experimental.
- Funciona bem com busca semântica.
- Suporta português.
- Evita depender de uma API paga da OpenAI para a etapa de embeddings.

Para este projeto, a recomendação principal é usar **`gemini-embedding-001`**, pois a aplicação trabalha com RAG textual e não precisa de recursos multimodais.

Os embeddings são usados para transformar cada chunk textual em um vetor numérico. Esses vetores são armazenados no ChromaDB e comparados com o vetor da pergunta do usuário durante a busca semântica.

### Modelo gerador de resposta

Para a geração da resposta final, a escolha recomendada é usar a **GroqCloud** com o modelo:

- `llama-3.3-70b-versatile`.

Motivos:

- Possui free tier adequado para testes e demonstrações.
- Costuma oferecer bons limites gratuitos para chamadas de chat.
- Tem baixa latência.
- Funciona bem para gerar respostas em português quando recebe um prompt claro.

Também é possível usar:

- **Gemini Flash**, usando a mesma plataforma dos embeddings.

No entanto, para maximizar o uso gratuito, a recomendação final é:

```text
Embeddings: gemini-embedding-001
Geração: GroqCloud com llama-3.3-70b-versatile
```

Assim, o Gemini fica responsável pela criação dos vetores semânticos e a GroqCloud fica responsável pela resposta textual final.

## Estrutura prevista do projeto

A estrutura sugerida para o projeto é:

```text
.
|-- app.py
|-- rag.py
|-- requirements.txt
|-- .env
|-- README.md
|-- chroma_db/
|-- uploaded_docs/
|-- Aula-07-Introducao-RAG.pdf
`-- Res 001-2024-CONEPE - Normatização Acadêmica.pdf
```

Descrição dos arquivos:

- `app.py`: interface web feita com Streamlit.
- `rag.py`: lógica principal do pipeline RAG.
- `requirements.txt`: dependências do projeto.
- `.env`: chaves de API do Gemini e da GroqCloud.
- `chroma_db/`: pasta onde o ChromaDB persistirá os embeddings.
- `uploaded_docs/`: pasta local onde a interface salva o PDF carregado pelo usuário.
- `README.md`: documentação do projeto.

## Separação entre frontend e backend

Mesmo usando Streamlit, o projeto será organizado separando a interface da lógica de RAG.

O arquivo `app.py` cuidará apenas da interface:

- exibir título;
- mostrar botões;
- receber o upload do PDF;
- receber a pergunta;
- chamar funções do backend;
- apresentar resposta e fontes.

O arquivo `rag.py` cuidará da lógica:

- extração do PDF;
- chunking;
- embeddings;
- ChromaDB;
- recuperação semântica;
- geração da resposta.

Essa separação deixa o projeto mais organizado e facilita explicar ao professor onde está cada parte da aplicação.

## Estratégia de chunking

A estratégia escolhida é:

> Chunking por seções aplicado a documento normativo, usando artigos como unidade semântica principal.

De acordo com a classificação apresentada na aula, essa estratégia se encaixa em **chunking por seções**, pois utiliza marcadores estruturais do documento, como:

- `TÍTULO`
- `Capítulo`
- `Art.`
- parágrafos
- incisos

No entanto, em vez de criar um chunk para cada título ou capítulo inteiro, a unidade principal será o **artigo**.

## Justificativa do chunking por artigo

Usar títulos ou capítulos inteiros como chunks deixaria os blocos muito grandes e misturaria assuntos diferentes. Isso poderia prejudicar a busca semântica, porque uma pergunta específica sobre matrícula, frequência ou avaliação poderia recuperar um bloco amplo demais.

Por outro lado, usar apenas parágrafos ou sentenças poderia fragmentar demais o conteúdo e separar incisos, parágrafos e complementos que fazem parte do mesmo artigo.

Por isso, a melhor unidade para este documento é:

```text
1 chunk = 1 artigo completo
```

Cada artigo deve carregar junto seus parágrafos, incisos e complementos até o início do próximo artigo.

## Metadados dos chunks

Cada chunk deve ser armazenado no ChromaDB com metadados que permitam rastrear sua origem.

Exemplo:

```json
{
  "fonte": "Res 001-2024-CONEPE - Normatização Acadêmica.pdf",
  "pagina_inicio": 8,
  "pagina_fim": 8,
  "titulo": "TÍTULO V - DA VIDA ACADÊMICA",
  "capitulo": "Capítulo III - Da Renovação da Matrícula",
  "artigo": "Art. 35"
}
```

Esses metadados serão usados para exibir as citações na resposta final.

## Formato recomendado do texto indexado

O texto enviado para embedding deve conter o conteúdo do artigo e também seu contexto hierárquico.

Exemplo:

```text
Fonte: Res 001-2024-CONEPE - Normatização Acadêmica.pdf
TÍTULO V - DA VIDA ACADÊMICA
Capítulo III - Da Renovação da Matrícula
Página: 8

Art. 35 O estudante que perder o prazo de Renovação de Matrícula poderá recorrer ao Colegiado de Curso...
```

Esse formato melhora a recuperação porque o embedding passa a representar não apenas o artigo isolado, mas também o assunto institucional ao qual ele pertence.

## Fluxo do pipeline RAG

O pipeline da aplicação será:

1. Carregar o PDF pela interface Streamlit.
2. Salvar o PDF carregado na pasta local `uploaded_docs/`.
3. Extrair o texto do PDF com `PyMuPDF`.
4. Limpar cabeçalhos, rodapés e numeração de página.
5. Identificar marcadores estruturais do documento.
6. Separar o texto em chunks por artigo.
7. Associar cada chunk aos metadados de título, capítulo, artigo e página.
8. Gerar embeddings com Gemini Embedding.
9. Armazenar texto, embeddings e metadados no ChromaDB.
10. Ao receber uma pergunta, gerar embedding da pergunta.
11. Buscar no ChromaDB os chunks mais similares.
12. Enviar pergunta e chunks recuperados ao modelo gerador.
13. Gerar resposta usando apenas o contexto recuperado.
14. Apresentar resposta com citação das fontes.
15. Exibir tudo em uma interface Streamlit.

## Fluxo da interface

O fluxo de uso da aplicação será:

1. O usuário abre a aplicação Streamlit.
2. O usuário carrega o documento PDF pela interface.
3. O usuário clica em **Processar e indexar PDF**.
4. A aplicação divide o PDF em chunks, gera embeddings e salva os dados no ChromaDB.
5. O usuário digita uma pergunta sobre a Resolução CONEPE.
6. A aplicação busca os chunks mais relevantes.
7. A aplicação gera uma resposta fundamentada.
8. A interface exibe a resposta e as fontes usadas.

A consulta só é liberada depois que o PDF é processado e indexado na sessão atual. O ChromaDB persiste dados em disco na pasta `chroma_db/`, mas a interface exige o processamento explícito do PDF para deixar claro o fluxo pedido na atividade.

A interface também mostra quantos chunks estão armazenados no ChromaDB. Caso exista um índice antigo, o usuário pode clicar em **Limpar índice ChromaDB** antes de processar outro PDF. Ao clicar em **Processar e indexar PDF**, o índice anterior é substituído, evitando acúmulo de documentos e embeddings.

## Dependências previstas

As principais dependências do projeto serão:

```text
streamlit
chromadb
pymupdf
python-dotenv
google-genai
groq
```

As dependências devem ser instaladas dentro de um ambiente virtual `.venv`, para evitar conflitos com bibliotecas de outros projetos Python.

## Ambiente virtual

O projeto usa um ambiente virtual Python na pasta `.venv/`.

Essa pasta não deve ser enviada para o Git, pois contém arquivos locais da instalação. Por isso, ela deve estar no `.gitignore`.

Para criar o ambiente virtual:

```powershell
python -m venv .venv
```

Para ativar no PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação por política de execução, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Com o ambiente virtual ativo, instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

## Variáveis de ambiente

As chaves de API devem ficar em um arquivo `.env`, evitando deixar dados sensíveis diretamente no código.

Exemplo:

```env
GEMINI_API_KEY=sua_chave_do_gemini
GROQ_API_KEY=sua_chave_da_groq
```

Essas variáveis serão carregadas pelo Python com `python-dotenv`.

## Como executar

1. Crie e ative o ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale as dependências dentro da `.venv`:

```powershell
python -m pip install -r requirements.txt
```

3. Crie o arquivo `.env` com base no `.env.example`:

```powershell
copy .env.example .env
```

4. Preencha as chaves no arquivo `.env`:

```env
GEMINI_API_KEY=sua_chave_do_gemini
GROQ_API_KEY=sua_chave_da_groq
```

5. Execute a aplicação com a `.venv` ativa:

```powershell
streamlit run app.py
```

6. Na interface, carregue o PDF e clique em **Processar e indexar PDF** antes da primeira pergunta.

## Verificação do chunking

Para testar apenas a extração e divisão do PDF, sem usar Gemini, Groq ou ChromaDB:

```bash
python rag.py
```

Também é possível informar outro PDF:

```bash
python rag.py caminho/do/documento.pdf
```

Esse comando mostra todos os chunks no terminal e também gera o arquivo `debug_chunks.csv`.

O CSV contém:

```text
chunk, fonte, artigo, titulo, capitulo, pagina_inicio, pagina_fim, texto
```

Esse arquivo serve para conferir se o chunking por artigos foi aplicado corretamente. Ele é um arquivo de debug e não precisa ser enviado ao Git.

## Prompt de geração recomendado

```text
Responda a pergunta usando apenas as informações presentes no contexto.
Se a informação não estiver no contexto, diga que não encontrou essa informação no documento.
Não invente regras, prazos ou exceções.
Cite as fontes utilizadas, incluindo artigo, capítulo e página quando disponíveis.

CONTEXTO:
{contexto_recuperado}

PERGUNTA:
{pergunta}
```

## Exemplos de perguntas para teste

- Quais são as formas de ingresso nos cursos de graduação da UNEMAT?
- O que acontece se o estudante perder o prazo de renovação de matrícula?
- Qual é a carga horária correspondente a um crédito?
- Quando uma turma pode ser cancelada?
- Quantos dias deve ter o ano letivo?
- Como funciona o trancamento de matrícula?
- O que é considerado reprovação por frequência?

## Critérios de avaliação atendidos

### Documento processado e indexado

O PDF será carregado, limpo, dividido em chunks e armazenado no ChromaDB.

### Chunking adequado implementado

O chunking será estrutural, por seções, usando artigos como unidade principal. Essa abordagem é adequada para documentos normativos.

### Busca semântica funcionando

A pergunta do usuário será convertida em embedding e comparada com os chunks armazenados no ChromaDB.

### Respostas fundamentadas no contexto

O modelo gerador receberá somente os chunks recuperados como contexto e será instruído a responder apenas com base neles.

### Citação de fontes

Cada resposta deverá citar os artigos, capítulos e páginas usados como fonte.

### Interface frontend

A aplicação terá uma interface web simples feita com Streamlit, permitindo ao usuário indexar o documento, enviar perguntas e visualizar respostas com fontes.

## Referências

- Aula 07 - Introdução ao RAG.
- ChromaDB: https://www.trychroma.com/
- Gemini API - Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Gemini API - Pricing: https://ai.google.dev/gemini-api/docs/pricing
- GroqCloud: https://console.groq.com/
