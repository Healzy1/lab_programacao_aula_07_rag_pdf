# Consulta RAG de Documentos PDF

Aplicação acadêmica em Python/Streamlit que implementa um pipeline RAG para consultar documentos PDF com respostas fundamentadas em trechos recuperados semanticamente.

O projeto foi desenvolvido para a atividade prática da Aula 07 - Introdução ao RAG. O documento usado nos testes da atividade foi a `Res 001-2024-CONEPE - Normatização Acadêmica.pdf`, mas a interface aceita o upload de outros PDFs.

## Objetivo

Implementar um pipeline RAG básico capaz de:

1. Receber um PDF pela interface.
2. Extrair e limpar o texto do documento.
3. Dividir o conteúdo em chunks.
4. Gerar embeddings dos chunks.
5. Armazenar texto, embeddings e metadados no ChromaDB.
6. Receber uma pergunta do usuário.
7. Recuperar chunks relevantes por busca semântica.
8. Selecionar automaticamente os melhores chunks para contexto.
9. Gerar uma resposta usando apenas o contexto recuperado.
10. Exibir as fontes utilizadas com artigo, capítulo, título e página.

## Tecnologias

- **Python**: linguagem principal do projeto.
- **Streamlit**: interface web.
- **PyMuPDF**: extração de texto do PDF.
- **ChromaDB**: banco vetorial persistido em disco.
- **Gemini Embedding** (`gemini-embedding-001`): geração dos embeddings.
- **GroqCloud** (`llama-3.3-70b-versatile`): geração das respostas.
- **python-dotenv**: carregamento das chaves de API a partir do `.env`.

## Estrutura do projeto

```text
.
|-- app.py
|-- rag.py
|-- requirements.txt
|-- .env.example
|-- README.md
|-- chroma_db/
|-- uploaded_docs/
|-- Aula-07-Introducao-RAG.pdf
`-- Res 001-2024-CONEPE - Normatização Acadêmica.pdf
```

Descrição dos principais arquivos e pastas:

- `app.py`: interface Streamlit.
- `rag.py`: extração, chunking, embeddings, recuperação e geração da resposta.
- `requirements.txt`: dependências do projeto.
- `.env.example`: exemplo das variáveis de ambiente necessárias.
- `chroma_db/`: arquivos internos do ChromaDB.
- `uploaded_docs/`: PDFs enviados e salvos no servidor da aplicação.
- `debug_chunks.csv`: arquivo opcional gerado para inspecionar os chunks.

## Como Funciona

O fluxo principal da aplicação é:

1. O usuário carrega um PDF na seção **Documento**.
2. O PDF é salvo em `uploaded_docs/`.
3. O usuário clica em **Processar documento**.
4. O app extrai o texto do PDF com PyMuPDF.
5. O texto é limpo para remover cabeçalhos, rodapés e linhas repetidas.
6. O conteúdo é dividido em chunks.
7. Os embeddings são gerados com Gemini.
8. Os chunks, embeddings e metadados são salvos no ChromaDB.
9. O usuário envia uma pergunta.
10. A pergunta é convertida em embedding.
11. O ChromaDB recupera chunks candidatos por similaridade vetorial.
12. O app filtra automaticamente os chunks mais relevantes.
13. O modelo gerador recebe a pergunta e os chunks selecionados.
14. A resposta é exibida junto com as fontes.

## Interface

A interface é dividida em:

- **Documento**: upload de PDF e botão **Processar documento**.
- **Documento processado**: mostra o documento ativo e a quantidade de trechos disponíveis.
- **Arquivos enviados**: lista os PDFs salvos no servidor da aplicação.
- **Pergunta**: campo principal para consultar o documento processado.
- **Resposta**: resposta gerada pelo modelo.
- **Trechos usados para responder**: chunks selecionados automaticamente para fundamentar a resposta.
- **Dados técnicos**: detalhes do ChromaDB, modelos usados, parâmetros da recuperação e distâncias vetoriais.

### Arquivos enviados vs documento processado

O projeto separa duas coisas:

- **Arquivos enviados**: PDFs salvos em `uploaded_docs/`, no servidor da aplicação.
- **Documento processado**: chunks e embeddings salvos no ChromaDB.

Isso permite limpar o processamento sem apagar o PDF enviado. Depois de limpar o processamento, é possível reprocessar um PDF já salvo usando **Processar arquivo salvo**.

Botões principais:

- **Processar documento**: processa o PDF recém-carregado.
- **Limpar processamento**: remove os chunks/embeddings ativos do ChromaDB.
- **Processar arquivo salvo**: reprocessa um PDF já existente em `uploaded_docs/`.
- **Remover arquivos enviados**: apaga os PDFs salvos em `uploaded_docs/`.

## Estratégia de Chunking

Para o documento da atividade, a estratégia usada é chunking estrutural por artigos.

A resolução possui uma estrutura normativa com:

- títulos;
- capítulos;
- artigos;
- parágrafos;
- incisos.

Por isso, a unidade principal escolhida foi:

```text
1 chunk = 1 artigo completo
```

Essa escolha evita chunks grandes demais, como capítulos inteiros, e também evita fragmentar excessivamente o texto em sentenças isoladas.

Cada chunk recebe metadados:

```json
{
  "fonte": "Res 001-2024-CONEPE - Normatização Acadêmica.pdf",
  "artigo": "Art. 35",
  "titulo": "TÍTULO V - DA VIDA ACADÊMICA",
  "capitulo": "Capítulo III - Da Renovação da Matrícula",
  "pagina_inicio": 8,
  "pagina_fim": 8
}
```

Esses metadados são usados na exibição das fontes e também no contexto enviado para geração da resposta.

## Seleção Automática de Contexto

Após a busca vetorial, o app não envia todos os chunks recuperados ao modelo gerador. Ele faz uma seleção automática.

Parâmetros atuais:

```text
Busca vetorial inicial: até 8 chunks candidatos
Filtro de relevância: distância vetorial em relação ao melhor candidato
Fator de proximidade: 1.25
Margem máxima de distância: 0.03
Contexto final: até 5 chunks enviados ao modelo
```

O fluxo é:

1. Recuperar até 8 chunks candidatos no ChromaDB.
2. Identificar o chunk com menor distância vetorial.
3. Manter somente chunks próximos ao melhor candidato.
4. Limitar o contexto final a no máximo 5 chunks.

Isso reduz ruído no prompt e evita enviar trechos pouco relevantes ao modelo.

## Regras de Resposta

O modelo gerador recebe instruções para:

- responder apenas com base no contexto;
- dizer quando a informação não estiver no documento;
- não inventar regras, prazos ou exceções;
- não converter percentuais, prazos, cargas horárias ou quantidades sem equivalência explícita;
- diferenciar regra geral de exceções e procedimentos especiais;
- responder primeiro com a regra geral quando houver regra geral e caso especial;
- citar fontes no formato `[Fonte N]`;
- listar as fontes utilizadas ao final.

## Como Executar

Crie e ative o ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação por política de execução:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Crie o `.env` a partir do exemplo:

```powershell
copy .env.example .env
```

Preencha as chaves:

```env
GEMINI_API_KEY=sua_chave_do_gemini
GROQ_API_KEY=sua_chave_da_groq
```

Execute a aplicação:

```powershell
streamlit run app.py
```

Depois, acesse a URL mostrada pelo Streamlit, normalmente:

```text
http://localhost:8501
```

## Como Usar

1. Abra a aplicação Streamlit.
2. Carregue um PDF na seção **Documento**.
3. Clique em **Processar documento**.
4. Aguarde a criação dos chunks e embeddings.
5. Digite uma pergunta sobre o documento processado.
6. Clique em **Consultar**.
7. Leia a resposta e confira os trechos usados para fundamentá-la.

Se o PDF já estiver salvo em **Arquivos enviados**, é possível selecioná-lo e clicar em **Processar arquivo salvo**.

## Verificação do Chunking

Para testar somente a extração e divisão do PDF, sem usar Gemini, Groq ou ChromaDB:

```powershell
python rag.py
```

Também é possível informar outro PDF:

```powershell
python rag.py caminho/do/documento.pdf
```

Esse comando imprime os chunks no terminal e gera `debug_chunks.csv`.

O CSV contém:

```text
chunk, fonte, artigo, titulo, capitulo, pagina_inicio, pagina_fim, texto
```

## Exemplos de Perguntas

Exemplos úteis para testar com a Resolução CONEPE:

```text
Quantos dias deve ter o ano letivo?
```

```text
O que acontece se o estudante perder o prazo de renovação de matrícula e como isso se relaciona com o trancamento de matrícula?
```

```text
Quantos dias eu posso faltar?
```

```text
Qual é o valor da mensalidade do curso?
```

A última pergunta é útil para verificar se o modelo evita inventar informações que não aparecem no documento.

## Critérios de Avaliação Atendidos

### Documento processado e indexado

O PDF é carregado, extraído, dividido em chunks e armazenado no ChromaDB com embeddings.

### Chunking adequado implementado

O chunking usa artigos como unidade semântica principal, adequado para documentos normativos.

### Busca semântica funcionando

A pergunta é convertida em embedding e comparada com os chunks armazenados no ChromaDB.

### Respostas fundamentadas no contexto

O modelo recebe apenas os chunks selecionados como contexto e é instruído a responder somente com base neles.

### Citação de fontes

As respostas devem citar fontes no formato `[Fonte N]`, e a interface exibe os trechos usados para fundamentar a resposta.

### Interface frontend

A aplicação possui interface Streamlit para upload, processamento, consulta, visualização de respostas, fontes e dados técnicos.

## Limitações

- O chunking foi otimizado para documentos normativos estruturados por artigos.
- PDFs muito diferentes, sem estrutura clara, podem exigir outra estratégia de chunking.
- A seleção automática por distância vetorial reduz ruído, mas não substitui um reranker dedicado.
- O modelo gerador pode variar a redação das respostas, mesmo seguindo o prompt.
- Os arquivos enviados ficam no servidor da aplicação, não no computador do usuário, caso o app esteja hospedado em nuvem.

## Referências

- Aula 07 - Introdução ao RAG.
- ChromaDB: https://www.trychroma.com/
- Gemini API - Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- GroqCloud: https://console.groq.com/
