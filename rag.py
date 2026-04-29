import csv
import os
import re
import sys
from pathlib import Path

import fitz
from dotenv import load_dotenv
from google import genai
from google.genai import types


PDF_PADRAO = "Res 001-2024-CONEPE - Normatização Acadêmica.pdf"
DB_PADRAO = "chroma_db"
COLECAO_PADRAO = "resolucao_conepe_001_2024"
MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_GERACAO = "llama-3.3-70b-versatile"
DIMENSAO_EMBEDDING = 768
CSV_DEBUG_PADRAO = "debug_chunks.csv"
CANDIDATOS_RECUPERACAO = 8
MAX_FONTES_CONTEXTO = 5
FATOR_PROXIMIDADE = 1.25
MARGEM_PROXIMIDADE = 0.03


ARTIGO_RE = re.compile(r"^Art\.\s*(\d+)(?:º|o)?\b")
TITULO_RE = re.compile(r"^T[ÍI]TULO\s+[IVXLCDM]+", re.IGNORECASE)
CAPITULO_RE = re.compile(r"^Cap[íi]tulo\s+[IVXLCDM]+", re.IGNORECASE)
RODAPE_RE = re.compile(r"^Resolução nº 001/2024.*Página\s+\d+\s+de\s+\d+", re.IGNORECASE)


class ChunkDocumento:
    """Representa um chunk do documento.

    Neste projeto, cada chunk é normalmente um artigo completo da resolução.
    """

    def __init__(self, texto, fonte, artigo, titulo, capitulo, pagina_inicio, pagina_fim):
        self.texto = texto
        self.fonte = fonte
        self.artigo = artigo
        self.titulo = titulo
        self.capitulo = capitulo
        self.pagina_inicio = pagina_inicio
        self.pagina_fim = pagina_fim

    @property
    def texto_para_embedding(self):
        partes = [
            f"Fonte: {self.fonte}",
            self.titulo,
            self.capitulo,
            f"Páginas: {self.pagina_inicio}-{self.pagina_fim}",
            self.texto,
        ]
        return "\n".join(parte for parte in partes if parte)

    @property
    def metadados(self):
        return {
            "fonte": self.fonte,
            "artigo": self.artigo,
            "titulo": self.titulo,
            "capitulo": self.capitulo,
            "pagina_inicio": self.pagina_inicio,
            "pagina_fim": self.pagina_fim,
        }


def limpar_linha(linha):
    """Remove espaços extras, cabeçalhos e rodapés repetidos do PDF."""

    linha = re.sub(r"\s+", " ", linha).strip()

    if not linha:
        return ""

    linhas_ignoradas = {
        "ESTADO DE MATO GROSSO",
        "SECRETARIA DE ESTADO DE CIÊNCIA E TECNOLOGIA",
        "UNIVERSIDADE DO ESTADO DE MATO GROSSO",
        "CARLOS ALBERTO REYES MALDONADO",
        "CONSELHO DE ENSINO, PESQUISA E EXTENSÃO - CONEPE",
    }

    if linha in linhas_ignoradas:
        return ""

    if RODAPE_RE.match(linha):
        return ""

    return linha


def extrair_linhas_pdf(caminho_pdf):
    """Extrai o PDF linha por linha mantendo o número da página."""

    caminho_pdf = Path(caminho_pdf)

    if not caminho_pdf.exists():
        raise FileNotFoundError(f"PDF não encontrado: {caminho_pdf}")

    linhas = []

    with fitz.open(caminho_pdf) as documento:
        for numero_pagina, pagina in enumerate(documento, start=1):
            texto = pagina.get_text("text")
            for linha_bruta in texto.splitlines():
                linha = limpar_linha(linha_bruta)
                if linha:
                    linhas.append({"texto": linha, "pagina": numero_pagina})

    return linhas


def juntar_linhas(linhas):
    """Junta linhas extraídas do PDF em um parágrafo contínuo."""

    texto = " ".join(linha.strip() for linha in linhas if linha.strip())
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = re.sub(r"\s+([,.;:])", r"\1", texto)
    return texto


def chunkar_por_artigos(caminho_pdf=PDF_PADRAO):
    """Divide a resolução em chunks usando artigos como unidade principal."""

    caminho_pdf = Path(caminho_pdf)
    linhas = extrair_linhas_pdf(caminho_pdf)
    fonte = caminho_pdf.name

    chunks = []
    titulo_atual = "Introdução"
    capitulo_atual = ""
    marcador_titulo_pendente = ""
    marcador_capitulo_pendente = ""

    preambulo = []
    artigo_atual = ""
    linhas_artigo = []
    pagina_inicio = 1
    pagina_fim = 1

    def salvar_artigo():
        nonlocal artigo_atual, linhas_artigo, pagina_inicio, pagina_fim

        if not artigo_atual or not linhas_artigo:
            return

        chunks.append(
            ChunkDocumento(
                texto=juntar_linhas(linhas_artigo),
                fonte=fonte,
                artigo=artigo_atual,
                titulo=titulo_atual,
                capitulo=capitulo_atual,
                pagina_inicio=pagina_inicio,
                pagina_fim=pagina_fim,
            )
        )

        artigo_atual = ""
        linhas_artigo = []

    def salvar_preambulo(pagina):
        nonlocal preambulo

        texto = juntar_linhas(preambulo)
        if not texto:
            return

        chunks.append(
            ChunkDocumento(
                texto=texto,
                fonte=fonte,
                artigo="Preâmbulo",
                titulo="Preâmbulo",
                capitulo="",
                pagina_inicio=1,
                pagina_fim=pagina,
            )
        )
        preambulo = []

    for item in linhas:
        linha = item["texto"]
        pagina = item["pagina"]

        if marcador_titulo_pendente:
            titulo_atual = f"{marcador_titulo_pendente} - {linha}"
            capitulo_atual = ""
            marcador_titulo_pendente = ""
            continue

        if marcador_capitulo_pendente:
            capitulo_atual = f"{marcador_capitulo_pendente} - {linha}"
            marcador_capitulo_pendente = ""
            continue

        if TITULO_RE.match(linha):
            salvar_artigo()
            marcador_titulo_pendente = linha
            continue

        if CAPITULO_RE.match(linha):
            salvar_artigo()
            marcador_capitulo_pendente = linha
            continue

        artigo_match = ARTIGO_RE.match(linha)
        if artigo_match:
            if artigo_atual:
                salvar_artigo()
            else:
                salvar_preambulo(pagina)

            artigo_atual = f"Art. {artigo_match.group(1)}"
            linhas_artigo = [linha]
            pagina_inicio = pagina
            pagina_fim = pagina
            continue

        if artigo_atual:
            linhas_artigo.append(linha)
            pagina_fim = pagina
        else:
            preambulo.append(linha)

    salvar_artigo()

    return chunks


class GeminiEmbeddingClient:
    """Cliente responsável por transformar textos em embeddings Gemini."""

    def __init__(self, modelo=MODELO_EMBEDDING, dimensao=DIMENSAO_EMBEDDING):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Defina GEMINI_API_KEY no arquivo .env.")

        self.modelo = modelo
        self.dimensao = dimensao
        self.client = genai.Client(api_key=api_key)

    def embed_documentos(self, textos, tamanho_lote=20):
        embeddings = []

        for inicio in range(0, len(textos), tamanho_lote):
            lote = textos[inicio : inicio + tamanho_lote]
            resposta = self.client.models.embed_content(
                model=self.modelo,
                contents=lote,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.dimensao,
                ),
            )
            embeddings.extend([list(item.values) for item in resposta.embeddings])

        return embeddings

    def embed_pergunta(self, pergunta):
        resposta = self.client.models.embed_content(
            model=self.modelo,
            contents=pergunta,
            config=types.EmbedContentConfig(
                task_type="QUESTION_ANSWERING",
                output_dimensionality=self.dimensao,
            ),
        )
        return list(resposta.embeddings[0].values)


class GeradorGroq:
    """Cliente responsável por gerar a resposta final com GroqCloud."""

    def __init__(self, modelo=MODELO_GERACAO):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Defina GROQ_API_KEY no arquivo .env.")

        self.modelo = modelo
        self.client = Groq(api_key=api_key)

    def gerar_resposta(self, pergunta, contextos):
        contexto_formatado = "\n\n".join(
            [
                (
                    f"[Fonte {indice}]\n"
                    f"Artigo: {ctx['metadados'].get('artigo', '')}\n"
                    f"Título: {ctx['metadados'].get('titulo', '')}\n"
                    f"Capítulo: {ctx['metadados'].get('capitulo', '')}\n"
                    f"Páginas: {ctx['metadados'].get('pagina_inicio', '')}-"
                    f"{ctx['metadados'].get('pagina_fim', '')}\n"
                    f"Texto: {ctx['texto']}"
                )
                for indice, ctx in enumerate(contextos, start=1)
            ]
        )

        prompt = f"""
Responda à pergunta usando apenas as informações presentes no contexto.
Se a informação não estiver no contexto, diga que não encontrou essa informação no documento.
Não invente regras, prazos ou exceções.
Não converta percentuais, prazos, cargas horárias ou quantidades em outras unidades se essa equivalência não estiver explicitamente no contexto.
Diferencie regras gerais de exceções, procedimentos especiais e casos específicos.
Quando o contexto trouxer regra geral e procedimento especial ou exceção, responda primeiro com a regra geral aplicável à pergunta e depois explique o caso especial como observação.
Quando uma informação depender de cálculo, curso, disciplina, calendário ou dado não informado, diga que o documento não traz um número exato.
Ao citar uma fonte, use obrigatoriamente o identificador no formato [Fonte N].
No final, liste as fontes utilizadas mantendo os identificadores [Fonte N] e incluindo artigo, capítulo e página quando disponíveis.

CONTEXTO:
{contexto_formatado}

PERGUNTA:
{pergunta}
""".strip()

        resposta = self.client.chat.completions.create(
            model=self.modelo,
            messages=[
                {
                    "role": "system",
                    "content": "Você responde perguntas sobre normas acadêmicas usando apenas o contexto fornecido.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=900,
        )

        return resposta.choices[0].message.content or ""


class RAGResolucaoConepe:
    """Orquestra o pipeline RAG: indexação, recuperação e geração."""

    def __init__(self, caminho_db=DB_PADRAO, nome_colecao=COLECAO_PADRAO):
        import chromadb

        load_dotenv()

        self.caminho_db = str(caminho_db)
        self.nome_colecao = nome_colecao
        self.chroma = chromadb.PersistentClient(path=self.caminho_db)
        self.colecao = self.chroma.get_or_create_collection(name=self.nome_colecao)
        self.embeddings = None
        self.gerador = None

    def obter_embeddings(self):
        if self.embeddings is None:
            self.embeddings = GeminiEmbeddingClient()
        return self.embeddings

    def obter_gerador(self):
        if self.gerador is None:
            self.gerador = GeradorGroq()
        return self.gerador

    def recriar_colecao(self):
        try:
            self.chroma.delete_collection(self.nome_colecao)
        except Exception:
            pass

        self.colecao = self.chroma.get_or_create_collection(name=self.nome_colecao)

    def limpar_indice(self):
        """Apaga os chunks e embeddings persistidos no ChromaDB."""

        self.recriar_colecao()

    def total_indexado(self):
        return self.colecao.count()

    def resumo_indice(self):
        """Retorna um resumo dos documentos ativos na coleção ChromaDB."""

        total = self.total_indexado()
        if total == 0:
            return []

        dados = self.colecao.get(limit=total, include=["metadatas"])
        metadados = dados.get("metadatas", [])
        resumo_por_fonte = {}

        for meta in metadados:
            if not meta:
                continue

            fonte = meta.get("fonte") or "Fonte não informada"
            item = resumo_por_fonte.setdefault(
                fonte,
                {
                    "fonte": fonte,
                    "chunks": 0,
                    "artigos": set(),
                    "pagina_inicio": None,
                    "pagina_fim": None,
                },
            )

            item["chunks"] += 1

            artigo = meta.get("artigo")
            if artigo:
                item["artigos"].add(artigo)

            pagina_inicio = meta.get("pagina_inicio")
            pagina_fim = meta.get("pagina_fim")

            if isinstance(pagina_inicio, int):
                if item["pagina_inicio"] is None or pagina_inicio < item["pagina_inicio"]:
                    item["pagina_inicio"] = pagina_inicio

            if isinstance(pagina_fim, int):
                if item["pagina_fim"] is None or pagina_fim > item["pagina_fim"]:
                    item["pagina_fim"] = pagina_fim

        resumo = []
        for item in resumo_por_fonte.values():
            item["artigos"] = len(item["artigos"])
            resumo.append(item)

        return sorted(resumo, key=lambda item: item["chunks"], reverse=True)

    def indexar_documento(self, caminho_pdf=PDF_PADRAO, recriar=True):
        chunks = chunkar_por_artigos(caminho_pdf)

        if not chunks:
            raise RuntimeError("Nenhum chunk foi criado a partir do documento.")

        if recriar:
            self.recriar_colecao()

        textos = [chunk.texto_para_embedding for chunk in chunks]
        vetores = self.obter_embeddings().embed_documentos(textos)
        ids = [f"chunk_{indice:03d}" for indice, _ in enumerate(chunks, start=1)]

        self.colecao.add(
            ids=ids,
            documents=[chunk.texto for chunk in chunks],
            embeddings=vetores,
            metadatas=[chunk.metadados for chunk in chunks],
        )

        return len(chunks)

    def recuperar(self, pergunta, k=5):
        total = self.total_indexado()
        if total == 0:
            raise RuntimeError("A coleção ainda não possui documentos indexados.")

        vetor_pergunta = self.obter_embeddings().embed_pergunta(pergunta)
        resultados = self.colecao.query(
            query_embeddings=[vetor_pergunta],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )

        documentos = resultados.get("documents", [[]])[0]
        metadados = resultados.get("metadatas", [[]])[0]
        distancias = resultados.get("distances", [[]])[0]

        return [
            {
                "texto": texto,
                "metadados": meta,
                "distancia": distancia,
            }
            for texto, meta, distancia in zip(documentos, metadados, distancias)
        ]

    def selecionar_fontes_relevantes(self, fontes):
        """Filtra candidatos usando a distância do melhor resultado como referência."""

        if len(fontes) <= 1:
            return fontes

        melhor_distancia = fontes[0].get("distancia")
        if melhor_distancia is None:
            return fontes[:MAX_FONTES_CONTEXTO]

        limite = min(
            melhor_distancia * FATOR_PROXIMIDADE,
            melhor_distancia + MARGEM_PROXIMIDADE,
        )
        selecionadas = [
            fonte
            for fonte in fontes
            if fonte.get("distancia") is not None and fonte["distancia"] <= limite
        ]

        return (selecionadas or fontes[:1])[:MAX_FONTES_CONTEXTO]

    def consultar(self, pergunta):
        candidatos = self.recuperar(pergunta, k=CANDIDATOS_RECUPERACAO)
        contextos = self.selecionar_fontes_relevantes(candidatos)
        resposta = self.obter_gerador().gerar_resposta(pergunta, contextos)

        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "fontes": contextos,
            "fontes_recuperadas": candidatos,
        }


def formatar_fonte(fonte):
    metadados = fonte["metadados"]
    artigo = metadados.get("artigo", "Artigo não informado")
    capitulo = metadados.get("capitulo", "")
    titulo = metadados.get("titulo", "")
    pagina_inicio = metadados.get("pagina_inicio", "")
    pagina_fim = metadados.get("pagina_fim", "")

    paginas = f"p. {pagina_inicio}" if pagina_inicio == pagina_fim else f"p. {pagina_inicio}-{pagina_fim}"
    partes = [artigo, capitulo, titulo, paginas]
    return " | ".join(parte for parte in partes if parte)


def gerar_csv_debug_chunks(chunks, caminho_csv=CSV_DEBUG_PADRAO):
    """Gera um CSV com todos os chunks identificados no documento."""

    with open(caminho_csv, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            [
                "chunk",
                "fonte",
                "artigo",
                "titulo",
                "capitulo",
                "pagina_inicio",
                "pagina_fim",
                "texto",
            ]
        )

        for indice, chunk in enumerate(chunks, start=1):
            escritor.writerow(
                [
                    indice,
                    chunk.fonte,
                    chunk.artigo,
                    chunk.titulo,
                    chunk.capitulo,
                    chunk.pagina_inicio,
                    chunk.pagina_fim,
                    chunk.texto,
                ]
            )


def imprimir_debug_chunks(chunks):
    """Mostra todos os chunks no terminal."""

    print(f"Chunks criados sem acessar APIs: {len(chunks)}")
    print()

    for indice, chunk in enumerate(chunks, start=1):
        if chunk.pagina_inicio == chunk.pagina_fim:
            paginas = f"p. {chunk.pagina_inicio}"
        else:
            paginas = f"p. {chunk.pagina_inicio}-{chunk.pagina_fim}"

        previa = chunk.texto[:180]
        print(f"Chunk {indice}")
        print(f"  Artigo: {chunk.artigo}")
        print(f"  Título: {chunk.titulo}")
        print(f"  Capítulo: {chunk.capitulo or 'Sem capítulo'}")
        print(f"  Página: {paginas}")
        print(f"  Prévia: {previa}...")
        print()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    caminho_pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_PADRAO
    chunks_criados = chunkar_por_artigos(caminho_pdf)
    imprimir_debug_chunks(chunks_criados)
    gerar_csv_debug_chunks(chunks_criados)
    print(f"CSV de debug gerado em: {CSV_DEBUG_PADRAO}")
