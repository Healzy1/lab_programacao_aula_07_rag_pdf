from pathlib import Path

import streamlit as st

from rag import (
    CANDIDATOS_RECUPERACAO,
    DB_PADRAO,
    FATOR_PROXIMIDADE,
    MAX_FONTES_CONTEXTO,
    MARGEM_PROXIMIDADE,
    MODELO_EMBEDDING,
    MODELO_GERACAO,
    RAGResolucaoConepe,
    formatar_fonte,
)


PASTA_UPLOADS = Path("uploaded_docs")
PASTA_CHROMA = Path(DB_PADRAO)


# Configuração básica da página no navegador.
st.set_page_config(
    page_title="RAG de Documentos PDF",
    layout="wide",
)


@st.cache_resource
def obter_rag():
    """Cria o objeto RAG uma única vez e reutiliza entre interações."""

    return RAGResolucaoConepe()


def salvar_pdf_enviado(arquivo_enviado):
    """Salva o PDF carregado pela interface em uma pasta local."""

    PASTA_UPLOADS.mkdir(exist_ok=True)
    caminho = PASTA_UPLOADS / Path(arquivo_enviado.name).name
    caminho.write_bytes(arquivo_enviado.getbuffer())
    return caminho


def chroma_tem_arquivos_internos():
    """Indica se a pasta do ChromaDB existe fisicamente no disco."""

    return PASTA_CHROMA.exists() and any(PASTA_CHROMA.iterdir())


def remover_pdfs_enviados():
    """Remove os PDFs salvos no servidor da aplicação."""

    if not PASTA_UPLOADS.exists():
        return 0

    removidos = 0
    for caminho_pdf in PASTA_UPLOADS.glob("*.pdf"):
        caminho_pdf.unlink()
        removidos += 1

    return removidos


def listar_pdfs_enviados():
    """Lista os PDFs salvos no servidor da aplicação."""

    if not PASTA_UPLOADS.exists():
        return []

    return sorted(PASTA_UPLOADS.glob("*.pdf"), key=lambda caminho: caminho.name.lower())


def exibir_fontes(fontes):
    """Mostra os chunks selecionados como contexto da resposta."""

    st.subheader("Trechos usados para responder")
    st.caption("Selecionados automaticamente para fundamentar a resposta.")

    for indice, fonte in enumerate(fontes, start=1):
        titulo = formatar_fonte(fonte)

        with st.expander(f"Fonte {indice}: {titulo}"):
            st.write(fonte["texto"])


def exibir_detalhes_tecnicos(fontes_recuperadas, fontes_enviadas):
    """Mostra detalhes técnicos da recuperação vetorial."""

    if not fontes_recuperadas:
        return

    with st.expander("Dados técnicos dos trechos usados"):
        st.write(
            f"{len(fontes_recuperadas)} chunks candidatos foram recuperados pela busca vetorial. "
            f"{len(fontes_enviadas)} foram selecionados para o contexto final."
        )

        for indice, fonte in enumerate(fontes_recuperadas, start=1):
            titulo = formatar_fonte(fonte)
            distancia = fonte.get("distancia")

            if distancia is not None:
                titulo = f"{titulo} | distância: {distancia:.4f}"

            st.write(f"**Chunk candidato {indice}:** {titulo}")


if "documento_indexado" not in st.session_state:
    st.session_state["documento_indexado"] = False

if "pdf_id" not in st.session_state:
    st.session_state["pdf_id"] = None

if "pdf_indexado" not in st.session_state:
    st.session_state["pdf_indexado"] = None


st.title("Consulta RAG de Documentos PDF")
st.caption("Envie um PDF, processe o conteúdo e faça perguntas com respostas fundamentadas nos trechos recuperados.")

rag = obter_rag()


def processar_pdf(caminho_pdf):
    """Processa um PDF e substitui o documento atualmente consultável."""

    try:
        with st.spinner("Processando PDF e preparando consulta..."):
            # O parâmetro recriar=True substitui o índice anterior para não acumular documentos.
            total = rag.indexar_documento(caminho_pdf, recriar=True)

        st.session_state["documento_indexado"] = True
        st.session_state["pdf_indexado"] = caminho_pdf.name
        st.success(f"{total} trechos preparados. Documento anterior substituído.")
        st.rerun()
    except Exception as erro:
        st.session_state["documento_indexado"] = False
        st.error(str(erro))


# Barra lateral: carregamento do documento e controle do índice vetorial.
with st.sidebar:
    st.header("Documento")

    arquivo_enviado = st.file_uploader(
        "Carregue o documento PDF",
        type=["pdf"],
        help="Este arquivo será processado para permitir consultas com base no conteúdo.",
    )

    pdf_path = None

    if arquivo_enviado is not None:
        # Se o usuário trocar o PDF, a aplicação obriga uma nova indexação.
        arquivo_id = f"{arquivo_enviado.name}:{arquivo_enviado.size}"
        if st.session_state["pdf_id"] != arquivo_id:
            st.session_state["pdf_id"] = arquivo_id
            st.session_state["documento_indexado"] = False
            st.session_state["pdf_indexado"] = None

        pdf_path = salvar_pdf_enviado(arquivo_enviado)
        st.success(f"PDF carregado: {pdf_path.name}")
    else:
        st.info("Carregue um PDF para começar.")

    if pdf_path is not None and not st.session_state["documento_indexado"]:
        st.warning("O PDF carregado ainda não foi processado.")

    if st.button("Processar documento", use_container_width=True, disabled=pdf_path is None):
        processar_pdf(pdf_path)

    st.divider()

    st.header("Documento processado")
    total_indexado = rag.total_indexado()
    resumo_indice = rag.resumo_indice()
    st.metric("Trechos disponíveis", total_indexado)

    if st.session_state["documento_indexado"]:
        st.success(f"Processado nesta sessão: {st.session_state['pdf_indexado']}")
    elif total_indexado > 0:
        st.info("Há um documento processado salvo e pronto para consulta.")
    elif chroma_tem_arquivos_internos():
        st.info("Nenhum documento processado ativo.")
    else:
        st.info("Nenhum documento processado.")

    if resumo_indice:
        st.markdown("**Documento ativo**")

        for item in resumo_indice:
            paginas = ""
            if item["pagina_inicio"] is not None and item["pagina_fim"] is not None:
                paginas = f"p. {item['pagina_inicio']}-{item['pagina_fim']}"

            detalhes = [
                f"{item['chunks']} trechos",
                f"{item['artigos']} artigos",
                paginas,
            ]
            detalhes = " | ".join(detalhe for detalhe in detalhes if detalhe)

            st.write(f"• **{item['fonte']}**")
            st.caption(detalhes)

    if st.button("Limpar processamento", use_container_width=True):
        rag.limpar_indice()
        st.session_state["documento_indexado"] = False
        st.session_state["pdf_indexado"] = None
        st.success("Documento processado removido.")
        st.rerun()

    with st.expander("Dados técnicos"):
        st.write("Banco vetorial: ChromaDB")
        st.write(f"Pasta do banco: `{DB_PADRAO}`")
        st.write(f"Chunks indexados: {total_indexado}")
        st.write("Estratégia de chunking: por artigo")
        st.write(f"Modelo de embeddings: `{MODELO_EMBEDDING}`")
        st.write(f"Modelo de geração: `{MODELO_GERACAO}`")
        st.write("Seleção de contexto: automática")
        st.write(f"Busca vetorial inicial: até {CANDIDATOS_RECUPERACAO} chunks candidatos")
        st.write("Filtro de relevância: distância vetorial em relação ao melhor candidato")
        st.write(f"Fator de proximidade: {FATOR_PROXIMIDADE}")
        st.write(f"Margem máxima de distância: {MARGEM_PROXIMIDADE}")
        st.write(f"Contexto final: até {MAX_FONTES_CONTEXTO} chunks enviados ao modelo")

        if chroma_tem_arquivos_internos() and total_indexado == 0:
            st.caption("A pasta do banco ainda contém arquivos internos, mas não há chunks consultáveis.")

    st.divider()

    st.header("Arquivos enviados")
    pdfs_enviados = listar_pdfs_enviados()

    if pdfs_enviados:
        st.caption(f"{len(pdfs_enviados)} arquivo(s) salvo(s) no servidor da aplicação.")

        for caminho_pdf in pdfs_enviados:
            tamanho_mb = caminho_pdf.stat().st_size / (1024 * 1024)
            st.write(f"• **{caminho_pdf.name}**")
            st.caption(f"{tamanho_mb:.2f} MB")

        arquivo_salvo = st.selectbox(
            "Arquivo salvo para processamento",
            pdfs_enviados,
            format_func=lambda caminho: caminho.name,
        )

        if st.button("Processar arquivo salvo", use_container_width=True):
            processar_pdf(arquivo_salvo)
    else:
        st.info("Nenhum PDF enviado salvo no servidor da aplicação.")

    if st.button("Remover arquivos enviados", use_container_width=True):
        removidos = remover_pdfs_enviados()
        st.session_state["pdf_id"] = None

        if removidos:
            st.success(f"{removidos} PDF(s) removido(s) dos arquivos enviados.")
            st.rerun()
        else:
            st.info("Nenhum PDF enviado encontrado.")


pergunta = st.text_area(
    "Pergunta",
    value=st.session_state.get("pergunta", ""),
    height=120,
    placeholder="Digite uma pergunta sobre o documento processado...",
)

consultar = st.button("Consultar", type="primary", use_container_width=True, disabled=total_indexado == 0)

if consultar:
    if not pergunta.strip():
        st.warning("Digite uma pergunta antes de consultar.")
    elif total_indexado == 0:
        st.warning("Carregue um PDF e clique em Processar documento antes de consultar.")
    elif pdf_path is not None and not st.session_state["documento_indexado"]:
        st.warning("O PDF carregado ainda não foi processado. Indexe o PDF antes de consultar.")
    else:
        try:
            with st.spinner("Buscando fontes relevantes e gerando resposta..."):
                # Consulta completa: pergunta -> embedding -> busca semântica -> resposta.
                resultado = rag.consultar(pergunta.strip())

            st.subheader("Resposta")
            st.write(resultado["resposta"])
            exibir_fontes(resultado["fontes"])
            exibir_detalhes_tecnicos(resultado.get("fontes_recuperadas", []), resultado["fontes"])
        except Exception as erro:
            st.error(str(erro))
