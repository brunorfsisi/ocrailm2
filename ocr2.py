import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import io
import json
from dotenv import load_dotenv
import os
load_dotenv()

def main():
    st.set_page_config(layout="wide") 
    st.sidebar.image("/media/brunorg/Acer/AzureOCR/LM4.png", use_column_width=True, caption="")
    # Define o layout da página para ocupar a tela inteira
    st.title("Extração de Dados com Inteligência Artificial")

    

    endpoint = os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT')
    key = os.getenv('AZURE_FORM_RECOGNIZER_KEY')

    # Inicialize o cliente do Azure Form Recognizer
    form_recognizer_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(str(key)))

    col1, col2 = st.columns(2)  # Divide a tela em duas colunas

    with col1:  # Coluna para upload e visualização da imagem
        uploaded_file = st.file_uploader("Carregue o arquivo", type=["pdf", "jpg", "png"], key="file_uploader")

        if uploaded_file is not None:
            file_stream = io.BytesIO(uploaded_file.getvalue())
            with st.spinner('Analisando documento...'):
                result = analyze_document_from_stream(form_recognizer_client, file_stream)

            file_stream.seek(0)  # Volta ao início do arquivo para reutilização
            image = Image.open(file_stream)
            
            if hasattr(result, 'tables') and result.tables:
                image_with_tables = draw_tables_on_image(image, result.tables)
                st.image(image_with_tables, caption='Imagem com Marcações das Tabelas')

    with col2:  # Coluna para exibição das tabelas e botões de download
        if uploaded_file is not None and hasattr(result, 'tables') and result.tables:
            for i, table in enumerate(result.tables):
                st.subheader(f'Tabela {i+1}')
                df = table_to_dataframe(table)
                st.table(df)

                # Converte o DataFrame em um arquivo Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=True)
                output.seek(0)
                st.download_button(label=f"Baixar Tabela {i+1} como Excel", data=output, file_name=f"tabela_{i+1}.xlsx", mime="application/vnd.ms-excel")

        elif uploaded_file is None:
            st.write("Aguardando upload do arquivo...")

def analyze_document_from_stream(form_recognizer_client, file_stream):
    """Analisa o documento fornecido como um stream de arquivo e retorna os resultados."""
    poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", file_stream)
    return poller.result()

def table_to_dataframe(table):
    """Converte uma tabela extraída pelo Form Recognizer em um DataFrame do Pandas."""
    rows = []
    for cell in table.cells:
        rows.append((cell.row_index, cell.column_index, cell.content))
    df = pd.DataFrame(rows, columns=["Row", "Column", "Content"]).pivot(index="Row", columns="Column", values="Content")
    return df

def draw_tables_on_image(image, tables):
    """Desenha retângulos ao redor das células das tabelas na imagem."""
    draw = ImageDraw.Draw(image)
    for table in tables:
        for cell in table.cells:
            # Para o SDK v3.x, 'bounding_regions' é usado em vez de 'bounding_box'
            # Checando se 'bounding_regions' existe e possui pelo menos uma região
            if hasattr(cell, 'bounding_regions') and cell.bounding_regions:
                # Acessando a primeira bounding_region (assumindo que a célula não se estende por várias páginas)
                bounding_box = cell.bounding_regions[0].polygon
                
                # 'polygon' é uma lista de pontos, onde cada ponto é um dicionário com as chaves 'x' e 'y'
                # As coordenadas do retângulo podem ser obtidas diretamente dos pontos do polígono
                points = [(point.x, point.y) for point in bounding_box]
                
                # Desenha o polígono ao redor da célula. Como é uma tabela, assumimos que os pontos formam um retângulo
                draw.polygon(points, outline="red", width=3)
    return image

if __name__ == "__main__":
    main()
