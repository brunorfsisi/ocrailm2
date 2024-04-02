import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import io
import json

def main():
    st.set_page_config(layout="wide") 
    st.sidebar.image("LM4.png", use_column_width=True, caption="")
    # Define o layout da página para ocupar a tela inteira
    st.title("Extração de Dados com Inteligência Artificial")

    # Carregar configurações do arquivo JSON
    with open('config.json', 'r') as f:
        config = json.load(f)

    endpoint = config['AZURE_ENDPOINT']
    key = config['AZURE_KEY']

    # Inicialize o cliente do Azure Form Recognizer
    form_recognizer_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(str(key)))

    col1, col2 = st.columns(2)  # Divide a tela em duas colunas
    result = None  # Inicialize a variável result fora dos blocos with

    all_tables = []  # Armazenará todas as tabelas

    with col1:  # Coluna para upload e visualização da imagem
        uploaded_file = st.file_uploader("Carregue o arquivo", type=["pdf", "jpg", "png"], key="file_uploader")
    
        if uploaded_file is not None:
            file_stream = io.BytesIO(uploaded_file.getvalue())
            with st.spinner('Analisando documento...'):
                # Updated the function call here to match the defined function
                result = analyze_document_from_stream(form_recognizer_client, file_stream)


            file_stream.seek(0)  # Volta ao início do arquivo para reutilização
            image = Image.open(file_stream)
            
            if hasattr(result, 'tables') and result.tables:
                image_with_tables = draw_tables_on_image(image, result.tables)
                st.image(image_with_tables, caption='Imagem com Marcações das Tabelas')

            # Exibir informações de confiança
            if hasattr(result, 'confidence'):
                st.subheader('Confiança:')
                st.write(result.confidence)

            # Exibir informações do campo 'Fields' se estiverem disponíveis
            if hasattr(result, 'fields'):
                st.subheader('Campos (Fields):')
                fields = result.fields
                for field in fields:
                    st.write(f"**{field.name}:** {field.value}")

            # Adiciona todas as tabelas à lista de tabelas
            if hasattr(result, 'tables'):
                all_tables.extend(result.tables)

    # Verificar se result possui métricas fora do bloco with col1
    if result is not None:
        if hasattr(result, 'metrics'):
            st.subheader('Métricas do Modelo:')
            metrics = result.metrics
            for metric_name, metric_value in metrics.items():
                st.write(f"**{metric_name}:** {metric_value}")

    with col2:  # Coluna para exibição das tabelas e botões de download
        if uploaded_file is not None and all_tables:
            concatenated_df = pd.DataFrame()  # Dataframe vazio para concatenar todas as tabelas

            for i, table in enumerate(all_tables):
                st.subheader(f'Tabela {i+1}')
                df = table_to_dataframe(table)
                st.table(df)

                # Concatena a tabela atual ao DataFrame concatenado
                concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

            if not concatenated_df.empty:
                # Converte o DataFrame concatenado em um arquivo Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    concatenated_df.to_excel(writer, index=False)
                output.seek(0)
                # Coloca o botão de download fora do loop, garantindo sua unicidade
                st.download_button(label="Baixar Todas as Tabelas como Excel", data=output, file_name="todas_as_tabelas.xlsx", mime="application/vnd.ms-excel")

        elif uploaded_file is None:
            st.write("Aguardando upload do arquivo...")

def analyze_document(form_recognizer_client, file_stream):
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
