import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO  # Adicionando a importação necessária
import time
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Função para a página de Extração de Dados com Inteligência Artificial
def data_extraction_page():
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
            file_stream = BytesIO(uploaded_file.getvalue())  # Alteração aqui
            with st.spinner('Analisando documento...'):
                result = analyze_document(form_recognizer_client, file_stream)

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
                output = BytesIO()  # Alteração aqui
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    concatenated_df.to_excel(writer, index=False)
                output.seek(0)
                # Coloca o botão de download fora do loop, garantindo sua unicidade
                st.download_button(label="Baixar Todas as Tabelas como Excel", data=output, file_name="todas_as_tabelas.xlsx", mime="application/vnd.ms-excel")

        elif uploaded_file is None:
            st.write("Aguardando upload do arquivo...")

    # Display Custom Extraction Model results
    

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

# Função para a página de Extração de Texto com Azure Computer Vision
def azure_cv_page():
    st.title("Teste de Acurácia do modelo")

    # Substitua pelos seus próprios valores
    endpoint = "https://imagem.cognitiveservices.azure.com/"
    key = "79a09c9aa1d3425f8dd94f4b08bb64a7"

    # Cria o cliente de Computer Vision
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    uploaded_file = st.file_uploader("Carregue uma imagem", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Converte o arquivo carregado em bytes e exibe a imagem
        image_bytes = BytesIO(uploaded_file.getvalue())
        st.image(uploaded_file, caption="Imagem Carregada", use_column_width=True)
        
        # Inicia a operação de reconhecimento de texto (OCR) na imagem
        with st.spinner('Extraindo texto...'):
            read_operation = computervision_client.read_in_stream(image=image_bytes, raw=True)
            operation_location = read_operation.headers["Operation-Location"]
            
            # Extrai o ID da operação da URL
            operation_id = operation_location.split("/")[-1]
            
            # Aguarda a operação de leitura concluir
            while True:
                read_results = computervision_client.get_read_result(operation_id)
                if read_results.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)

        # Exibe os resultados e desenha os bounding boxes
        if read_results.status == 'succeeded':
            image_cv = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

            for result in read_results.analyze_result.read_results:
                for line in result.lines:
                    bbox = [int(coord) for coord in line.bounding_box]
                    cv2.rectangle(image_cv, (bbox[0], bbox[1]), (bbox[4], bbox[5]), (255, 0, 0), 2)

            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            st.image(image_pil, caption="Imagem com Bounding Boxes", use_column_width=True)

            # Exibe o texto reconhecido
            st.write("Palavras reconhecidas:")
            lines_data = []
            for result in read_results.analyze_result.read_results:
                for line in result.lines:
                    line_data = {
                        "Texto": line.text,
                        "Confiança": line.appearance.style.confidence
                    }
                    lines_data.append(line_data)

            if lines_data:
                df = pd.DataFrame(lines_data)
                st.table(df)
            else:
                st.write("Nenhum texto foi detectado na imagem.")


# Função principal
def main():
    st.set_page_config(layout="wide") 
    st.sidebar.image("LM4.png", use_column_width=True, caption="")
    # Definindo as opções da barra lateral (sidebar)
    page_options = ["Extração de Dados", "Acurácia do modelo"]
    selected_page = st.sidebar.selectbox("Escolha uma página", page_options)

    # Determinando qual página mostrar com base na escolha do usuário
    if selected_page == "Extração de Dados":
        data_extraction_page()
    elif selected_page == "Acurácia do modelo":
        azure_cv_page()


if __name__ == "__main__":
    main()
