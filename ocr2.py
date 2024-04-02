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
    st.title("Extração de Dados com Inteligência Artificial")

    with open('config.json', 'r') as f:
        config = json.load(f)

    endpoint = config['AZURE_ENDPOINT']
    key = config['AZURE_KEY']

    form_recognizer_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(str(key)))

    col1, col2 = st.columns(2)
    result = None
    all_tables = []

    with col1:
        uploaded_file = st.file_uploader("Carregue o arquivo", type=["pdf", "jpg", "png"], key="file_uploader")
    
        if uploaded_file is not None:
            file_stream = io.BytesIO(uploaded_file.getvalue())
            with st.spinner('Analisando documento...'):
                result = analyze_document(form_recognizer_client, file_stream)

            file_stream.seek(0)
            image = Image.open(file_stream)
            
            if hasattr(result, 'tables') and result.tables:
                image_with_tables = draw_tables_on_image(image, result.tables)
                st.image(image_with_tables, caption='Imagem com Marcações das Tabelas')

            if hasattr(result, 'confidence'):
                st.subheader('Confiança:')
                st.write(result.confidence)

            if hasattr(result, 'fields'):
                st.subheader('Campos (Fields):')
                fields = result.fields
                for field in fields:
                    st.write(f"**{field.name}:** {field.value}")

            if hasattr(result, 'tables'):
                all_tables.extend(result.tables)

    if result is not None and hasattr(result, 'metrics'):
        st.subheader('Métricas do Modelo:')
        metrics = result.metrics
        for metric_name, metric_value in metrics.items():
            st.write(f"**{metric_name}:** {metric_value}")

    with col2:
        if uploaded_file is not None and all_tables:
            concatenated_df = pd.DataFrame()

            for i, table in enumerate(all_tables):
                st.subheader(f'Tabela {i+1}')
                df = table_to_dataframe(table)
                st.table(df)
                concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

            if not concatenated_df.empty:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    concatenated_df.to_excel(writer, index=False)
                output.seek(0)
                st.download_button(label="Baixar Todas as Tabelas como Excel", data=output, file_name="todas_as_tabelas.xlsx", mime="application/vnd.ms-excel")
        elif uploaded_file is None:
            st.write("Aguardando upload do arquivo...")

def analyze_document(form_recognizer_client, file_stream):
    poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", file_stream)
    return poller.result()

def table_to_dataframe(table):
    rows = []
    for cell in table.cells:
        rows.append((cell.row_index, cell.column_index, cell.content))
    df = pd.DataFrame(rows, columns=["Row", "Column", "Content"]).pivot(index="Row", columns="Column", values="Content")
    return df

def draw_tables_on_image(image, tables):
    draw = ImageDraw.Draw(image)
    for table in tables:
        for cell in table.cells:
            if hasattr(cell, 'bounding_regions') and cell.bounding_regions:
                bounding_box = cell.bounding_regions[0].polygon
                points = [(point.x, point.y) for point in bounding_box]
                draw.polygon(points, outline="red", width=3)
    return image

if __name__ == "__main__":
    main()

