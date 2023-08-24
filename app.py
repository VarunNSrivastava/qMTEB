# my_table_app_gradio.py

import gradio as gr
import pandas as pd

def display_table():
    # Create a sample dataframe
    data = {
        "Model": ["ModelA", "ModelB", "ModelC"],
        "Score": [0.92, 0.85, 0.89],
        "Quantized Score": [0.91, 0.84, 0.88]
    }
    df = pd.DataFrame(data)

	df.index.name = "Rank"
    # Convert the DataFrame to an HTML table string
    html_table = df.to_html(classes="wide_table")


	html_content = f"""
    <style>
        .wide_table {
            width: 100%;
        }
    </style>
    {html_table}
    """

    return html_table

# Create Gradio interface
iface = gr.Interface(fn=display_table, live=True, inputs=[], outputs="html")

iface.launch()

