import gradio as gr
import pandas as pd
import json
import os

# Given list of tasks for clustering and pair classification

def display_table():
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)
    df = pd.DataFrame(data)

    df = df.reset_index()
    df.columns = ['Rank', 'Model', 'Score', 'Quantized Score']

    html_table = df.to_html()

    html_content = f"""
    <style>
        .wide_table {{
            width: 100%;
        }}
    </style>
    {html_table}
    """

    return html_content


iface = gr.Interface(fn=display_table, live=True, inputs=[], outputs="html")

iface.launch()
