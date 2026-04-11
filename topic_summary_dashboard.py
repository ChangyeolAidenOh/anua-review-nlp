import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import base64
import os

df1 = pd.read_csv("outputs_cleansing_oil/topic_summary.csv")
df1["product"] = "cleansing_oil"

df2 = pd.read_csv("outputs_toner/topic_summary.csv")
df2["product"] = "toner"

df_summary = pd.concat([df1, df2], ignore_index=True)

## Dash app
app = dash.Dash(__name__)
app.title = "ANUA Review Dashboard"

# Layout
app.layout = html.Div([
    html.H1("🧴 ANUA Review Topic Modeling Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Select Product:"),
            dcc.Dropdown(
                id="product-selector",
                options=[{"label": p, "value": p} for p in sorted(df_summary["product"].unique())],
                value=sorted(df_summary["product"].unique())[0]
            )
        ], style={"width": "48%", "display": "inline-block", "paddingRight": "2%"}),

        html.Div([
            html.Label("Select Preprocessing Mode:"),
            dcc.Dropdown(
                id="mode-selector",
                options=[{"label": m, "value": m} for m in sorted(df_summary["mode"].unique())],
                value="unigram"
            )
        ], style={"width": "48%", "display": "inline-block"})
    ], style={"marginBottom": 30}),

    html.H3(" Coherence Score by Rating"),
    html.P("This line chart shows how well-defined the topics are (measured by coherence score) for each star rating."),
    dcc.Graph(id="coherence-graph"),

    html.H3(" Topic Summary Table"),
    html.P("This table displays the number of reviews analyzed, the optimal number of topics discovered, the coherence score, and the keywords associated with each topic."),
    html.Div([
        html.Button("Download CSV", id="download-btn", n_clicks=0),
        dcc.Download(id="download-dataframe-csv")
    ], style={"marginBottom": 15}),
    dash_table.DataTable(
        id="summary-table",
        columns=[{"name": col, "id": col} for col in df_summary.columns],
        data=df_summary.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto"
        },
        style_data_conditional=[
            {
                "if": {"column_id": "keywords"},
                "whiteSpace": "pre-line",
            }
        ],
        page_size=10,
        sort_action='native'
    )
    ,

    html.H3(" Topic Distribution Image"),
    html.P("Below is the topic coherence visualization image for the selected configuration. It helps evaluate how many distinct topics exist."),
    html.Img(id="topic-image", style={"maxWidth": "100%", "border": "1px solid #ccc", "padding": "10px", "marginTop": "10px"})
])

# === 콜백 ===
@app.callback(
    [Output("coherence-graph", "figure"),
     Output("summary-table", "data"),
     Output("topic-image", "src")],
    [Input("product-selector", "value"),
     Input("mode-selector", "value")]
)
def update_dashboard(product, mode):
    filtered = df_summary[(df_summary["product"] == product) & (df_summary["mode"] == mode)]
    fig = px.line(
        filtered,
        x="rating",
        y="coherence",
        markers=True,
        title=f"{product} - {mode} Coherence by Rating"
    )

    # 이미지 경로 구성
    rating_sample = filtered["rating"].min()
    image_path = f"outputs_{product}/plot_{mode}_rating{rating_sample}.png"
    encoded_image = None
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded_image = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return fig, filtered.to_dict("records"), encoded_image


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n_clicks):
    return dcc.send_data_frame(df_summary.to_csv, "anua_topic_summary.csv")

if __name__ == '__main__':
    app.run(debug=True)
