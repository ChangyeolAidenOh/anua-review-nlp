# Stylish Dash app will go here
import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import base64
import os

# 데이터 로드
df1 = pd.read_csv("outputs_cleansing_oil/topic_summary.csv")
df1["product"] = "cleansing_oil"
df2 = pd.read_csv("outputs_toner/topic_summary.csv")
df2["product"] = "toner"
df_summary = pd.concat([df1, df2], ignore_index=True)

# 앱 초기화
app = dash.Dash(__name__, external_stylesheets=["/assets/style.css"])

# 레이아웃 구성
app.layout = html.Div([
    html.H1("ANUA Amazon Review Topic Modeling Dashboard", className="title"),

    html.Div([
        html.Div([
            html.Label("🧴 제품 선택:"),
            dcc.Dropdown(
                id='product-selector',
                options=[{"label": p, "value": p} for p in sorted(df_summary["product"].unique())],
                value=sorted(df_summary["product"].unique())[0],
                className="dropdown"
            ),
        ], className="filter-box"),

        html.Div([
            html.Label("🧪 전처리 방식:"),
            dcc.Dropdown(
                id='mode-selector',
                options=[{"label": m, "value": m} for m in sorted(df_summary["mode"].unique())],
                value="unigram",
                className="dropdown"
            ),
        ], className="filter-box"),

        html.Div([
            html.Label("⭐ 별점 선택:"),
            dcc.Dropdown(
                id='rating-selector',
                options=[{"label": f"{r} Stars", "value": r} for r in sorted(df_summary["rating"].unique())],
                value=1,
                className="dropdown"
            ),
        ], className="filter-box")
    ], className="filters"),

    html.H3("📈 Coherence Score by Rating", className="section-title"),
    html.P("This graph shows how well-defined the topics are at different rating levels.", className="section-desc"),
    dcc.Graph(id="coherence-graph"),

    html.H3("📊 요약 테이블", className="section-title"),
    html.P("Table summarizing topic modeling results per rating and preprocessing method.", className="section-desc"),
    dash_table.DataTable(
        id="summary-table",
        columns=[{"name": col, "id": col} for col in df_summary.columns],
        data=df_summary.to_dict("records"),
        style_table={"overflowX": "auto"},
        page_size=10,
        sort_action='native',
        style_header={"backgroundColor": "#e3e3e3", "fontWeight": "bold"},
        style_data={"backgroundColor": "#fdfdfd", "color": "#333"}
    ),

    html.H3("🖼️ 토픽 구조 이미지", className="section-title"),
    html.P("Below is the optimal topic coherence visualization for the selected combination.", className="section-desc"),
    html.Img(id="topic-image", style={"maxWidth": "600px", "border": "1px solid gray"}),

    html.Br(),

    html.A("📥 Download Summary CSV", href="/assets/sample.csv", download="summary.csv", className="download-link")
])

# 콜백 함수
@app.callback(
    [
        dash.dependencies.Output("coherence-graph", "figure"),
        dash.dependencies.Output("summary-table", "data"),
        dash.dependencies.Output("topic-image", "src")
    ],
    [
        dash.dependencies.Input("product-selector", "value"),
        dash.dependencies.Input("mode-selector", "value"),
        dash.dependencies.Input("rating-selector", "value")
    ]
)
def update_dashboard(selected_product, selected_mode, selected_rating):
    filtered = df_summary[df_summary["product"] == selected_product]

    fig = px.line(
        filtered[filtered["mode"] == selected_mode],
        x="rating",
        y="coherence",
        markers=True,
        title=f"{selected_product} - {selected_mode} Coherence Score by Rating"
    )

    img_path = f"outputs_{selected_product}/plot_{selected_mode}_rating{selected_rating}.png"
    img_src = None
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_src = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    return fig, filtered.to_dict("records"), img_src

if __name__ == '__main__':
    app.run(debug=True)
