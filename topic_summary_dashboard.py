import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import base64
import os

# === 데이터 불러오기 ===
df1 = pd.read_csv("outputs_cleansing_oil/topic_summary.csv")
df1["product"] = "cleansing_oil"

df2 = pd.read_csv("outputs_toner/topic_summary.csv")
df2["product"] = "toner"

df_summary = pd.concat([df1, df2], ignore_index=True)

# === Dash 앱 생성 ===
app = dash.Dash(__name__)
app.title = "ANUA Review Dashboard"

# === 레이아웃 ===
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

    html.H3("📈 Coherence Score by Rating"),
    html.P("This line chart shows how well-defined the topics are (measured by coherence score) for each star rating."),
    dcc.Graph(id="coherence-graph"),

    html.H3("📄 Topic Summary Table"),
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
            "whiteSpace": "normal",  # 🔹 줄바꿈 허용
            "height": "auto"  # 🔹 높이 자동 확장
        },
        style_data_conditional=[
            {
                "if": {"column_id": "keywords"},
                "whiteSpace": "pre-line",  # 🔹 keywords만 줄바꿈 허용
            }
        ],
        page_size=10,
        sort_action='native'
    )
    ,

    html.H3("🖼️ Topic Distribution Image"),
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
    rating_sample = filtered["rating"].min()  # 예시용 하나만 선택
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

# === 실행 ===
if __name__ == '__main__':
    app.run(debug=True)




# import pandas as pd
# import dash
# from dash import dcc, html, dash_table
# from dash.dependencies import Input, Output
# import plotly.express as px
# import base64
# import os
# import dash_bootstrap_components as dbc
#
# # === 1. 데이터 불러오기 === #
# df1 = pd.read_csv("outputs_cleansing_oil/topic_summary.csv")
# df1["product"] = "cleansing_oil"
#
# df2 = pd.read_csv("outputs_toner/topic_summary.csv")
# df2["product"] = "toner"
#
# df_summary = pd.concat([df1, df2], ignore_index=True)
#
# # === 2. Dash 앱 생성 (Bootstrap 테마 적용) === #
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#
# # === 3. 레이아웃 구성 === #
# app.layout = dbc.Container([
#     html.H1("ANUA Amazon Review Topic Modeling Dashboard", className="my-4 text-center"),
#
#     dbc.Row([
#         dbc.Col([
#             html.Label("🧴 제품 선택:"),
#             dcc.Dropdown(
#                 id='product-selector',
#                 options=[{"label": p, "value": p} for p in sorted(df_summary["product"].unique())],
#                 value=sorted(df_summary["product"].unique())[0]
#             )
#         ], md=4),
#
#         dbc.Col([
#             html.Label("🧪 전처리 방식:"),
#             dcc.Dropdown(
#                 id='mode-selector',
#                 options=[{"label": m, "value": m} for m in sorted(df_summary["mode"].unique())],
#                 value="unigram"
#             )
#         ], md=4),
#
#         dbc.Col([
#             html.Label("⭐ 별점 선택:"),
#             dcc.Dropdown(
#                 id='rating-selector',
#                 options=[{"label": f"{r} Stars", "value": r} for r in sorted(df_summary["rating"].unique())],
#                 value=1
#             )
#         ], md=4)
#     ], className="mb-4"),
#
#     dbc.Card([
#         dbc.CardHeader("📈 Coherence Score by Rating"),
#         dbc.CardBody([
#             dcc.Graph(id="coherence-graph")
#         ])
#     ], className="mb-4"),
#
#     dbc.Card([
#         dbc.CardHeader("📊 요약 테이블 (정렬 가능)"),
#         dbc.CardBody([
#             dash_table.DataTable(
#                 id="summary-table",
#                 columns=[{"name": col, "id": col} for col in df_summary.columns],
#                 data=df_summary.to_dict("records"),
#                 style_table={"overflowX": "auto"},
#                 page_size=10,
#                 sort_action='native'
#             )
#         ])
#     ], className="mb-4"),
#
#     dbc.Card([
#         dbc.CardHeader("🖼️ 토픽 구조 이미지 (별점/전처리 기준)"),
#         dbc.CardBody([
#             html.Img(id="topic-image", style={"maxWidth": "100%", "border": "1px solid gray"})
#         ])
#     ])
# ], fluid=True)
#
# # === 4. 콜백 함수 설정 === #
# @app.callback(
#     [
#         Output("coherence-graph", "figure"),
#         Output("summary-table", "data"),
#         Output("topic-image", "src")
#     ],
#     [
#         Input("product-selector", "value"),
#         Input("mode-selector", "value"),
#         Input("rating-selector", "value")
#     ]
# )
# def update_dashboard(selected_product, selected_mode, selected_rating):
#     filtered = df_summary[df_summary["product"] == selected_product]
#
#     fig = px.line(
#         filtered[filtered["mode"] == selected_mode],
#         x="rating",
#         y="coherence",
#         markers=True,
#         title=f"{selected_product} - {selected_mode} Coherence Score by Rating"
#     )
#
#     img_path = f"outputs/{selected_product}/plot_{selected_mode}_rating{selected_rating}.png"
#     img_src = None
#     if os.path.exists(img_path):
#         with open(img_path, "rb") as f:
#             img_src = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
#
#     return fig, filtered.to_dict("records"), img_src
#
# # === 5. 앱 실행 === #
# if __name__ == '__main__':
#     app.run(debug=True)





# import pandas as pd
# import dash
# from dash import dcc, html, dash_table
# import plotly.express as px
# import base64
# import os
# from dash.dependencies import Input, Output
# import io
#
# # === 데이터 불러오기 === #
# df1 = pd.read_csv("outputs_cleansing_oil/topic_summary.csv")
# df1["product"] = "cleansing_oil"
#
# df2 = pd.read_csv("outputs_toner/topic_summary.csv")
# df2["product"] = "toner"
#
# df_summary = pd.concat([df1, df2], ignore_index=True)
#
# # === Dash 앱 초기화 === #
# app = dash.Dash(__name__)
# app.title = "ANUA Dashboard"
# app._favicon = None
#
# # === 레이아웃 구성 === #
# app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "margin": "40px"}, children=[
#     html.H1("🌿 ANUA Amazon Review Topic Modeling Dashboard", style={"textAlign": "center", "marginBottom": "40px"}),
#
#     html.Div([
#         html.Div([
#             html.Label("🧴 제품 선택"),
#             dcc.Dropdown(
#                 id='product-selector',
#                 options=[{"label": p, "value": p} for p in sorted(df_summary["product"].unique())],
#                 value=sorted(df_summary["product"].unique())[0],
#                 style={"marginBottom": "15px"}
#             ),
#
#             html.Label("🧪 전처리 방식"),
#             dcc.Dropdown(
#                 id='mode-selector',
#                 options=[{"label": m, "value": m} for m in sorted(df_summary["mode"].unique())],
#                 value="unigram",
#                 style={"marginBottom": "15px"}
#             ),
#
#             html.Label("⭐ 별점 선택"),
#             dcc.Dropdown(
#                 id='rating-selector',
#                 options=[{"label": f"{r} Stars", "value": r} for r in sorted(df_summary["rating"].unique())],
#                 value=1,
#                 style={"marginBottom": "15px"}
#             ),
#
#             html.Button("📥 요약 CSV 다운로드", id="btn-download", n_clicks=0, style={"marginTop": "10px"})
#         ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "30px"}),
#
#         html.Div([
#             html.H3("📈 Coherence Score by Rating"),
#             html.P("This chart shows how topic coherence varies by rating. A higher score indicates more meaningful topics."),
#             dcc.Graph(id="coherence-graph")
#         ], style={"width": "68%", "display": "inline-block", "verticalAlign": "top", "backgroundColor": "#f9f9f9",
#                   "padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px"})
#     ]),
#
#     html.H3("📊 요약 테이블", style={"marginTop": "40px"}),
#     html.P("Each row shows the number of topics (k), coherence score, and number of reviews per rating."),
#     dcc.Download(id="download-summary"),
#     dash_table.DataTable(
#         id="summary-table",
#         columns=[{"name": col, "id": col} for col in df_summary.columns],
#         data=df_summary.to_dict("records"),
#         style_table={"overflowX": "auto", "border": "1px solid #ccc"},
#         style_cell={"padding": "10px", "textAlign": "center"},
#         style_header={"fontWeight": "bold", "backgroundColor": "#e1e1e1"},
#         page_size=10,
#         sort_action='native'
#     ),
#
#     html.H3("🖼️ 토픽 구조 이미지", style={"marginTop": "40px"}),
#     html.P("This plot shows the Coherence Score by Topic Count (k). It helps determine the most appropriate number of topics."),
#     html.Img(id="topic-image", style={"maxWidth": "600px", "border": "1px solid gray", "padding": "5px",
#                                       "borderRadius": "6px", "marginBottom": "40px"})
# ])
#
# # === 콜백 함수: 그래프, 표, 이미지 업데이트 === #
# @app.callback(
#     [
#         Output("coherence-graph", "figure"),
#         Output("summary-table", "data"),
#         Output("topic-image", "src")
#     ],
#     [
#         Input("product-selector", "value"),
#         Input("mode-selector", "value"),
#         Input("rating-selector", "value")
#     ]
# )
# def update_dashboard(selected_product, selected_mode, selected_rating):
#     filtered = df_summary[df_summary["product"] == selected_product]
#
#     fig = px.line(
#         filtered[filtered["mode"] == selected_mode],
#         x="rating",
#         y="coherence",
#         markers=True,
#         hover_data=["optimal_k", "num_reviews"],
#         title=f"{selected_product.replace('_', ' ').title()} - {selected_mode} Coherence by Rating"
#     )
#     fig.update_layout(template="simple_white")
#
#     img_path = f"outputs_{selected_product}/plot_{selected_mode}_rating{selected_rating}.png"
#     img_src = None
#     if os.path.exists(img_path):
#         with open(img_path, "rb") as f:
#             img_src = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
#
#     return fig, filtered.to_dict("records"), img_src
#
# # === 콜백 함수: 다운로드 버튼 === #
# @app.callback(
#     Output("download-summary", "data"),
#     Input("btn-download", "n_clicks"),
#     prevent_initial_call=True
# )
# def download_csv(n):
#     buffer = io.StringIO()
#     df_summary.to_csv(buffer, index=False)
#     buffer.seek(0)
#     return dict(content=buffer.getvalue(), filename="ANUA_topic_summary.csv")
#
# # === 앱 실행 === #
# if __name__ == '__main__':
#     app.run(debug=True)











# import pandas as pd
# import dash
# from dash import dcc, html, dash_table
# import plotly.express as px
# import base64
# import os
#
# # 데이터 로드
# df1 = pd.read_csv("outputs_cleansing_oil/topic_summary.csv")
# df1["product"] = "cleansing_oil"
# df2 = pd.read_csv("outputs_toner/topic_summary.csv")
# df2["product"] = "toner"
# df_summary = pd.concat([df1, df2], ignore_index=True)
#
# # 앱 초기화
# app = dash.Dash(__name__, external_stylesheets=["/assets/style.css"])
#
# # 레이아웃 구성
# app.layout = html.Div([
#     html.H1("ANUA Amazon Review Topic Modeling Dashboard", className="title"),
#
#     html.Div([
#         html.Div([
#             html.Label("🧴 제품 선택:"),
#             dcc.Dropdown(
#                 id='product-selector',
#                 options=[{"label": p, "value": p} for p in sorted(df_summary["product"].unique())],
#                 value=sorted(df_summary["product"].unique())[0],
#                 className="dropdown"
#             ),
#         ], className="filter-box"),
#
#         html.Div([
#             html.Label("🧪 전처리 방식:"),
#             dcc.Dropdown(
#                 id='mode-selector',
#                 options=[{"label": m, "value": m} for m in sorted(df_summary["mode"].unique())],
#                 value="unigram",
#                 className="dropdown"
#             ),
#         ], className="filter-box"),
#
#         html.Div([
#             html.Label("⭐ 별점 선택:"),
#             dcc.Dropdown(
#                 id='rating-selector',
#                 options=[{"label": f"{r} Stars", "value": r} for r in sorted(df_summary["rating"].unique())],
#                 value=1,
#                 className="dropdown"
#             ),
#         ], className="filter-box")
#     ], className="filters"),
#
#     html.H3("📈 Coherence Score by Rating", className="section-title"),
#     html.P("This graph shows how well-defined the topics are at different rating levels.", className="section-desc"),
#     dcc.Graph(id="coherence-graph"),
#
#     html.H3("📊 요약 테이블", className="section-title"),
#     html.P("Table summarizing topic modeling results per rating and preprocessing method.", className="section-desc"),
#     dash_table.DataTable(
#         id="summary-table",
#         columns=[{"name": col, "id": col} for col in df_summary.columns],
#         data=df_summary.to_dict("records"),
#         style_table={"overflowX": "auto"},
#         page_size=10,
#         sort_action='native',
#         style_header={"backgroundColor": "#e3e3e3", "fontWeight": "bold"},
#         style_data={"backgroundColor": "#fdfdfd", "color": "#333"}
#     ),
#
#     html.H3("🖼️ 토픽 구조 이미지", className="section-title"),
#     html.P("Below is the optimal topic coherence visualization for the selected combination.", className="section-desc"),
#     html.Img(id="topic-image", style={"maxWidth": "600px", "border": "1px solid gray"}),
#
#     html.Br(),
#
#     html.A("📥 Download Summary CSV", href="/assets/sample.csv", download="summary.csv", className="download-link")
# ])
#
# # 콜백 함수
# @app.callback(
#     [
#         dash.dependencies.Output("coherence-graph", "figure"),
#         dash.dependencies.Output("summary-table", "data"),
#         dash.dependencies.Output("topic-image", "src")
#     ],
#     [
#         dash.dependencies.Input("product-selector", "value"),
#         dash.dependencies.Input("mode-selector", "value"),
#         dash.dependencies.Input("rating-selector", "value")
#     ]
# )
# def update_dashboard(selected_product, selected_mode, selected_rating):
#     filtered = df_summary[df_summary["product"] == selected_product]
#
#     fig = px.line(
#         filtered[filtered["mode"] == selected_mode],
#         x="rating",
#         y="coherence",
#         markers=True,
#         title=f"{selected_product} - {selected_mode} Coherence Score by Rating"
#     )
#
#     img_path = f"outputs_{selected_product}/plot_{selected_mode}_rating{selected_rating}.png"
#     img_src = None
#     if os.path.exists(img_path):
#         with open(img_path, "rb") as f:
#             img_src = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
#
#     return fig, filtered.to_dict("records"), img_src
#
# if __name__ == '__main__':
#     app.run(debug=True)