import os
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

# National data

df = pd.read_excel('https://www.census.gov/hfp/btos/downloads/National.xlsx')

# Clean national data

ai_map = {'In the last two weeks, did this business use Artificial Intelligence (AI) in producing goods or services? '
          '(Examples of AI: machine learning, natural language processing, virtual agents, voice recognition, etc.)':'Used AI last 2 weeks',
          'During the next six months, do you think this business will be using Artificial Intelligence (AI) in producing goods or services? '
          '(Examples of AI: machine learning, natural language processing, virtual agents, voice recognition, etc.)': 'Intend to use AI next 6 months'}

dates_map = {'202319': '09/10/2023', '202320': '09/24/2023', '202321':'10/08/2023', '202322': '10/22/2023', '202323':'11/05/2023','202324':'11/19/2023',
             '202325':'12/03/2023','202326': '12/17/2023','202401':'12/31/2023','202402':'01/14/2024','202403':'01/28/2024','202404':'02/11/2024',
             '202405': '02/25/2024', '202406': '03/10/2024','202407': '03/24/2024', '202408': '04/07/2024', '202409': '04/21/2024',
             '202410': '05/05/2024', '202411':'05/19/2024', '202412':'06/02/2024', '202413': '06/16/2024', '202414': '06/30/2024',
             '202415': '07/14/2024', '202416': '07/28/2024', '202417': '08/11/2024', '202418': '08/25/2024', '202419':	'09/8/2024',
             '202420': '09/22/2024', '202421': '10/6/2024', '202422':	'10/20/2024','202423':'11/3/2024','202424': '11/17/2024', '202425': '12/1/2024',
             '202426': '12/15/2024', '202501': '12/29/2024', '202502': '01/12/2025', '202503': '01/26/2025', '202504': '02/9/2025', '202505': '02/23/2025',
             '202506': '03/9/2025', '202507': '03/23/2025', '202508':	'04/6/2025','202509': '04/20/2025', '202510': '05/4/2025', '202511': '05/18/2025',
             '202512': '06/1/2025', '202513': '06/15/2025', '202514': '06/29/2025'}

def tweak_national(national_df):
  return (
      df
        .dropna()
        .loc[lambda df_: df_["Question"].str.contains('AI | Artificial Intelligence')]
        .drop(['Question ID', 'Answer ID'], axis='columns')
        .melt(id_vars=['Question', 'Answer'], value_name='percentage', var_name='end_date')
        .assign(end_date = lambda df_: pd.to_datetime(df_['end_date'].astype('str').map(dates_map)),
              percentage= lambda df_: df_['percentage'].str.rstrip('%').astype('float'),
              Question = lambda df_: df_["Question"].map(ai_map)
              )
        .dropna()
)

ai_df = tweak_national(df)

# Filter the DataFrames

ai_df_yes = ai_df.loc[ai_df['Answer'] == 'Yes']
ai_df_no = ai_df.loc[ai_df['Answer'] == 'No']
ai_df_dont_know = ai_df.loc[ai_df['Answer'] == 'Do not know']

# National Yes graph

fig_yes = (
    ai_df_yes
      .groupby(['Question', pd.Grouper(key='end_date', freq='ME')])['percentage']
      .mean().round(2)
      .reset_index()
      .pipe(lambda df_: px.line(df_, x='end_date', y='percentage', color='Question', template='plotly_white', labels={'percentage': '% of Firms', 'end_date': 'Month/Year'}))
)

# Update layout
fig_yes.update_layout(height=400, title_text='Did you use AI? Yes', font=dict(family="Times New Roman", size=16), showlegend=False)
fig_yes.update_annotations(font=dict(family="Times New Roman", size=16))
fig_yes.update_traces(line=dict(width=3.5))

# National No graph

fig_no = (
    ai_df_no
      .groupby(['Question', pd.Grouper(key='end_date', freq='ME')])['percentage']
      .mean().round(2)
      .reset_index()
      .pipe(lambda df_: px.line(df_, x='end_date', y='percentage', color='Question', template='plotly_white', labels={'percentage': '% of Firms', 'end_date': 'Month/Year'}))
)

# Update layout
fig_no.update_layout(height=400, title_text='Did you use AI? No', font=dict(family="Times New Roman", size=16))
fig_no.update_annotations(font=dict(family="Times New Roman", size=16))
fig_no.update_traces(line=dict(width=3.5))

# States df

states_df = pd.read_excel('https://www.census.gov/hfp/btos/downloads/State.xlsx',
                          na_values='S')

# States data cleanup

states_df = (
    states_df
      .drop(['Question ID', 'Answer ID'], axis = 'columns')
      .dropna()
      .loc[lambda df_: df_["Question"].str.contains('AI')]
      .assign(
              Question = lambda df_: df_['Question'].map(ai_map),
              )
      .melt(id_vars=['State','Question', 'Answer'], value_name='percentage', var_name='end_date')
      .assign(
              end_date = lambda df_: pd.to_datetime(df_['end_date'].astype('str').map(dates_map)),
              percentage = lambda df_: df_['percentage'].str.rstrip('%').astype('float')
              )
      .groupby(['State', 'Question', 'Answer', pd.Grouper(key='end_date', freq='ME')])['percentage']
      .mean()
      .reset_index()
)

# Sector-Employees df

sector_empl = pd.read_excel('https://www.census.gov/hfp/btos/downloads/Sector%20by%20Employment%20Size%20Class.xlsx',
                            na_values='S')

# naics_codes

naics_codes = pd.read_excel('https://www.census.gov/naics/2022NAICS/2022_NAICS_Descriptions.xlsx')

# naics codes cleanup

naics_codes = (
    naics_codes
      .assign(
          sector = lambda df_: df_['Code'].astype('str'),
          title = lambda df_: df_['Title'].str.replace('T$','', regex=True).str.replace('and', '&')
      )
      .loc[lambda df_: df_['sector'].str.len() == 2]
      .drop(['Code', 'Title', 'Description'], axis='columns')
      .reset_index(drop=True)
)

# Firm size dictionary

label_to_size = {'A': 'Small', 'B': 'Small', 'C': 'Small', 'D': "Small", 'E': 'Medium', 'F': 'Medium', 'G': "Large"}

# sector and employees cleanup

sector_empl = (
    sector_empl
      .drop(['Question ID', 'Answer ID'], axis = 'columns')
      .dropna()
      .loc[lambda df_: df_["Question"].str.contains('AI | Artificial Intelligence')]
      .loc[lambda df_: df_["Sector"] != 'XX']
      .assign(
              question = lambda df_: df_['Question'].map(ai_map),
              emp_size = lambda df_: df_["Empsize"].map(label_to_size)
              )
      .drop(['Empsize', 'Question'], axis='columns')
      .melt(id_vars=['Sector', 'emp_size','question', 'Answer'], value_name='percentage', var_name='end_date')
      .assign(
               end_date = lambda df_: pd.to_datetime(df_['end_date'].astype('str').map(dates_map)),
               percentage = lambda df_: df_['percentage'].str.rstrip('%').astype('float'),
               sector = lambda df_: df_['Sector'].astype('str')
               )
       .drop('Sector', axis='columns')
       .groupby(['sector', 'emp_size', 'question', 'Answer', pd.Grouper(key='end_date', freq='ME')])['percentage']
       .mean().round(2)
       .reset_index()
       .pipe(lambda df_: pd.merge(naics_codes, df_, on='sector'))
       .rename(columns = {'title': 'industry'})
       .drop('sector', axis='columns')
)

# Instantiate the dash app

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

server = app.server

# Layout of the dashboard

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("National AI Adoption Tracker",
                        className='text-center text-success mb-4'),
                width=12),
        dbc.Col(html.P("AI is one of the transformative technologies of our times. "
                       "How US businesses adopt this technology is of utmost importance. "
                       "The US Census Bureau added supplemental content on AI to its Business Trends and Outlook Survey "
                       "asking if US businesses are using AI in creating products and services. The top two graphs show you "
                       "businesses responding Yes and No to the question of using AI. The bottom two graphs "
                       "allow you to explore the data by US states and by industry sectors and firm sizes." ,
                       className='text-primary'), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Did you use AI? Yes")),
                dbc.CardBody([
                    dcc.Graph(id='national-yes-fig', figure=fig_yes)
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Did you use AI? No")),
                dbc.CardBody([
                    dcc.Graph(id='national-no-fig', figure=fig_no)
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6),
    ], justify='center'),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("US States")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='state-dropdown', placeholder='Select US states',
                        style={'height': '20px', 'width': '100%'},
                        options=[{'label': s, 'value': s} for s in states_df['State'].unique()],
                        multi=True,
                        className='p-1'
                    ),
                    html.Div(style={'height': '20px'}),  # Adjusting space
                    dcc.Dropdown(
                        id='question-dropdown-state', placeholder='Select a question',
                        style={'height': '20px', 'width': '100%'},
                        options=[
                            {'label': 'Intend to use AI next 6 months', 'value': 'Intend'},
                            {'label': 'Used AI last 2 weeks', 'value': 'Used'}
                        ],
                        className='p-1'
                    ),
                    html.Br(),  # Adding space
                    html.Label("Select One Answer Choice"),
                    dcc.Checklist(
                        id='answer-checkbox-state',
                        options=[
                            {'label': 'Yes', 'value': 'Yes'},
                            {'label': 'No', 'value': 'No'},
                            {'label': 'Do not know', 'value': 'Do not know'}
                        ],
                        value=[],
                        inline=True,
                        labelClassName="p-1"
                    ),
                    dcc.Graph(id='states-plot', figure={})  # Initialize with empty figure
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6, className='p-2'),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Industries and Firm Sizes")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='industry-dropdown', placeholder="Select an industry",
                        style={'height': '20px', 'width': '100%'},
                        options=[{'label': industry, 'value': industry} for industry in
                                 sector_empl['industry'].unique()],
                        className='p-1'
                    ),
                    html.Div(style={'height': '20px'}),  # Adjusting space
                    dcc.Dropdown(
                        id='question-dropdown-sector', placeholder='Select a question',
                        style={'height': '20px', 'width': '100%'},
                        options=[{'label': question, 'value': question} for question in
                                 sector_empl['question'].unique()],
                        className='p-1'
                    ),
                    html.Br(),
                    html.Label("Select One Answer Choice"),
                    dcc.Checklist(
                        id='answer-checkbox-sector',
                        options=[
                            {'label': 'Yes', 'value': 'Yes'},
                            {'label': 'No', 'value': 'No'},
                            {'label': 'Do not know', 'value': 'Do not know'}
                        ],
                        value=[],  # No initial value
                        inline=True,
                        labelClassName="p-1"
                    ),
                    dcc.Graph(id='sector-empl-plot', figure={})
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6, className='p-2')
    ], justify='center'),

    html.Div([
        dbc.Row([
            dbc.Col(html.Button('Data Details PDF', id='btn-pdf', className='btn btn-success'), width=6,
                    style={'textAlign': 'left'}),
            dbc.Col(html.Img(src="/assets/aatiny.jpg", style={'marginRight': '50px'}, height="50px"), width=6,
                    style={'textAlign': 'right'}),
        ], style={"position": "fixed", "bottom": 8, "left": 8, "right": 8, "zIndex": 999}),
        dcc.Download(id='download-link'),
    ])
], fluid=True)

# Callback for the States plot

@app.callback(
    Output('states-plot', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('question-dropdown-state', 'value'),
     Input('answer-checkbox-state', 'value')]
     )
def update_states_plot(selected_states, selected_question, selected_answers):
    # If any selection is missing, return empty figure
    if not (selected_states and selected_question and selected_answers):
        return {}

    # Filter DataFrame based on selections
    filtered_df = states_df[(states_df['State'].isin(selected_states)) &
                     (states_df['Question'].str.contains(selected_question)) &
                     (states_df['Answer'].isin(selected_answers))]

    # If no data after filtering, return empty figure
    if filtered_df.empty:
        return {}

    # Calculate median for the selected data
    median_value = filtered_df['percentage'].median()

    fig = px.line(filtered_df, x='end_date', y='percentage', color='State',
                  title='', color_discrete_sequence=px.colors.qualitative.Light24,
                  width=500, height=400,
                  template='plotly_white', labels={'percentage': 'Percentage',
                                                   'end_date': 'Month/Year',
                                                   'State': 'States'})
    fig.update_traces(line=dict(width=3.5))

    # Add dashed line representing median with annotation below the line
    fig.add_shape(type='line',
                  x0=filtered_df['end_date'].min(), y0=median_value,
                  x1=filtered_df['end_date'].max(), y1=median_value,
                  line=dict(color='red', width=2, dash='dash'),
                  name='Median'
                  )
    fig.add_annotation(x=filtered_df['end_date'].max(), y=median_value - 0.5,
                       xref="x", yref="y",
                       text="National Median",
                       showarrow=False,
                       font=dict(family="Times New Roman", size=12, color="red")
                       )

    return fig

# Callback for industry sector and firm size plot

@app.callback(
    Output('sector-empl-plot', 'figure'),
    [Input('industry-dropdown', 'value'),
     Input('question-dropdown-sector', 'value'),
     Input('answer-checkbox-sector', 'value')]
)
def update_sector_plot(selected_industry, selected_question, selected_answers):
    # If any selection is missing, return an empty figure
    if not (selected_industry and selected_question and selected_answers):
        return {}

    # Convert single selected values to lists to match the logic in the filtering
    if isinstance(selected_industry, str):
        selected_industry = [selected_industry]
    if isinstance(selected_question, str):
        selected_question = [selected_question]
    if isinstance(selected_answers, str):
        selected_answers = [selected_answers]

    # Filter DataFrame based on selections
    filtered_df = sector_empl[(sector_empl['industry'].isin(selected_industry)) &
                     (sector_empl['question'].isin(selected_question)) &
                     (sector_empl['Answer'].isin(selected_answers))]

    # If no data after filtering, return an empty figure
    if filtered_df.empty:
        return {}
    median_value = filtered_df['percentage'].median()

    # Create the line plot
    fig = px.line(filtered_df, x='end_date', y='percentage', color='emp_size', color_discrete_sequence=px.colors.qualitative.Light24,
                  title='',
                  labels={'percentage': 'Percentage', 'end_date': 'Month/Year', 'emp_size': 'Firm Size'},
                  template='plotly_white',
                  width=500, height=400,)
    fig.update_xaxes(
        tickvals=filtered_df['end_date'].unique(),
        tickformat= '%b %Y'
    )

    fig.update_traces(line=dict(width=3.5))

    # Add dashed line representing median with annotation below the line
    fig.add_shape(type='line',
                  x0=filtered_df['end_date'].min(), y0=median_value,
                  x1=filtered_df['end_date'].max(), y1=median_value,
                  line=dict(color='red', width=2, dash='dash'),
                  name='Median'
                  )
    fig.add_annotation(x=filtered_df['end_date'].max(), y=median_value - 0.5,
                       xref="x", yref="y",
                       text="National Median",
                       showarrow=False,
                       font=dict(family="Times New Roman", size=12, color="red")
                       )

    return fig

# Callback for Checklist

@app.callback(
    Output('answer-checkbox-state', 'value'),
    [Input('answer-checkbox-state', 'value')]
)
def update_checklist_value(selected_values):
    # If multiple values are selected, keep only the last one
    if len(selected_values) > 1:
        return [selected_values[-1]]
    return selected_values

@app.callback(
    Output('answer-checkbox-sector', 'value'),
    [Input('answer-checkbox-sector', 'value')]
)
def update_checklist_value(selected_values):
    # If multiple values are selected, keep only the last one
    if len(selected_values) > 1:
        return [selected_values[-1]]
    return selected_values

# Download PDF

@app.callback(
    Output('download-link', 'data'),
    Input('btn-pdf', 'n_clicks'),
    prevent_initial_call=True
)
def trigger_download(n_clicks):
    if n_clicks:
        file_path = 'assets/BTOS_AI_Data_Description.pdf'
        return dcc.send_file(file_path)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
