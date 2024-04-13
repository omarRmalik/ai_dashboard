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
             '202410': '05/05/2024', '202411':'05/19/2024', '202412':'06/02/2024', '202413': '06/16/2024', '202414': '06/30/2024'}

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
fig_yes.update_layout(height=400, title_text='Used AI: Yes', font=dict(family="Times New Roman", size=16))
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
fig_no.update_layout(height=400, title_text='Used AI: No', font=dict(family="Times New Roman", size=16))
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

# Instantiate the dash app

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
# Layout of the dashboard

app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(html.H1("National AI Adoption Tracker",
                        className='text-center text-primary mb-4'),
                width=12)
    ),

    dbc.Row([

        dbc.Col([
           dcc.Graph(id='national-yes-fig', figure=fig_yes)
        ], width={'size':6, 'offset':0, 'order':1},
           xs=12, sm=12, md=12, lg=6, xl=6
        ),

        dbc.Col([
           dcc.Graph(id='national-no-fig', figure=fig_no)
        ], width={'size':6, 'offset':0, 'order':2},
           xs=12, sm=12, md=12, lg=6, xl=6
        ),

    ], justify='center'),

    dbc.Row([
        dbc.Col([
                html.Div([
                html.H2(children='AI Use across US States', style={'textAlign': 'left', 'color': 'green'}),
                html.Label("Select States"),
                dcc.Dropdown(
                        id='state-dropdown',
                        style={'height': '30px', 'width': '800px'},
                        options=[{'label': s, 'value': s} for s in states_df['State'].unique()],
                        multi=True,
                        value=[],  # Default value empty list
                ),
                html.Br(),  # Adding space
                html.Label("Select Question"),
                dcc.Dropdown(
                        id='question-dropdown',
                        style={'height': '30px', 'width': '800px'},
                        options=[
                            {'label': 'Intend to use AI next 6 months', 'value': 'Intend'},
                            {'label': 'Used AI last 2 weeks', 'value': 'Used'}
                        ],
                        value='None',  # Default value is None
                ),
                html.Br(),  # Adding space
                html.Label("Select Answer"),
                dcc.Checklist(
                    id='answer-checkbox',
                    options=[
                            {'label': 'Yes', 'value': 'Yes'},
                            {'label': 'No', 'value': 'No'},
                            {'label': 'Do not know', 'value': 'Do not know'}
                    ],
                value=[],
                inline=True
                ),
        dcc.Graph(id='states-plot', figure={})  # Initialize with empty figure
        ])
            ,
        ], width={'size':5, 'offset':1},
           xs=12, sm=12, md=12, lg=5, xl=5
        )

    ], justify='left')


])

# Callback for the States plot

@app.callback(
    Output('states-plot', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('question-dropdown', 'value'),
     Input('answer-checkbox', 'value')]
)
def update_plot(selected_states, selected_question, selected_answers):
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
                  title='Percentage of Firms Using AI in creating products and services', color_discrete_sequence=px.colors.qualitative.Light24,
                  width=800, height=400,
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


if __name__=='__main__':
    app.run_server(debug=True, port=8000)