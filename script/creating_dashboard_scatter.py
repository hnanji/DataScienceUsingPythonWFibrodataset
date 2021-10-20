"""
Generating scatter plots
need tio import plotly graph objects to plot scatter plot
The graph component takes a figure object thhat has the data and the layout deescription
the scatter plot is dione using the graph_objs scatter property
to ensure plot is a scatter plot, we pass thhe mode attribute and setr it as markers otherwise we would have lines on the graph
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

app  =  dash.Dash()

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/' +
    '5d1ea79569ed194d432e56108a04d188/raw/' +
    'a9f9e8076b837d541398e999dcbac2b2826a81f8/'+
    'gdp-life-exp-2007.csv')


app.layout = html.Div([
 
dcc.Graph(
        id='life-exp-vs-gdp',  # sort of name of this graph
        figure={
            'data': [
                go.Scatter(
                	# ignoring the continent bit, could would have been x=df['gdp per capita']
                    x=df[df['continent'] == i]['gdp per capita'], # gpd displayed for the different continents
                    y=df[df['continent'] == i]['life expectancy'], # life expencatncy displayed for the different continents
                    text=df[df['continent'] == i]['country'],  # different continents grouped accoring to country
                    mode='markers',
                    opacity=0.8,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.continent.unique()
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                yaxis={'title': 'Life Expectancy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
       
])

if __name__ == '__main__':
    app.run_server(debug=True)

