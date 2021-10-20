"""
The aim of this script is to practise creating dasgboards using Dash
Thee following packages must be install before building dash applications
The core dash backend.
Dash front-end
Dash HTML components
Dash core components
Plotly
"""

"""
# Data App layout

# dash applicatioon has two parts-layout and the second part describes the interactivity
#dash has html class that  enables us to generate html content with python(need to import dash-core component and dash html component)
# the next chuck oif code shuld be in a file called app.py that imports the packages
# we inialise dash by calling the dash class of dash to create layout
# we can use DIV class ffrom dash_html to create HTMT DIV
# The <div> tag defines a division or a section in an HTML document. 
# The <div> tag is used as a container for HTML elements
# Any sort of content can be put inside the <div> tag!
# graph render interactive data visyalisatioon using plotly.js(graph expects a figure object with data to be plotted amnd layoit details)
# backgroun of the graph could be changesd using style attribute


"""

# import packages
import dash
import dash_core_components as dcc
import dash_html_components as html


app = dash.Dash()
colors = {
    'background': '#111111',  # changing this  siz digit number changes the background colour of the display
    'text': '#7FDBFF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',  # positioin could be changes from right to left position for hello Dash
            'color': colors['text']
        }
    ),
    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(
        id='Graph1',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
])

if __name__ == '__main__':
#debug is set to true to ensure tyhat we dont have to keep refreshing server any time changes arev made to the graph
	
   app.run_server(debug=True)









