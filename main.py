import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Textarea(id='textarea1', placeholder='Enter text here...', style={'width': '100%', 'height': 200}),
        dcc.Textarea(id='textarea2', placeholder='Enter text here...', style={'width': '100%', 'height': 200}),
        html.Button('Concatenate', id='button', n_clicks=0, style={'margin-top': '10px'}),
    ], style={'display': 'inline-block', 'width': '45%', 'vertical-align': 'top'}),
    html.Div([
        html.Div(id='output-div', style={'white-space': 'pre-line'}),
        html.Div(id='labels-div', style={'margin-top': '10px'}),
    ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'}),
], style={'text-align': 'center'})

@app.callback(
    Output('output-div', 'children'),
    Output('labels-div', 'children'),
    Input('button', 'n_clicks'),
    State('textarea1', 'value'),
    State('textarea2', 'value')
)
def update_output_div(n_clicks, text1, text2):
    if n_clicks > 0:
        concatenated_text = text1 + '\n' + text2
        return concatenated_text, [html.Label(concatenated_text.split('\n')[i]) for i in range(min(5, len(concatenated_text.split('\n'))))]
    return '', []

if __name__ == '__main__':
    app.run_server(debug=True)
