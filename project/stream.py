from typing import Dict, Union 
import numpy as np 
from datetime import datetime 
from river.compose import Pipeline, Discard, FuncTransformer, TargetTransformRegressor
from river.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px


def train(data, models: Dict[str, Pipeline]) -> Dict[str, np.ndarray]:
    preds = {name: [] for name in models.keys()}
    for x, y in data:
        for name, model in models.items():
            preds[name].append(model.predict_one(x))
            model.learn_one(x, y)
    return {name: np.array(pred) for name, pred in preds.items()}

def plot_preds(preds: Dict[str, np.ndarray], target: np.ndarray, metrics: Dict[str, float]):
    fig = go.Figure()
    colors = px.colors.qualitative.D3
    fig.add_trace(go.Scatter(y=target, name='Temperature', line=dict(color='black')))
    for i, name in enumerate(preds.keys()):
        fig.add_trace(go.Scatter(y=preds[name], name=f'{name} ({metrics[name]:.2})', line=dict(color=colors[i])))
    fig.update_layout(width=1000, height=600, title_text='Temperature', template='seaborn', 
                      margin=dict(t=60, b=20, l=20, r=20))
    return fig


def get_date_features(x: Dict[str, str]) -> Dict[str, float]:
    month = int(x['Date'].split("/")[1])
    season = ((month-1) % 12)//4

    time_format = "%H:%M:%S"
    datetime_object = datetime.strptime(x['Time'], time_format)

    hour_of_day = datetime_object.hour

    time_cos = np.cos(hour_of_day)
    time_sin = np.sin(hour_of_day)
    
    # cast to float 
    x.update({'month':month, 'season':season, 'time_cos': time_cos, 'time_sin':time_sin}) 
    return x
             
             
class SequentialImputer(object):
    def __init__(self, detect = np.isnan, target: bool = False):
        self.last = dict() if not target else 0
        self.detect = detect
        self.__class__.__name__ = 'SequentialImputer'
        
    
    def __call__(self, x: Union[Dict[str, float], float]):
        if isinstance(x, dict):
            x = {feat: self.last[feat] if self.detect(value) else value for feat, value in x.items()}
        else:
            x = self.last if self.detect(x) else x
        self.last = x 
        return x
    
    @property
    def __name__(self):
        return 'SequentialImputer'
    

def cast_float(x: str):
    try: 
        return float(x)
    except ValueError:
        return x
    
    

def add_preprocess(model: Pipeline, use_time: bool = True, impute: bool = True) -> Pipeline:
    steps = []
    if use_time:
        steps.append(FuncTransformer(get_date_features))
    steps.append(Discard('Time', 'Date'))
    if impute:
        steps.append(FuncTransformer(SequentialImputer(lambda x: x == -200)))
        imp = SequentialImputer(lambda x: x == -200, target=True)
        model = TargetTransformRegressor(model, func=imp, inverse_func=imp)
    return Pipeline(*steps, StandardScaler(), model)