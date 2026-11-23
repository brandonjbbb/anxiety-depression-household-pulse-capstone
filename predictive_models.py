import pandas as pd,numpy as np,warnings
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error as MAE,mean_squared_error,r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
d=pd.read_csv('/home/claude/clean/household_pulse_clean.csv',parse_dates=['Start Date','End Date'])
d['t']=(d['Start Date']-pd.Timestamp('2020-04-23')).dt.days/30.4
d['month']=d['Start Date'].dt.month;d['year']=d['Start Date'].dt.year
cat=['Indicator','Group','Subgroup','State'];num=['t','month']
def mk(m):return Pipeline([('p',ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),cat),('n','passthrough',num)])),('m',m)])
def models():return {'Mean baseline':DummyRegressor(),'Linear Regression':LinearRegression(),
 'Random Forest':RandomForestRegressor(300,min_samples_leaf=3,random_state=42,n_jobs=-1),
 'Gradient Boosting':GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=42)}
rows=[];preds={}
for split,(lo,te_y) in {'A: train 2020-2022, test 2023':(2022,2023),'B: train 2020-2023, test 2024':(2023,2024)}.items():
    tr=d[d.year<=lo];te=d[d.year==te_y]
    for n,m in models().items():
        p=mk(m).fit(tr[cat+num],tr.Value);pr=p.predict(te[cat+num]);preds[(split[0],n)]=(te,pr,p)
        rows.append((split,n,len(tr),len(te),MAE(te.Value,pr),mean_squared_error(te.Value,pr)**.5,r2_score(te.Value,pr),(pr-te.Value).mean()))
R=pd.DataFrame(rows,columns=['Split','Model','nTrain','nTest','MAE','RMSE','R2','Bias']);print(R.round(2).to_string())
R.to_csv('results.csv',index=False)
# shuffled
from sklearn.model_selection import train_test_split
a,b=train_test_split(d,test_size=.2,random_state=42)
for n,m in list(models().items())[1:]:
    p=mk(m).fit(a[cat+num],a.Value);pr=p.predict(b[cat+num]);print('shuffled',n,round(MAE(b.Value,pr),2),round(mean_squared_error(b.Value,pr)**.5,2),round(r2_score(b.Value,pr),3))
# importance split A RF (permutation-free impurity grouped)
for sp in 'AB':
    p=preds[(sp,'Random Forest')][2];names=p['p'].get_feature_names_out();imp=p['m'].feature_importances_;g={}
    for nm,i in zip(names,imp):
        k=nm.split('__')[1].split('_')[0] if nm.startswith('c__') else nm.split('__')[1];g[k]=g.get(k,0)+i
    G=pd.Series(g).sort_values(ascending=False);print(sp,G.round(3).to_dict());G.to_csv(f'imp{sp}.csv')
# national time series
n=d[(d.Group=='National Estimate')&(d.Indicator=='Symptoms of Anxiety Disorder or Depressive Disorder')].sort_values('Start Date')
out=[]
for sp,lo,ty in [('A',2022,2023),('B',2023,2024)]:
    tr=n[n.year<=lo].Value.values;te=n[n.year==ty].Value.values;h=len(te);x=np.arange(len(tr))
    f={'ARIMA(1,1,0)':ARIMA(tr,order=(1,1,0)).fit().forecast(h),'Exp. smoothing (damped trend)':ExponentialSmoothing(tr,trend='add',damped_trend=True).fit().forecast(h),
       'Linear trend':np.polyval(np.polyfit(x,tr,1),np.arange(len(tr),len(tr)+h)),'Naive (last value)':np.repeat(tr[-1],h)}
    for k,v in f.items():out.append((sp,k,h,MAE(te,v),mean_squared_error(te,v)**.5))
T=pd.DataFrame(out,columns=['Split','Model','nTest','MAE','RMSE']);print(T.round(2).to_string());T.to_csv('ts.csv',index=False)
# save preds for plotting (GB/RF split A, national)
te,pr,_=preds[('A','Gradient Boosting')];te=te.copy();te['pred']=pr;te.to_pickle('teA.pkl')
for k in ['A','B']:
    te,pr,_=preds[(k,'Random Forest')];te=te.copy();te['pred']=pr;te.to_pickle(f'teRF{k}.pkl')
d.to_pickle('d.pkl')
