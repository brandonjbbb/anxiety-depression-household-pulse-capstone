import pandas as pd,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
d=pd.read_pickle('d.pkl');R=pd.read_csv('results.csv')
# Fig 5 pipeline
fig,ax=plt.subplots(figsize=(9,3.1));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3)
steps=['Cleaned data\n(16,087 rows,\nSQLite / CSV)','Features\nIndicator, Group,\nSubgroup, State,\ntime, month','One-hot encode\n+ time-based\ntrain/test split','Train models\nLinear, Random\nForest, Gradient\nBoosting','Evaluate\nMAE, RMSE, R²\n+ feature\nimportance','Compare with\nARIMA / ETS\n(national series)']
w=1.45
for i,s in enumerate(steps):
    x=0.1+i*1.65
    ax.add_patch(FancyBboxPatch((x,0.9),w,1.5,boxstyle='round,pad=0.03',fc='#e8f0fa',ec='#2b5c9b'))
    ax.text(x+w/2,1.65,s,ha='center',va='center',fontsize=8)
    if i<5: ax.annotate('',xy=(x+1.62,1.65),xytext=(x+w+0.03,1.65),arrowprops=dict(arrowstyle='->',lw=1.5))
ax.text(5,0.35,'Target (dependent variable): Value = weighted % of adults with symptoms',ha='center',fontsize=9,style='italic')
plt.tight_layout();plt.savefig('fig5_pipeline.png',dpi=200);plt.close()
# Fig 6 results
fig,axs=plt.subplots(1,2,figsize=(11,4))
ax=axs[0]
n=d[(d.Group=='National Estimate')&(d.Indicator=='Symptoms of Anxiety Disorder or Depressive Disorder')].sort_values('Start Date')
ax.plot(n['Start Date'],n.Value,'-',color='#444',lw=1.8,label='Actual (national)')
for k,c,lab in [('A','#d95f02','Predicted, split A (test 2023)'),('B','#1b9e77','Predicted, split B (test 2024)')]:
    t=pd.read_pickle(f'teRF{k}.pkl');t=t[(t.Group=='National Estimate')&(t.Indicator=='Symptoms of Anxiety Disorder or Depressive Disorder')].sort_values('Start Date')
    ax.plot(t['Start Date'],t.pred,'--o',ms=3,color=c,label=lab)
ax.axvspan(pd.Timestamp('2023-10-30'),pd.Timestamp('2024-01-09'),color='#ccc',alpha=.5)
ax.text(pd.Timestamp('2023-11-05'),38,'break',fontsize=8,rotation=90,va='top')
import matplotlib.dates as md; ax.xaxis.set_major_locator(md.YearLocator()); ax.xaxis.set_major_formatter(md.DateFormatter('%Y')); ax.set_ylabel('% of adults with symptoms');ax.set_title('Random Forest predictions vs. actual',fontsize=10);ax.legend(fontsize=7,loc='lower left');ax.spines[['top','right']].set_visible(False)
ax=axs[1]
ms=['Mean baseline','Linear Regression','Random Forest','Gradient Boosting'];import numpy as np
x=np.arange(4);wd=.38
for j,(sp,c,l) in enumerate([('A','#d95f02','A: test 2023'),('B','#1b9e77','B: test 2024')]):
    v=[R[(R.Split.str[0]==sp)&(R.Model==m)].RMSE.values[0] for m in ms]
    b=ax.bar(x+(j-.5)*wd,v,wd,color=c,label=l)
    for xi,vi in zip(x+(j-.5)*wd,v):ax.text(xi,vi+.2,f'{vi:.1f}',ha='center',fontsize=7)
ax.set_xticks(x);ax.set_xticklabels([m.replace(' ','\n') for m in ms],fontsize=8);ax.set_ylabel('RMSE (percentage points)');ax.set_title('Test error by model',fontsize=10);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
plt.tight_layout();plt.savefig('fig6_results.png',dpi=200);plt.close()
