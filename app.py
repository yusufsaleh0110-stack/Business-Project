from pathlib import Path
import json, joblib, numpy as np, pandas as pd, streamlit as st
RATIO_INFO={
'ROA':'Return on Assets','ROE':'Return on Equity','ROS':'Return on Sales','L1':'Cash Ratio','L2':'Quick Ratio','L3':'Current Ratio','ATD':'Asset Turnover Days','DTR':'Days Total Receivables Outstanding','TAT':'Total Asset Turnover','ITD':'Inventory Turnover Days','DA':'Debt-to-Assets Ratio','DE':'Debt-to-Equity Ratio','FL':'Financial Leverage','ROI':'Return on Investment','DIR':'Debt to Income Ratio','DCR':'Debt Service Coverage Ratio','ACR':'Asset Coverage Ratio','LP':'Labor Productivity','BL':'Bank Liabilities to Debt Ratio','LRR':'Labor-to-Revenue Ratio','WAR':'Wages to Added Value Ratio'}
CODES=list(RATIO_INFO); FEATURES=[f'{r}_tminus{yr}' for yr in (3,2,1) for r in CODES]
GUIDANCE={
'L1':('Liquidity / cash','Review cash-flow monitoring, near-term cash requirements and short-term obligations.'),'L2':('Liquidity','Review liquid assets relative to current liabilities and monitor short-term payment capacity.'),'L3':('Working capital','Review working-capital management, receivables, inventory and current liabilities.'),'DTR':('Receivables','Review collection periods, overdue receivables and customer credit controls.'),'ITD':('Inventory','Review inventory levels, slow-moving stock and working capital tied up in inventory.'),'DA':('Debt burden','Review the debt structure and whether liabilities are high relative to assets.'),'DE':('Debt structure','Review leverage and, where appropriate, discuss debt restructuring/refinancing options with a qualified adviser.'),'FL':('Leverage','Review reliance on debt financing and debt-servicing capacity.'),'DIR':('Debt burden','Review debt commitments relative to income and identify avoidable financing pressure.'),'DCR':('Debt service','Review cash available for debt service and upcoming repayment commitments.'),'ACR':('Asset coverage','Review the extent to which assets provide coverage for financial obligations.'),'ROA':('Profitability','Review asset utilisation and cost drivers that may be reducing returns.'),'ROE':('Profitability','Review profitability, financing structure and persistent reductions in shareholder returns.'),'ROS':('Operating margin','Review pricing, operating costs and potential cost-reduction opportunities.'),'ROI':('Investment return','Review low-return activities and capital allocation.'),'TAT':('Asset efficiency','Review whether assets are generating sufficient revenue.'),'ATD':('Asset efficiency','Review the speed with which assets support revenue generation.'),'LP':('Productivity','Review labour productivity and operational efficiency.'),'LRR':('Labour cost','Review labour costs relative to revenue and operational efficiency.'),'WAR':('Wage pressure','Review wage costs relative to value added and opportunities for efficiency improvement.'),'BL':('Liability structure','Review bank liabilities within the overall debt structure.')}

def load():
    try:return joblib.load('models/best_model.joblib'),json.loads(Path('models/metadata.json').read_text(encoding='utf-8'))
    except:return None,None

def band(p):
    # Presentation bands only; model classification threshold remains separate.
    return ('Low','Routine monitoring is appropriate.') if p<.30 else (('Moderate','Review the main financial warning factors and monitor the position closely.') if p<.60 else ('High','Strong model-based distress signal. Review the financial position promptly and consider qualified professional advice.'))

def local_effects(model,row,reference):
    """Model-agnostic local sensitivity: change one feature to its training median and measure probability change."""
    base=float(model.predict_proba(row[FEATURES])[0,1]); effects={}
    for f in FEATURES:
        alt=row[FEATURES].copy(); alt.loc[alt.index[0],f]=reference.get(f,0.0)
        effects[f]=base-float(model.predict_proba(alt)[0,1])
    return effects

def explain_effect(f,s):
    code=f.split('_tminus')[0]; yr=f.split('_tminus')[1]
    direction='increases' if s>0 else 'reduces'
    return f"**{RATIO_INFO[code]} ({code}), {yr} year(s) prior:** this value {direction} the model-estimated distress probability relative to replacing it with the training-data median."

st.set_page_config(page_title='European SME Bankruptcy Early Warning',page_icon='📊',layout='wide')
st.title('European SME Bankruptcy Early-Warning Decision-Support Tool')
st.caption('Python machine-learning prototype using the Drotár et al. Slovak SME bankruptcy dataset')
model,meta=load()
if model is None: st.error('No trained model found. Run: python download_data.py, then python train_model.py'); st.stop()
reference=meta.get('feature_medians',{})
with st.sidebar:
 st.subheader('Dataset & model'); st.write('Dataset: Slovak SMEs (Drotár et al., 2019)'); st.write('Selected model:',meta['best_model']); st.write('Total observations:',f"{meta['total_rows']:,}"); st.write('Bankrupt:',f"{meta['bankrupt_rows']:,}"); st.write('Non-bankrupt:',f"{meta['nonbankrupt_rows']:,}"); st.write('Bankruptcy rate:',f"{meta['bankruptcy_rate']:.2%}"); st.warning('Research diagnostic prototype only. It is not legal, accounting, lending, investment or financial advice.')

t1,t2,t3,t4,t5=st.tabs(['Single SME Assessment','Batch Assessment','Model Evaluation','Dataset Information','Methodology'])
with t1:
 st.info('Enter the 21 financial ratios for each of the three years before evaluation. Use the same definitions/units as the source dataset.')
 vals={}
 for yr in (1,2,3):
  with st.expander(f'{yr} year(s) before evaluation',expanded=(yr==1)):
   cols=st.columns(3)
   for i,c in enumerate(CODES): vals[f'{c}_tminus{yr}']=cols[i%3].number_input(f'{RATIO_INFO[c]} ({c})',value=float(reference.get(f'{c}_tminus{yr}',0.0)),format='%.6f',key=f'{c}_{yr}')
 if st.button('Assess bankruptcy risk',type='primary'):
  row=pd.DataFrame([{f:vals[f] for f in FEATURES}]); p=float(model.predict_proba(row)[0,1]); level,msg=band(p)
  a,b,c=st.columns(3); a.metric('Estimated bankruptcy probability',f'{p:.1%}'); b.metric('Decision-support band',level); c.metric('Model classification', 'Distress' if p>=meta.get('threshold',.5) else 'Non-distress')
  st.info(msg); st.caption('Low (<30%), Moderate (30–<60%) and High (≥60%) are application presentation bands, not validated legal/financial bankruptcy thresholds. The fitted classifier uses the separate threshold shown in Model Evaluation.')
  effects=local_effects(model,row,reference)
  ranked=sorted(effects.items(),key=lambda x:abs(x[1]),reverse=True)[:8]
  st.subheader('Factor-level plain-language interpretation')
  st.caption('Local sensitivity explanation: each feature is individually replaced by its training-set median while all other inputs remain unchanged. Positive effects raise the estimated distress probability; negative effects reduce it. This is explanatory decision support, not causal inference.')
  for f,s in ranked: st.write(explain_effect(f,s))
  st.subheader('Targeted managerial decision-support')
  seen=set(); shown=0
  for f,s in ranked:
   if s<=0: continue
   code=f.split('_tminus')[0]
   if code in seen: continue
   seen.add(code); title,txt=GUIDANCE[code]; st.write(f'**{title}:** {txt}'); shown+=1
   if shown==5: break
  if not shown: st.write('No major adverse factor was identified by the local sensitivity check. Continue routine monitoring and review the complete financial position rather than relying on this prediction alone.')
  st.caption('These actions are prompts for managerial review only. They do not prescribe a particular financial decision.')
with t2:
 up=st.file_uploader('Upload CSV containing the 63 model feature columns',type='csv')
 if up:
  df=pd.read_csv(up); miss=[c for c in FEATURES if c not in df]
  if miss: st.error('Missing required columns: '+', '.join(miss[:12])+(' ...' if len(miss)>12 else ''))
  else:
   out=df.copy(); out['bankruptcy_probability']=model.predict_proba(df[FEATURES])[:,1]; out['decision_support_band']=out.bankruptcy_probability.apply(lambda x:band(float(x))[0]); out['model_classification']=np.where(out.bankruptcy_probability>=meta.get('threshold',.5),'Distress','Non-distress'); st.dataframe(out,use_container_width=True); st.download_button('Download predictions',out.to_csv(index=False).encode(),'predictions.csv','text/csv')
with t3:
 st.subheader('Model selection')
 st.write(f"**Selected model:** {meta['best_model']}")
 st.write('**Selection rule:** highest test ROC-AUC; F1-score is used as the secondary ordering criterion. This prioritises discrimination and imbalance-aware evaluation rather than accuracy alone.')
 st.write(f"**Classification threshold:** {meta.get('threshold',.5):.2f}")
 p=Path('outputs/model_comparison.csv')
 if p.exists():
  m=pd.read_csv(p); st.subheader('Logistic Regression vs Random Forest vs Gradient Boosting'); st.dataframe(m.style.format({x:'{:.4f}' for x in ['accuracy','precision','recall','f1','roc_auc']}),use_container_width=True); st.bar_chart(m.set_index('model')[['precision','recall','f1','roc_auc']])
 c=Path('outputs/confusion_matrix.csv')
 if c.exists(): st.subheader('Selected-model confusion matrix'); st.dataframe(pd.read_csv(c,index_col=0))
 r=Path('outputs/classification_report.txt')
 if r.exists(): st.subheader('Classification report'); st.code(r.read_text())
with t4:
 st.write('**Source:** Drotár, P., Gnip, P., Zoričak, M. and Gazda, V. (2019), Dataset of financial ratios of Slovak companies, Mendeley Data, Version 2, DOI: 10.17632/j89csb932y.2.')
 st.write('The data contain 21 financial ratios for three consecutive years before evaluation, across agriculture, construction, manufacturing and retail, with bankrupt and non-bankrupt SME classes.')
 a,b,c=st.columns(3); a.metric('Total SMEs',f"{meta['total_rows']:,}"); b.metric('Bankrupt',f"{meta['bankrupt_rows']:,}"); c.metric('Non-bankrupt',f"{meta['nonbankrupt_rows']:,}")
 st.subheader('Class distribution'); st.bar_chart(pd.DataFrame({'Observations':[meta['nonbankrupt_rows'],meta['bankrupt_rows']]},index=['Non-bankrupt','Bankrupt']))
 st.write(f"Bankruptcy observations represent **{meta['bankruptcy_rate']:.2%}** of the prepared dataset, demonstrating the severe class imbalance addressed during model training.")
 st.subheader('Financial ratios'); st.dataframe(pd.DataFrame({'Code':CODES,'Financial ratio':[RATIO_INFO[c] for c in CODES]}),hide_index=True,use_container_width=True)
 st.write('**Feature construction:** each of the 21 ratios is retained for each of the three years before evaluation (t−3, t−2 and t−1), producing **63 time-specific model predictors**.')
 st.caption('The dataset is Slovak; results should not be presented as representative of all European SMEs without additional validation.')
with t5:
 st.subheader('Implemented ML pipeline')
 st.write('**Raw official SME CSV files → validation and numeric conversion → 63 three-year ratio features → stratified train/test split → median missing-value imputation → class-imbalance treatment → Logistic Regression / Random Forest / Gradient Boosting training → Precision, Recall, F1 and ROC-AUC evaluation → best-model selection → risk prediction → factor-level explanation → managerial decision-support framing.**')
 st.write('**Missing values:** median imputation is fitted within each model pipeline using training data.')
 st.write('**Scaling:** robust scaling is applied to Logistic Regression; tree-based models do not require scaling.')
 st.write('**Class imbalance:** Logistic Regression uses balanced class weights, Random Forest uses balanced-subsample weights, and Gradient Boosting receives balanced sample weights.')
 st.write('**Evaluation:** the held-out test set is stratified so the rare bankrupt class is represented proportionally. Precision, Recall, F1 and ROC-AUC are reported alongside accuracy.')
 st.write('**Interpretability:** single-SME explanations use model-agnostic local sensitivity against training medians and translate the strongest adverse financial factors into plain-language review prompts.')
 st.write('**Scope boundary:** the application is an early-warning diagnostic/decision-support prototype. It does not incorporate macroeconomic volatility, industry-specific micro-structures or subjective managerial conduct, and it does not provide professional financial advice.')
