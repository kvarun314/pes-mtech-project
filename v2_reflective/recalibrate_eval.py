import json,numpy as np
from sklearn.linear_model import LogisticRegression
def load(pf,rf):
    rows={json.loads(l)["review_id"]:json.loads(l) for l in open(rf)}
    P=[];y=[];L=[]
    for l in open(pf):
        r=json.loads(l);P.append(r["probs"]);y.append(r["gt_rating"]-1);L.append(len(rows[r["review_id"]]["review_text"]))
    return np.log(np.clip(np.array(P),1e-6,1)),np.array(y),np.log1p(np.array(L))
Xd,yd,Ld=load(".scratch/kaggle_dev_probs.jsonl",".scratch/kaggle_dev_rows.jsonl")
Xe,ye,Le=load(".scratch/kaggle_eval_probs.jsonl",".scratch/kaggle_eval_rows.jsonl")
base=(Xe.argmax(1)==ye);print("eval n",len(ye),"base exact",base.mean().round(4))
rng=np.random.default_rng(0)
for name,fd,fe,C in(("logprobs C=1",Xd,Xe,1),("logprobs+len C=10",np.c_[Xd,Ld],np.c_[Xe,Le],10)):
    m=LogisticRegression(C=C,max_iter=3000).fit(fd,yd);p=m.predict(fe);ok=(p==ye)
    d=ok.astype(float)-base;bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]
    rec=[round(float(((p==k)&(ye==k)).sum()/(ye==k).sum()),2) for k in range(5)]
    print(name,"exact",ok.mean().round(4),"within1",(abs(p-ye)<=1).mean().round(4),"MAE",abs(p-ye).mean().round(3),"delta CI",np.percentile(bs,[2.5,97.5]).round(3),"recall",rec)
print("base recall",[round(float(((Xe.argmax(1)==k)&(ye==k)).sum()/(ye==k).sum()),2) for k in range(5)],"base within1",(abs(Xe.argmax(1)-ye)<=1).mean().round(4))
