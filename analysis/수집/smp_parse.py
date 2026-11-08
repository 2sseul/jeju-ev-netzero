import re,glob,sys,os
import pandas as pd, numpy as np
R=sys.argv[1]
t=open(f'{R}/sample.txt').read(); i=t.find('c24 ='); print(t[i:i+900])
rows=[]
for f in sorted(glob.glob(f'{R}/20[0-9][0-9][0-9][0-9].txt')):
    s=open(f).read()
    for block in s.split('gridData.push')[:-1] if 'gridData.push' in s else [s]:
        vals=re.findall(r'c(\d+)\s*=\s*textFormmat\("(-?[\d.,]*)"',block)
        tail=s.split('gridData.push')[1] if False else ''
    # 날짜와 값 함께 추출
    for m in re.finditer(r'((?:c\d+\s*=\s*textFormmat\("[^"]*",count\);\s*)+)(.*?)gridData\.push\(\{(.*?)\}\)',s,re.S):
        v=dict((int(a),b) for a,b in re.findall(r'c(\d+)\s*=\s*textFormmat\("([^"]*)"',m.group(1)))
        d=re.search(r'(\d{4})[./-]?(\d{2})[./-]?(\d{2})',m.group(3)) or re.search(r'(\d{4})[./-]?(\d{2})[./-]?(\d{2})',m.group(2))
        rows.append((d.group(0) if d else None,*[v.get(k) for k in range(1,28)]))
d=pd.DataFrame(rows,columns=['date']+[f'c{i}' for i in range(1,28)])
print(d.shape,d.head(3).iloc[:,:6].to_string()); print(d.tail(2).iloc[:,[0,1,24,25,26,27]].to_string())
d.to_pickle(f'{R}/parsed.pkl')
