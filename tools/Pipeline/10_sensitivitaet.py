#!/usr/bin/env python3
"""
Ergaenzende Auswertung fuer subsec:auswertung und subsubsec:robustheit.

A  Differenz der Abfaelle (Pruefgroesse H3), Variante gegen basis und gegen filler
B  Sensitivitaet: unkorrigiertes S nach eq:selektivitaet
C  Sensitivitaet: ohne Aufgaben mit Antwortleckage (V14)
D  Sensitivitaet: Redundanz nur ueber Woerter ohne Entsprechung im Kern (V13)

Nur lesend auf promptanalyse.db. Aufruf: python3 10_sensitivitaet.py
"""
import sqlite3, sys, math
sys.path.insert(0, '.')
import dbio
from collections import defaultdict
c = sqlite3.connect('file:../../results/promptanalyse.db?mode=ro', uri=True)
c.row_factory = sqlite3.Row
V = ('filler','semantic','instruction','demonstration')
def mki(w):
    n=len(w); m=sum(w)/n; se=math.sqrt(sum((x-m)**2 for x in w)/(n-1)/n); return m,m-1.96*se,m+1.96*se,n,se
def pone(m,se): return 1-0.5*(1+math.erf((m/se)/math.sqrt(2))) if se else 0.0
# Leckage-Aufgaben
leak=set(r[0] for r in c.execute("select frage_id from prompt where antwortleckage=1"))
print('Aufgaben mit Leckage:',len(leak))
# Genauigkeit
acc=defaultdict(dict)
for r in c.execute("select frage_id,variante,rho_ziel,korrekt from v_studie_b where wiederholung=1"):
    acc[(r['variante'],r['rho_ziel'])][r['frage_id']]=r['korrekt']
tasks=sorted(acc[('basis',0.0)])
def did(excl=set()):
    for rho in (0.25,0.5,0.75):
        for v in V:
            d=[(acc[(v,rho)][t]-acc[(v,0.0)][t])-(acc[('basis',rho)][t]-acc[('basis',0.0)][t]) for t in tasks if t not in excl]
            m,lo,hi,n,se=mki(d); print('  DiD',rho,v,'%+.1f [%+.1f; %+.1f] n=%d p1=%.4f'%(100*m,100*lo,100*hi,n,pone(m,se)))
print('A) Differenz der Abfaelle (Variante minus basis), Prozentpunkte'); did()
def vors(excl):
    for v in V:
        d=[acc[(v,0.5)][t]-acc[('basis',0.5)][t] for t in tasks if t not in excl]
        m,lo,hi,n,se=mki(d); print('  Vorsprung 0.5',v,'%+.1f [%+.1f; %+.1f]'%(100*m,100*lo,100*hi))
# Selektivitaet
def sel(col, excl=set()):
    d=defaultdict(dict)
    for r in c.execute("select frage_id,variante,rho_ziel,%s x from v_bedingung where rho_ziel in (0.25,0.5)"%col):
        if r['x'] is not None and r['frage_id'] not in excl: d[(r['variante'],r['rho_ziel'])][r['frage_id']]=r['x']
    for rho in (0.25,0.5):
        for v in V:
            w=list(d[(v,rho)].values()); m,lo,hi,n,se=mki(w); print('  S',rho,v,'%.3f [%.3f; %.3f] n=%d'%(math.exp(m),math.exp(lo),math.exp(hi),n))
        for a,b in (('filler','semantic'),('semantic','instruction'),('instruction','demonstration'),('filler','instruction')):
            ts=[t for t in d[(a,rho)] if t in d[(b,rho)]]
            m,lo,hi,n,se=mki([d[(a,rho)][t]-d[(b,rho)][t] for t in ts])
            print('  K',rho,a,'>',b,'%.2f [%.2f; %.2f] p1=%.4f'%(math.exp(m),math.exp(lo),math.exp(hi),pone(m,se)))
print('B) unkorrigiertes S (ln_s)'); sel('ln_s')
print('C) ohne Leckage-Aufgaben: S frei'); sel('ln_s_frei',leak); vors(leak); print('  DiD ohne Leckage'); did(leak)
# D) ueberlappungsbereinigt
core=defaultdict(set); words=defaultdict(list)
for r in c.execute("select w.wort_id,w.prompt_id,w.form,w.schluessel,w.ist_wort,w.ist_ziffer,w.herkunft from wort w"):
    if not r['ist_wort']: continue
    if r['herkunft']=='kern': core[r['prompt_id']].add(r['schluessel'])
    words[r['prompt_id']].append(r)
info={}
for pid,ws in words.items():
    for w in ws:
        if dbio.ist_geschuetzt(w['form'],w['ist_ziffer']): continue
        if w['herkunft']=='kern': info[w['wort_id']]='k'
        else: info[w['wort_id']]='r_ol' if w['schluessel'] in core[pid] else 'r_neu'
kinfo={r['kompressat_id']:(r['frage_id'],r['variante'],r['rho_ziel']) for r in c.execute("select kompressat_id,frage_id,variante,rho_ziel from v_bedingung where rho_ziel in (0.25,0.5) and variante!='basis'")}
cnt=defaultdict(lambda:[0,0,0,0])
for r in c.execute("select kompressat_id,wort_id,erhalten from wort_erhalt"):
    if r[0] not in kinfo: continue
    t=info.get(r[1])
    if t=='k': x=cnt[r[0]]; x[0]+=1; x[1]+=r[2]
    elif t=='r_neu': x=cnt[r[0]]; x[2]+=1; x[3]+=r[2]
d=defaultdict(dict); share=defaultdict(list)
for kid,(nk,ek,nr,er) in cnt.items():
    f,v,rho=kinfo[kid]
    if nr==0: continue
    s=((ek+.5)/(nk+1))/((er+.5)/(nr+1)); d[(v,rho)][f]=math.log(s)
print('D) S frei, Redundanz nur ueber Woerter ohne Entsprechung im Kern')
for rho in (0.25,0.5):
    for v in V:
        w=list(d[(v,rho)].values()); m,lo,hi,n,se=mki(w); print('  S',rho,v,'%.3f [%.3f; %.3f] n=%d'%(math.exp(m),math.exp(lo),math.exp(hi),n))
    for a,b in (('filler','semantic'),('semantic','instruction'),('instruction','demonstration'),('filler','instruction')):
        ts=[t for t in d[(a,rho)] if t in d[(b,rho)]]
        m,lo,hi,n,se=mki([d[(a,rho)][t]-d[(b,rho)][t] for t in ts])
        print('  K',rho,a,'>',b,'%.2f [%.2f; %.2f] p1=%.4f n=%d'%(math.exp(m),math.exp(lo),math.exp(hi),pone(m,se),n))
# Anteil neuer Redundanzwoerter
nn=defaultdict(list)
for pid,ws in words.items():
    fr=[w for w in ws if w['herkunft']!='kern' and not dbio.ist_geschuetzt(w['form'],w['ist_ziffer'])]
    if fr: nn[pid].append(sum(1 for w in fr if w['schluessel'] not in core[pid])/len(fr))
vp={r[0]:r[1] for r in c.execute("select prompt_id,variante from prompt")}
agg=defaultdict(list)
for pid,l in nn.items(): agg[vp[pid]].append(l[0])
for v,l in agg.items(): print('  Anteil freier Redundanzwoerter ohne Kernentsprechung',v,round(sum(l)/len(l),3))
# A2) Differenz der Abfaelle gegen filler (strukturelle Lesart von H3)
print('A2) Differenz der Abfaelle gegen filler, Prozentpunkte')
for rho in (0.25, 0.5):
    for v in ('semantic', 'instruction', 'demonstration'):
        d = [(acc[(v, rho)][t] - acc[(v, 0.0)][t]) - (acc[('filler', rho)][t] - acc[('filler', 0.0)][t]) for t in tasks]
        m, lo, hi, n, se = mki(d)
        print('  ', rho, v, '%+.1f [%+.1f; %+.1f]' % (100*m, 100*lo, 100*hi))
