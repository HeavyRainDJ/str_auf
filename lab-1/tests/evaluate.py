"""Run unmodified submissions and compare controlled outputs to independent oracle."""
from pathlib import Path
import argparse, json, subprocess, sys, time, os, hashlib, traceback
import numpy as np
from scipy.sparse import load_npz, isspmatrix_csr
import spacy
from oracle import expected

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def run(model, version, full=False):
    code = ROOT/'models'/model/version/'solution.py'
    target = ROOT/'results'/model/version
    target.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONHASHSEED='0')
    report = {'model':model, 'version':version, 'python':sys.version, 'code_sha256':hashlib.sha256(code.read_bytes()).hexdigest(), 'tests':[]}
    cp = subprocess.run([sys.executable, '-m', 'py_compile', str(code)], capture_output=True, text=True, encoding='utf-8', env=env)
    report['compilation'] = {'passed':cp.returncode==0, 'stderr':cp.stderr}
    if cp.returncode:
        report['tests'].append({'case':'compilation','passed':False,'error':cp.stderr})
    else:
        nlp = spacy.load('ru_core_news_sm', disable=['ner'])
        cases = read(ROOT/'tests'/'expected_texts.json')
        cases['corrupt'] = None
        if full: cases['full_corpus'] = None
        for case, texts in cases.items():
            source = ROOT.parent/'тексты для тестов' if case=='full_corpus' else ROOT/'tests'/'fixtures'/case
            out = target/case
            command = [sys.executable, str(code), '--input',str(source), '--output',str(out)]
            start = time.perf_counter()
            try:
                cp = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=600)
                item = {'case':case,'command':command,'seconds':round(time.perf_counter()-start,3),'exit_code':cp.returncode,'checks':[]}
                (target/(case+'.log')).write_text(cp.stdout+'\nSTDERR:\n'+cp.stderr,encoding='utf-8')
                if case=='corrupt':
                    error_logs = list(out.glob('*error*.json')) if out.exists() else []
                    meaningful = any('broken.docx' in p.read_text(encoding='utf-8') for p in error_logs)
                    item['passed'] = bool(meaningful or (cp.returncode!=0 and 'broken.docx' in cp.stdout+cp.stderr))
                    item['checks'].append({'name':'corrupt_file_reported', 'passed':item['passed']})
                elif cp.returncode:
                    item['passed']=False; item['error']=cp.stderr[-2500:]
                else:
                    oracle = expected(texts, nlp) if texts is not None else None
                    prefixes = ['word_char','doc_tfidf','word_syntax','doc_syntax']
                    adapter_path = code.parent/'adapter.json'
                    adapter = read(adapter_path) if adapter_path.exists() else {}
                    for pref in prefixes:
                        try:
                            spec = adapter.get(pref, {'matrix':pref+'.npz','rows':pref+'_rows.json','features':pref+'_features.json'})
                            m = load_npz(out/spec['matrix']); rows=read(out/spec['rows']); features=read(out/spec['features'])
                            checks = {'csr':isspmatrix_csr(m), 'shape':m.shape==(len(rows),len(features)), 'finite_nonnegative':bool(np.isfinite(m.data).all() and (m.data>=0).all()),'rows_sorted_unique':rows==sorted(set(rows)), 'features_sorted_unique':features==sorted(set(features))}
                            if oracle is not None:
                                em, er, ef = oracle[pref]
                                checks['rows_exact']=rows==er; checks['features_exact']=features==ef
                                checks['values_exact']=False
                                if rows==er and features==ef and m.shape==em.shape:
                                    diff=m-em; checks['values_exact']=bool(not diff.nnz or np.max(np.abs(diff.data))<1e-6)
                            elif pref.startswith('doc_'):
                                wanted=[v['path'] for v in read(ROOT/'tests'/'corpus_manifest.json')]
                                checks['all_50_documents']=rows==wanted
                            if pref=='doc_tfidf':
                                norms=np.asarray(m.multiply(m).sum(axis=1)).ravel()
                                checks['l2_normalized']=bool(np.all(np.isclose(norms,1,atol=1e-6)|np.isclose(norms,0,atol=1e-6)))
                            elif m.nnz:
                                checks['integer_counts']=bool(np.allclose(m.data,np.round(m.data)))
                            item['checks'].append({'name':pref,'passed':all(checks.values()),'shape':list(m.shape),'nnz':m.nnz,'details':checks})
                        except Exception as exc:
                            item['checks'].append({'name':pref,'passed':False,'error':str(exc)})
                    item['checks'].append({'name':'metadata','passed':(out/'metadata.json').exists()})
                    item['passed']=all(c['passed'] for c in item['checks'])
                report['tests'].append(item)
                print(model, version, case, 'PASS' if item['passed'] else 'FAIL', item['seconds'], flush=True)
            except Exception:
                report['tests'].append({'case':case,'passed':False,'error':traceback.format_exc()})
    report['passed']=all(t['passed'] for t in report['tests'])
    (target/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report:',target/'evaluation.json',flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('version'); p.add_argument('--full',action='store_true'); a=p.parse_args(); run(a.model,a.version,a.full)
