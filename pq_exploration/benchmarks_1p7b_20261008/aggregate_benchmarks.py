"""Independently recompute saved predictions and emit the required report tables."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,math,re,string
ROOT=Path(__file__).resolve().parent
TASKS=['mnli','mrpc','qnli','qqp','rte','sst2','squadv2','leaderboard_mmlu_pro']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def qa_score(pred,gold):
    p,g=norm(pred),norm(gold);exact=int(p==g);p,g=p.split(),g.split()
    if not p or not g:return exact,float(p==g)
    shared=sum((collections.Counter(p)&collections.Counter(g)).values())
    return exact,(2*shared/(len(p)+len(g)) if shared else 0.0)
def verify(task,rec):
    rows=[]
    for chunk in rec['chunks']:
        assert sha(chunk['path'])==chunk['sha256']
        part=read(chunk['path']);assert part['protocol_sha256']==rec['protocol_sha256']
        assert part['entry_name']==rec['entry_name'] and part['evaluation_manifest_sha256']==rec['evaluation_manifest_sha256']
        rows.extend(part['records'])
    assert len(rows)==rec['questions'] and [r['index'] for r in rows]==list(range(len(rows)))
    m=rec['metrics']
    if task=='squadv2':
        values=[]
        for r in rows:
            assert abs(r['no_answer_probability']-math.exp(r['noanswer_logprob']))<1e-12
            gold=[g for g in r['answers']['text'] if norm(g)] or ['']
            scores=[qa_score(r['prediction_text'],g) for g in gold]
            values.append((max(s[0] for s in scores),max(s[1] for s in scores)))
        for i,key in enumerate(['exact','f1']):assert abs(m[key]-100*sum(v[i] for v in values)/len(rows))<1e-8
        assert m['total']==len(rows)
    else:
        for r in rows:
            assert all(math.isfinite(x) for x in r['scores'])
            assert r['prediction']==max(range(len(r['scores'])),key=lambda j:r['scores'][j])
        correct=sum(r['prediction']==r['answer'] for r in rows)
        assert m['correct']==correct and m['total']==len(rows) and abs(m['accuracy']-correct/len(rows))<1e-12
        if task in ['mrpc','qqp']:
            tp=sum(r['prediction']==r['answer']==1 for r in rows)
            fp=sum(r['prediction']==1 and r['answer']==0 for r in rows)
            fn=sum(r['prediction']==0 and r['answer']==1 for r in rows)
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
            assert abs(m['f1']-f1)<1e-12
    return m
def main(allow_partial):
    p=read(ROOT/'protocol.json');old=read(ROOT.parent/'study_1p7b_20261008_mixed21/round21_summary.json')
    completed={};missing=[]
    for e in p['entries']:
        name=e['name'];completed[name]={}
        for task in TASKS:
            path=ROOT/'results'/name/(task+'.json')
            if not path.exists():missing.append([name,task]);continue
            r=read(path);assert r['completed'] and r['protocol_sha256']==sha(ROOT/'protocol.json')
            assert r['requests_sha256']==p['requests'][task]['sha256']
            completed[name][task]=verify(task,r)
    assert allow_partial or not missing,missing
    report='Qwen3-1.7B：新增GLUE六项、SQuAD v2、MMLU-Pro\n\n'
    report+=('完整评测与独立汇总核验已完成。' if not missing else f'评测进行中：已完成{32-len(missing)}/32个模型×任务，未完成项不填入成绩。')+'\n\n'
    report+='模型：原始FP16与第21轮由文章监测集冻结的PQ/RQ/IB mixed。新增benchmark不参与模型选择或微调。\n\n'
    report+='| Method | Lookups | Additions | Multiplications | On-chip memory | PPL | MMLU | E_ref (mJ/token) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name in ['fp16','LUT-LLM','pq_mixed','rq_mixed','ib_mixed']:
        if name=='LUT-LLM':
            report+='| LUT-LLM资源参考 | 704.643 M | 726.663 M | 0.573 M | 1764.000 MiB | 未测 | 未测 | 2.969054 |\n';continue
        r=old['rows'][name];v=r['resources']
        report+=f"| {name} | {v['lookups']/1e6:.3f} M | {v['additions']/1e6:.3f} M | {v['multiplications']/1e6:.3f} M | {v['memory_MiB']:.3f} MiB | {r['test_ppl']:.6f} | {r['mmlu']*100:.3f}% | {v['score_mJ']:.6f} |\n"
    report+='\n以上PPL和普通MMLU复用第21轮结果，非本次新增测量。资源覆盖相同196个decoder linears；共同排除embedding/head、QK/AV、归一化和非线性。量化模型额外516292次倒数/token已计入算术门槛。E_ref=(3.8L+0.4A+1.3M)/1e9是7nm逻辑代理，非FPGA实测能耗。\n\n'
    report+='| Method | MNLI acc | MRPC acc / F1 | QNLI acc | QQP acc / F1 | RTE acc | SST-2 acc | SQuAD v2 EM / F1 | MMLU-Pro acc |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for name in ['fp16','LUT-LLM','pq_mixed','rq_mixed','ib_mixed']:
        values=[]
        for task in TASKS:
            m=completed.get(name,{}).get(task)
            if not m:values.append('未测' if name=='LUT-LLM' else '待完成');continue
            if task=='squadv2':values.append(f"{m['exact']:.3f} / {m['f1']:.3f}")
            elif task in ['mrpc','qqp']:values.append(f"{m['accuracy']*100:.3f} / {m['f1']*100:.3f}")
            else:values.append(f"{m['accuracy']*100:.3f}")
        report+='| '+name+' | '+' | '.join(values)+' |\n'
    report+='\n新增成绩均以百分制展示。GLUE只覆盖论文列出的六项，逐项报告，不命名为完整GLUE总分。MNLI使用matched验证集。GLUE为0-shot；MMLU-Pro为lm-eval leaderboard_mmlu_pro的5-shot直接选项似然，区别于普通MMLU及生成式CoT MMLU-Pro。SQuAD v2为0-shot标准原始提示词，贪心生成，换行/EOS停止，最多256token；主EM/F1的无答案概率阈值预先固定为1.0，不报告在评测标签上搜索出的best分数。所有任务不使用chat模板。\n\n'
    report+='LUT-LLM论文表III参考（与本协议未证明一致，不并入我们的同协议准确率比较）：\n| Source | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuAD v2 | MMLU-Pro |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n| 论文FP16 | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 72.8 | 33.1 |\n| 论文LUT-LLM最终量化 | 86.9 | 82.8 | 90.4 | 89.5 | 76.5 | 90.6 | 69.7 | 30.8 |\n\n'
    report+='论文来源：https://arxiv.org/html/2511.06174v2#S5.T3\n评测配置来源：lm-evaluation-harness 0.4.12的glue、squadv2、leaderboard/mmlu_pro任务。固定版本、提示词、数据SHA、模型SHA与完整题数详见protocol.json。\n未进行语义近重复或预训练去污染；此前WikiText/MMLU反复查看的限制仍然成立。\n'
    (ROOT/'BENCHMARK_REPORT_ZH.txt').write_text(report,encoding='utf-8')
    summary=dict(completed=not missing,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
        verified_task_results=32-len(missing),missing=missing,results=completed,independent_aggregate_recomputation=True)
    (ROOT/'benchmark_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['results','missing']}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');a=ap.parse_args();main(a.allow_partial)
