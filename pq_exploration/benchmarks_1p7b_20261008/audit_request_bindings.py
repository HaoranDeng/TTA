"""Check saved predictions against the independently frozen request order and gold labels."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
from pathlib import Path
import datetime,hashlib,json
import torch
torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    protocol=read(ROOT/'protocol.json');checked=[];predictions=0
    for task,info in protocol['requests'].items():
        paths=[ROOT/'results'/e['name']/(task+'.json') for e in protocol['entries']]
        paths=[p for p in paths if p.exists()]
        if not paths:continue
        assert sha(info['path'])==info['sha256']
        rows=torch.load(info['path'],weights_only=True)['rows']
        for path in paths:
            result=read(path);assert result['completed'] and result['requests_sha256']==info['sha256']
            count=0
            for chunk in result['chunks']:
                assert sha(chunk['path'])==chunk['sha256']
                for rec in read(chunk['path'])['records']:
                    row=rows[count];assert rec['index']==count==row['index']
                    assert rec['prompt_sha256']==row['prompt_sha256']
                    if task=='squadv2':assert rec['id']==row['id'] and rec['answers']==row['answers']
                    else:
                        assert rec['answer']==row['answer'] and len(rec['scores'])==len(row['choice_ids'])
                        if task=='leaderboard_mmlu_pro':assert rec['category']==row['category']
                    count+=1
            assert count==len(rows)==result['questions']
            checked.append(dict(entry=result['entry_name'],task=task,questions=count,result_sha256=sha(path)))
            predictions+=count
    proof=dict(audit_completed=True,full_benchmark_complete=len(checked)==32,verified_complete_task_results=len(checked),
        verified_predictions=predictions,protocol_sha256=sha(ROOT/'protocol.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checked)
    target=ROOT/'request_binding_audit.json';temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(proof,indent=2)+'\n');temp.replace(target)
    print(json.dumps({k:v for k,v in proof.items() if k!='checks'}))
if __name__=='__main__':main()
