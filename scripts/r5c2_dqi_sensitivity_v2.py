"""Isolated physical R5-C2 CLI. Formal stages require a matching freeze."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cvn_rl.experiments.r5c2_v2.protocol import resolve,save_resolution,bundle_root,read_json,write_json,check_protected,file_hash,project_path

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['acknowledge-workspace','resolve','check-physics','validate','prepare-data','pilot-freshness','train-predictors','pilot-runtime','freeze','run-rl','summarize','plot','verify','_task'])
    parser.add_argument('--config',default='configs/r2_r05c02_dqi_sensitivity_v2.json'); parser.add_argument('--protocol')
    parser.add_argument('--stage',choices=['pilot','formal'],default='formal'); parser.add_argument('--targets'); parser.add_argument('--conditions'); parser.add_argument('--seeds')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--skip-full-probe',action='store_true'); parser.add_argument('--full-probe',action='store_true')
    parser.add_argument('--phase',choices=['protocol','runs'],default='protocol'); parser.add_argument('--condition'); parser.add_argument('--seed',type=int)
    parser.add_argument('--workspace-stable',action='store_true')
    a=parser.parse_args()
    if a.command=='acknowledge-workspace':
        from cvn_rl.experiments.r5c2_v2.protocol import acknowledge_workspace
        print(json.dumps(acknowledge_workspace(a.workspace_stable),ensure_ascii=False,indent=2)); return
    if a.command=='_task':
        p=read_json(a.protocol); check_protected()
        if any(file_hash(project_path(n))!=h for n,h in p['source_hashes'].items()): raise ValueError('task code differs from resolved protocol')
        from cvn_rl.experiments.r5c2_v2.runner import run_task
        run_task(p,a.condition,a.seed,a.stage,a.full_probe); return
    p=resolve(a.config); root=save_resolution(p)
    if a.command=='resolve': result={'protocol_hash':p['protocol_hash'],'root':str(root),'conditions':19,'tasks':(len(p['conditions'])+1)*len(p['entry']['rl_seeds'])}
    elif a.command=='check-physics':
        from cvn_rl.experiments.r5c2_v2.physics import generate
        b=generate(p['entry'],91000,1000,10,'physical_check'); result={'passed':True,'physical_hash':b.physical_hash(),'rate_mbps_min':float(b.arrays['rate_mbps'].min()),'rate_mbps_max':float(b.arrays['rate_mbps'].max()),'rho_min':float(b.arrays['rho'].min()),'rho_max':float(b.arrays['rho'].max())}
        write_json(root/'physical_check.json',result)
    elif a.command=='validate':
        from cvn_rl.experiments.r5c2_v2.validation import validate
        result=validate(p)
    elif a.command=='prepare-data':
        from cvn_rl.experiments.r5c2_v2.data import prepare
        r=prepare(p,a.resume); result={'rows':sum(x['samples'] for x in r['entries']),'targets':len(r['targets']),'wall_seconds':r['wall_seconds']}
    elif a.command=='pilot-freshness':
        from cvn_rl.experiments.r5c2_v2.pilot import pilot_freshness
        result=pilot_freshness(p)
    elif a.command=='train-predictors':
        from cvn_rl.experiments.r5c2_v2.training import train_all
        result=train_all(p,a.stage,a.targets.split(',') if a.targets else None,a.resume)
    elif a.command=='pilot-runtime':
        from cvn_rl.experiments.r5c2_v2.pilot import pilot_runtime
        result=pilot_runtime(p,[int(s) for s in a.seeds.split(',')] if a.seeds else None,not a.skip_full_probe,a.resume)
    elif a.command=='freeze':
        from cvn_rl.experiments.r5c2_v2.execution import freeze
        r=freeze(p,a.phase); result={'phase':a.phase,'tasks':len(r.get('tasks',[])),'protocol_hash':p['protocol_hash']}
    elif a.command=='run-rl':
        from cvn_rl.experiments.r5c2_v2.execution import run_cases
        r=run_cases(p,a.stage,a.conditions.split(',') if a.conditions else None,[int(s) for s in a.seeds.split(',')] if a.seeds else None,a.resume)
        result={'completed':sum(x['status']=='completed' for x in r.values())}
    else:
        from cvn_rl.experiments.r5c2_v2.analysis import summarize,plot,verify
        result={'summarize':summarize,'plot':plot,'verify':verify}[a.command](p)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
