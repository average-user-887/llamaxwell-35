#!/usr/bin/env python3
"""Add raw numerical reconciliation from independently revalidated R2 receipts."""
import argparse
from pathlib import Path

HERE=Path(__file__).resolve().parent


def prime(owner, upstream, receipts, pair_paths, controller):
    import importlib.util
    import hashlib
    import json

    def load(path,name):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def read(path):
        return json.loads(path.read_text())

    def write(path,result):
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    digest=hashlib.sha256((upstream/'SHA256SUMS').read_bytes()).hexdigest()
    if digest!='b1c6b32156ddf7c5e2507a1669cd0d7954b2abeb8cb24a258f43cf20071c0ab1':
        raise ValueError('published R1 pair manifest differs')
    for line in (upstream/'SHA256SUMS').read_text().splitlines():
        sha,file=line.split(maxsplit=1)
        if Path(file).name!=file or hashlib.sha256((upstream/file).read_bytes()).hexdigest()!=sha:
            raise ValueError('published R1 pair artifact differs')
    base=load(upstream/'pair-evidence.py','r1_pair_accounting')
    exporter=load(HERE/'export-model.py','r2_receipt_exporter')
    verifier=load(HERE/'replay.py','r2_receipt_replay')
    model=exporter.pin_owner(owner.resolve())
    by_stage={stage:Path(path).resolve() for stage,path in receipts}
    if len(receipts)!=len(by_stage) or set(by_stage) not in ({'shadow','native'},{'shadow','native','serial'}):
        raise ValueError('supply exact shadow and native attempts, with one optional serial control')
    pair=None if not pair_paths else tuple(i.resolve() for i in pair_paths)
    if pair and (pair!=(by_stage.get('serial'),by_stage['native']) or not controller):
        raise ValueError('paired export requires the selected serial and native plus controller comparison')
    if not pair and ('serial' in by_stage or controller):
        raise ValueError('include the strict comparison with a terminal serial control')
    evidence=exporter.export(owner.resolve(),receipts,pair)
    if not all(one['disposition']=='validated' for one in evidence['attempts']):
        raise ValueError('R2 terminal replay refused a selected arm')
    verifier.verify_model(evidence,read(HERE/'source-pins.json'))
    result={'schema':1,'source_freeze_sha256':evidence['source_freeze_sha256'],
            'descending_scope':['active native rate','same-build pair when present'],
            'shadow_raw_logits_retained':True,'arms':{},'pair':None}
    for arm in evidence['attempts']:
        stage=arm['stage']
        result['arms'][stage]=base.phases_from_receipt(by_stage[stage],stage,arm['model'])
    if pair:
        serial,native=pair
        if model.compare(serial,native)!=read(controller):
            raise ValueError('independent R2 replay differs from the controller comparison')
        result['pair']={}
        for phase in base.PHASES:
            files=[read(folder/f'{phase}.json') for folder in pair]
            left,right=[v['token_ids'] for v in files]
            first=next((i for i,(a,b) in enumerate(zip(left,right),1) if a!=b),None)
            result['pair'][phase]={'first_cross_mode_difference_ordinal':first,'cross_mode_full_text_equal':files[0]['text']==files[1]['text'],
                                       'native_minus_serial_client_wall_s':files[1]['wall_s']-files[0]['wall_s']}
        actual={k:v for k,v in evidence.items() if k!='attempts'}
        actual['attempts']=[v for v in evidence['attempts'] if v['stage']!='shadow']
        base.rederive(actual,result)
    r1=model.parent.module(owner.resolve().parent/'break100-compact-model-r1/model.py','r1_cli_reference')
    scoped={}
    for stage in model.STAGES:
        env,args=model.settings(stage)
        oldenv,oldargs=r1.settings(stage)
        assert args==oldargs
        assert env==['GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=1' if i=='GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=0' else i for i in oldenv]
        spec=args[args.index('--spec-type'):]
        scoped[stage]={'command_arguments_equal_to_r1':True,'speculation_argument_suffix':spec,
                        'down4_feature_value':1,'compact_feature_value':'shadow' if stage=='shadow' else '1'}
    write(HERE/'typed-cli.json',{'same_mapped_runtime_as_r1':model.profile==r1.profile,'only_feature_delta':'GGML_CUDA_OCCAMY_MOE_Q2_DOWN4 from 0 to 1',
                                 'request_context_tokens':80000,'slots':1,'effective_experts':8,'cap_seconds':430,'stages':scoped})
    write(HERE/'results.json',evidence)
    write(HERE/'reconciliation.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--owner-card',type=Path,required=True)
    p.add_argument('--r1-pair-source',type=Path,required=True)
    p.add_argument('--receipt',nargs=2,action='append',required=True)
    p.add_argument('--pair',nargs=2,type=Path)
    p.add_argument('--controller-pair',type=Path)
    a=p.parse_args()
    prime(a.owner_card,a.r1_pair_source,a.receipt,a.pair,a.controller_pair)
