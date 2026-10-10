"""Audit and plot the P3 prediction effect across discrete controllers."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS={"theta_oracle","theta_central_markov","theta_central_frozen","theta_central_frozen_placebo","theta_neutral"}
LABELS={
    "ddqn_no_esn":"DDQN",
    "proposed_esn_ddqn":"ESN-DDQN",
    "lstm_ddqn":"LSTM-DDQN",
    "sac_discrete":"Discrete SAC",
    "ppo_discrete":"PPO",
}
CONTROLLER_ORDER = list(LABELS.values())
MODEL_SHA="c0d645858adf54bdcd04b9c0956629a9b11968a60013ff7d21ec5d8381cb7732"

def support_level(passes):
    n=sum(bool(v) for v in passes.values())
    return "strong_support" if passes and n==len(passes) else "partial_support" if n>=2 else "not_supported"

def read_run(path,agent):
    path=Path(path); df=pd.read_csv(path/"tables/final_checkpoint_results.csv")
    df=df.loc[df.is_final_checkpoint.astype(bool)].copy()
    if set(df.agent)!={agent}: raise ValueError(f"{path}: agent mismatch")
    if len(df)!=15 or set(df.module_variant)!=CONDITIONS: raise ValueError(f"{path}: expected 5 conditions x 3 seeds")
    counts=df.groupby("module_variant").seed.agg(["count","nunique"])
    if not ((counts["count"]==3)&(counts["nunique"]==3)).all(): raise ValueError(f"{path}: incomplete/repeated seeds")
    if df.duplicated(["agent","scenario","seed","module_variant"]).any(): raise ValueError(f"{path}: duplicate key")
    if not df.evaluation_parameters_frozen.astype(bool).all(): raise ValueError(f"{path}: evaluation not frozen")
    fp=df[df.module_variant.isin(["theta_central_frozen","theta_central_frozen_placebo"])]
    if set(fp.prediction_model_hash)!={MODEL_SHA}: raise ValueError(f"{path}: model SHA mismatch")
    gate=json.loads((path/"analysis/e2_r1_gate.json").read_text(encoding="utf-8")); manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    df["controller"]=LABELS[agent]
    df["backbone"]=df["controller"]  # Backward-compatible CSV column.
    df["source_run"]=str(path.resolve())
    return df,gate,manifest

def validate_and_merge(runs):
    frames=[]; gates={}; manifests={}
    for agent,path in runs.items():
        df,gate,manifest=read_run(path,agent); label=LABELS[agent]
        frames.append(df); gates[label]=gate; manifests[label]=manifest
    merged=pd.concat(frames,ignore_index=True)
    for seed,group in merged.groupby("seed"):
        if group.evaluation_physical_trace_hash.nunique()!=1: raise ValueError(f"seed {seed}: physical trace mismatch")
    return merged,gates,manifests

def paired_effects(df):
    specs={"frozen_minus_placebo_goodput":("served_goodput_mbps","theta_central_frozen","theta_central_frozen_placebo"),"frozen_minus_neutral_goodput":("served_goodput_mbps","theta_central_frozen","theta_neutral"),"frozen_minus_placebo_pu_violation":("pu_access_violation_rate","theta_central_frozen","theta_central_frozen_placebo")}; rows=[]
    for controller,group in df.groupby("controller",sort=False):
        ix=group.set_index(["seed","module_variant"])
        for effect,(metric,left,right) in specs.items():
            for seed in sorted(group.seed.unique()): rows.append({"controller":controller,"backbone":controller,"seed":int(seed),"effect":effect,"value":float(ix.loc[(seed,left),metric]-ix.loc[(seed,right),metric]),"left_condition":left,"right_condition":right})
    return pd.DataFrame(rows)

def summarize(effects):
    rows=[]; rng=np.random.default_rng(20260703)
    for (controller,effect),group in effects.groupby(["controller","effect"],sort=False):
        values=group.value.to_numpy(float); means=rng.choice(values,(10_000,len(values)),replace=True).mean(1); low,high=np.quantile(means,[.025,.975])
        rows.append({"controller":controller,"backbone":controller,"effect":effect,"mean":values.mean(),"ci95_low":low,"ci95_high":high})
    return pd.DataFrame(rows)

def plot_effects(effects,target):
    order=[name for name in CONTROLLER_ORDER if name in set(effects.controller)]
    palette={
        "DDQN":"#2878B5", "ESN-DDQN":"#D95319",
        "LSTM-DDQN":"#7E57C2", "Discrete SAC":"#E9C46A",
        "PPO":"#2A9D8F",
    }
    panels=[("frozen_minus_placebo_goodput","A  Frozen − placebo goodput","Goodput difference (Mbps)"),("frozen_minus_neutral_goodput","B  Frozen − neutral goodput","Goodput difference (Mbps)"),("frozen_minus_placebo_pu_violation","C  Frozen − placebo violation","PU violation difference")]
    summary=summarize(effects); fig,axes=plt.subplots(1,3,figsize=(10.2,3.25),constrained_layout=True)
    for axis,(effect,title,ylabel) in zip(axes,panels):
        for x,controller in enumerate(order):
            color=palette[controller]
            values=effects[(effects.controller==controller)&(effects.effect==effect)].value.to_numpy(); axis.scatter(np.full(len(values),x),values,s=25,color=color,alpha=.75,zorder=3)
            row=summary[(summary.controller==controller)&(summary.effect==effect)].iloc[0]
            axis.errorbar(x,row["mean"],yerr=[[row["mean"]-row.ci95_low],[row.ci95_high-row["mean"]]],fmt="D",color="black",capsize=4,markersize=5,zorder=4)
        axis.axhline(0,color=".45",ls="--",lw=.9); axis.set_xticks(range(len(order)),order,rotation=18,ha="right"); axis.set_title(title,fontsize=10); axis.set_ylabel(ylabel); axis.grid(axis="y",alpha=.22)
    fig.savefig(Path(target).with_suffix(".pdf"),bbox_inches="tight"); fig.savefig(Path(target).with_suffix(".jpg"),dpi=300,bbox_inches="tight"); plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    for name in ["ddqn","esn","ppo"]: p.add_argument(f"--{name}-run",type=Path)
    p.add_argument(
        "--run", action="append", default=[], metavar="AGENT_ID=RUN_DIR",
        help="Repeatable controller run; supports lstm_ddqn and sac_discrete.",
    )
    p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    runs={}
    legacy={
        "ddqn_no_esn":a.ddqn_run,
        "proposed_esn_ddqn":a.esn_run,
        "ppo_discrete":a.ppo_run,
    }
    runs.update({agent:path for agent,path in legacy.items() if path is not None})
    for item in a.run:
        if "=" not in item:
            p.error(f"--run must be AGENT_ID=RUN_DIR, got {item!r}")
        agent,path=item.split("=",1)
        if agent not in LABELS:
            p.error(f"unsupported controller {agent!r}; choose from {sorted(LABELS)}")
        runs[agent]=Path(path)
    if not runs:
        p.error("provide legacy run flags or at least one --run AGENT_ID=RUN_DIR")
    merged,gates,manifests=validate_and_merge(runs); effects=paired_effects(merged); summary=summarize(effects); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    merged.to_csv(out/"prediction_cross_backbone_final_rows.csv",index=False)
    effects.merge(summary,on=["controller","backbone","effect"]).to_csv(out/"prediction_cross_backbone_paired_effects.csv",index=False)
    passes={name:bool(gate.get("passed")) for name,gate in gates.items()}; payload={"passed_by_backbone":passes,"support_level":support_level(passes),"individual_gates":gates}
    (out/"prediction_cross_backbone_gate.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    source={"source_runs":{LABELS[x]:str(Path(y).resolve()) for x,y in runs.items()},"resolved_config_hashes":{n:g.get("resolved_config_hash") for n,g in gates.items()},"model_sha256":MODEL_SHA,"source_manifest_sha256":{n:hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest() for n,m in manifests.items()}}
    (out/"prediction_cross_backbone_manifest.json").write_text(json.dumps(source,indent=2),encoding="utf-8")
    for name,gate in gates.items(): (out/f"prediction_cross_backbone_{name.lower().replace('-','_')}_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
    plot_effects(effects,out/"prediction_cross_backbone_effects")
    print(json.dumps({"output_dir":str(out),"support_level":payload["support_level"],"passes":passes},indent=2))

if __name__=="__main__": main()
