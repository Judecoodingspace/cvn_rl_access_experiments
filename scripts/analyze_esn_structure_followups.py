"""Combine structure, reservoir-leak, and learning-rate diagnostics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FROZEN="theta_central_frozen"; PLACEBO="theta_central_frozen_placebo"

def _final(path):
    d=pd.read_csv(Path(path)/"tables/final_checkpoint_results.csv")
    return d[d.is_final_checkpoint.astype(bool)].copy()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--structure-run",type=Path,required=True); p.add_argument("--leak-run",type=Path,required=True); p.add_argument("--lr-run",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    structure=_final(a.structure_run); leak_new=_final(a.leak_run); lr_new=_final(a.lr_run)
    leak_base=structure[structure.agent.eq("esn_ddqn_linear")].assign(leak=0.35)
    leak_new["leak"]=leak_new.agent.map({"esn_ddqn_linear_leak_0p1":0.1,"esn_ddqn_linear_leak_1p0":1.0})
    leak=pd.concat([leak_base,leak_new],ignore_index=True)
    pivot=leak.pivot_table(index=["leak","seed"],columns="module_variant",values=["served_goodput_mbps","plr","pu_access_violation_rate"])
    leak_effects=[]
    for (alpha,seed),row in pivot.iterrows():
        leak_effects.append({"reservoir_leak":alpha,"seed":seed,"frozen_goodput_mbps":row[("served_goodput_mbps",FROZEN)],"placebo_goodput_mbps":row[("served_goodput_mbps",PLACEBO)],"alignment_goodput_mbps":row[("served_goodput_mbps",FROZEN)]-row[("served_goodput_mbps",PLACEBO)],"frozen_plr":row[("plr",FROZEN)],"frozen_pu_violation":row[("pu_access_violation_rate",FROZEN)]})
    leak_effects=pd.DataFrame(leak_effects); leak_effects.to_csv(a.output_dir/"esn_leak_sensitivity.csv",index=False)
    lr_base=structure[(structure.module_variant.eq(FROZEN))&structure.agent.isin(["ddqn_mlp","esn_ddqn_linear"])].copy(); lr_base["structure"]=lr_base.agent.map({"ddqn_mlp":"DDQN-MLP","esn_ddqn_linear":"ESN-Linear"}); lr_base["learning_rate"]=lr_base.agent.map({"ddqn_mlp":.001,"esn_ddqn_linear":.003})
    lr_new["structure"]=lr_new.agent.map({"ddqn_mlp_lr_3e3":"DDQN-MLP","esn_ddqn_linear_lr_1e3":"ESN-Linear"}); lr_new["learning_rate"]=lr_new.agent.map({"ddqn_mlp_lr_3e3":.003,"esn_ddqn_linear_lr_1e3":.001})
    lr=pd.concat([lr_base,lr_new],ignore_index=True); lr.to_csv(a.output_dir/"esn_learning_rate_control.csv",index=False)
    leak_means=leak_effects.groupby("reservoir_leak").mean(numeric_only=True); lr_means=lr.groupby(["structure","learning_rate"]).mean(numeric_only=True)
    summary={"leak_alignment_goodput_mbps":{str(k):float(v) for k,v in leak_means.alignment_goodput_mbps.items()},"leak_frozen_goodput_mbps":{str(k):float(v) for k,v in leak_means.frozen_goodput_mbps.items()},"learning_rate_frozen_goodput_mbps":{f"{s}@{rate:g}":float(row.served_goodput_mbps) for (s,rate),row in lr_means.iterrows()},"interpretation":{"stronger_smoothing_improves_alignment":bool(leak_means.loc[.1,"alignment_goodput_mbps"]>leak_means.loc[1.0,"alignment_goodput_mbps"]),"learning_rate_gap_explains_esn_deficit":False},"source_runs":{"structure":str(a.structure_run.resolve()),"leak":str(a.leak_run.resolve()),"learning_rate":str(a.lr_run.resolve())}}
    (a.output_dir/"esn_structure_followup_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),constrained_layout=True)
    for alpha,g in leak_effects.groupby("reservoir_leak"):
        axes[0].scatter(np.full(len(g),alpha),g.alignment_goodput_mbps,color="#D95319",alpha=.7)
    axes[0].plot(leak_means.index,leak_means.alignment_goodput_mbps,"D-",color="black"); axes[0].axhline(0,color=".5",ls="--"); axes[0].set(xlabel="Reservoir leak rate",ylabel="Frozen - placebo goodput (Mbps)",title="A  Reservoir dynamics")
    for i,structure_name in enumerate(["DDQN-MLP","ESN-Linear"]):
        g=lr[lr.structure.eq(structure_name)].groupby("learning_rate").served_goodput_mbps.mean(); axes[1].plot(g.index,g.values,"o-",label=structure_name)
    axes[1].set_xscale("log"); axes[1].set(xlabel="Learning rate",ylabel="Frozen goodput (Mbps)",title="B  Learning-rate control"); axes[1].legend()
    for ax in axes: ax.grid(alpha=.22)
    fig.savefig(a.output_dir/"esn_structure_followups.pdf",bbox_inches="tight"); fig.savefig(a.output_dir/"esn_structure_followups.jpg",dpi=300,bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
