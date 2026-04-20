"""
tools/flexoelectric_diagnostic.py
===================================
Flexoelectric coupling coefficient measurement — Series 1 and Series 2.

Usage:
    python tools/flexoelectric_diagnostic.py <session_output.txt>
    python tools/flexoelectric_diagnostic.py <session_output.txt> --series2

PHYSICAL MODEL:
    P = mu * G
    P = polarization  (top pocket score)
    G = gradient      (mean |delta_ns| between adjacent pkt=0 words)
    mu = flexoelectric coupling coefficient

Series 1: constant question, varying gradient — measures mu
Series 2: same words, different order — tests tensor vs scalar coupling
"""

import sys, re, json, math
from pathlib import Path

PROMPT_REGISTRY = {
    0: {"label":"S1_FLAT",        "series":1,"design":"flat",        "target":0.3},
    1: {"label":"S1_MEDIUM",      "series":1,"design":"medium",      "target":1.8},
    2: {"label":"S1_STEEP",       "series":1,"design":"steep",       "target":4.8},
    3: {"label":"S1_ALTERNATING", "series":1,"design":"alternating", "target":3.3},
    4: {"label":"S2_ASCENDING",   "series":2,"design":"ascending",   "target":3.5},
    5: {"label":"S2_DESCENDING",  "series":2,"design":"descending",  "target":3.5},
    6: {"label":"S2_ALTERNATING", "series":2,"design":"alternating", "target":3.5},
}

def parse_session(text):
    blocks  = re.split(r'\n  > ', text)
    results = []
    idx     = 0
    for block in blocks[1:]:
        lines  = block.split('\n')
        prompt = lines[0].strip()
        if prompt.lower() in ('status','quit','groups','vocab','diag','carry',''):
            continue
        meta = PROMPT_REGISTRY.get(idx, {"label":f"P{idx}","series":0,
                                         "design":"unknown","target":0.0})
        d = {
            "idx":idx,"label":meta["label"],"series":meta["series"],
            "design":meta["design"],"target":meta["target"],
            "prompt":prompt[:100],"pkt0":[],"pkt1":[],
            "res":None,"stress":None,"polarity":None,
            "score":None,"output":None,"locked":False,
        }
        for line in lines:
            m = re.match(
                r'\s+(\S+)\s+\| t=([+-][\d.]+) \| grp=\s*(-?\d+) \| net=([+-][\d.]+) \| pkt=(\d)',
                line)
            if m:
                word,t,grp,ns,pkt = m.groups()
                e = {"word":word.rstrip('.,?!;:'),"ns":float(ns),"pkt":int(pkt)}
                if int(pkt)==0: d["pkt0"].append(e)
                else:           d["pkt1"].append(e)
        m = re.search(r'res=([\d.]+)', block)
        if m: d["res"] = float(m.group(1))
        m = re.search(r'Field stress\s+:\s+([\d.]+)', block)
        if m: d["stress"] = float(m.group(1))
        m = re.search(r'polarity ([+-][\d.]+)', block)
        if m: d["polarity"] = float(m.group(1))
        m = re.search(r'pocket scores: \S+\([^,]+,([\d.]+)\)', block)
        if m: d["score"] = float(m.group(1))
        m = re.search(r'Geometric Output.*?\n  (.*?)(?:\n|$)', block, re.DOTALL)
        if m:
            d["output"] = re.sub(r'\[.*?\]','',m.group(1)).strip()
            d["locked"] = "parity locked" in block
        results.append(d)
        idx += 1
    return results

def metrics(words):
    if not words:
        return {"mean":0,"var":0,"grad":0,"n":0,"seq":[]}
    ns   = [abs(w["ns"]) for w in words]
    mean = sum(ns)/len(ns)
    var  = sum((x-mean)**2 for x in ns)/len(ns)
    grad = sum(abs(ns[i+1]-ns[i]) for i in range(len(ns)-1))/max(len(ns)-1,1)
    return {"mean":round(mean,4),"var":round(var,4),"grad":round(grad,4),
            "n":len(ns),"seq":[round(x,3) for x in ns]}

def pearson(x,y):
    n=len(x)
    if n<2: return 0.0
    mx,my=sum(x)/n,sum(y)/n
    num=sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den=math.sqrt(sum((xi-mx)**2 for xi in x)*sum((yi-my)**2 for yi in y))
    return round(num/den,4) if den>0 else 0.0

def fit_linear(x,y):
    n=len(x)
    if n<2: return (0,0,0)
    mx,my=sum(x)/n,sum(y)/n
    num=sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den=sum((xi-mx)**2 for xi in x)
    s=num/den if den>0 else 0
    return (round(s,4),round(my-s*mx,4),pearson(x,y))

def analyze(results, series2=False):
    print("FLEXOELECTRIC COUPLING COEFFICIENT EXPERIMENT")
    print("="*65)
    print()

    rows=[]
    for r in results:
        if not r["pkt0"]: continue
        m=metrics(r["pkt0"])
        rows.append({**r,"m":m,"G":m["grad"],"mean_ns":m["mean"]})

    if not rows:
        print("No data."); return

    for r in rows:
        m=r["m"]
        print(f"[{r['label']}]  series={r['series']}  design={r['design']}")
        print(f"  Prompt:    {r['prompt'][:68]}")
        print(f"  NS seq:    {m['seq']}")
        print(f"  Mean ns:   {m['mean']:.4f}  Gradient: {m['grad']:.4f}  (target {r['target']:.1f})")
        print(f"  Score(P):  {r['score']}  Res: {r['res']}  Locked: {r['locked']}")
        print(f"  Output:    {r['output']}")
        print()

    # Series 1
    s1=[r for r in rows if r["series"]==1]
    if len(s1)>=3:
        print("─"*65)
        print("SERIES 1  —  P = mu * G")
        print()
        valid=[(r["G"],r["score"]) for r in s1 if r["score"] is not None]
        G,P=zip(*valid) if valid else ([],[])
        mu,b,rv=fit_linear(list(G),list(P))

        print(f"  {'Label':<18s} {'G actual':>10s} {'G target':>10s} {'P score':>10s}")
        print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10}")
        for r in s1:
            print(f"  {r['label']:<18s} {r['G']:>10.4f} {r['target']:>10.1f} {(r['score'] or 0):>10.4f}")
        print()
        print(f"  Fit:     P = {mu:.4f} * G  +  {b:.4f}")
        print(f"  r = {rv:.4f}   r² = {rv**2:.4f}  ({round(rv**2*100,1)}% variance explained)")
        q=("EXCELLENT" if abs(rv)>=0.95 else "GOOD" if abs(rv)>=0.85
           else "MODERATE" if abs(rv)>=0.70 else "WEAK")
        print(f"  Fit quality: {q}")
        print()

        # Saturation
        if len(valid)>=4:
            gs=sorted(valid)
            deltas=[(gs[i+1][1]-gs[i][1])/max(gs[i+1][0]-gs[i][0],0.001)
                    for i in range(len(gs)-1)]
            if deltas and deltas[-1]<deltas[0]*0.5:
                print(f"  SATURATION detected — diminishing returns at high gradient")
                print(f"  Slopes: {[round(d,3) for d in deltas]}")
            else:
                print(f"  No saturation — linear across tested range")
        print()
        print(f"  COUPLING COEFFICIENT:  mu = {mu:.4f}")
        print()

        phi=(1+math.sqrt(5))/2
        AD=0.016395102
        ls=3.6298
        exprs=[
            ("AD x LAYER_SCALE",   round(AD*ls,4)),
            ("1 / LAYER_SCALE",    round(1/ls,4)),
            ("AD x PHI^2",         round(AD*phi**2,4)),
            ("1 / (PHI x pi)",     round(1/(phi*math.pi),4)),
            ("sqrt(AD x PHI)",     round(math.sqrt(AD*phi),4)),
            ("AD x pi",            round(AD*math.pi,4)),
            ("1 / pi^2",           round(1/math.pi**2,4)),
        ]
        print(f"  Comparing mu={mu:.4f} to analytical expressions from the geometry:")
        for expr,val in exprs:
            diff=abs(val-mu)
            flag=" *** MATCH ***" if diff<0.01 else (" <- close" if diff<0.03 else "")
            print(f"    {expr:<22s} = {val:.4f}   delta={diff:.4f}{flag}")

    # Series 2
    s2=[r for r in rows if r["series"]==2]
    if len(s2)>=2:
        print()
        print("─"*65)
        print("SERIES 2  —  DIRECTIONAL ANALYSIS (tensor vs scalar)")
        print()
        print(f"  {'Label':<22s} {'Gradient':>10s} {'Score':>10s}  Output")
        print(f"  {'─'*22} {'─'*10} {'─'*10}  {'─'*30}")
        for r in s2:
            print(f"  {r['label']:<22s} {r['G']:>10.4f} {(r['score'] or 0):>10.4f}  {(r['output'] or '')[:35]}")

        scores={r["design"]:r["score"] for r in s2 if r["score"]}
        if len(scores)>=2:
            spread=max(scores.values())-min(scores.values())
            best=max(scores,key=scores.get)
            alt=scores.get("alternating",0)
            asc=scores.get("ascending",0)
            dsc=scores.get("descending",0)
            print()
            print(f"  Score spread: {spread:.4f}")
            if spread<0.05:
                print("  SCALAR COUPLING — direction does not affect output.")
                print("  mu is a scalar. Gradient magnitude is the sole driver.")
            elif best=="alternating" and alt>max(asc,dsc)*1.05:
                print("  TENSOR COUPLING — alternating profile outperforms.")
                print("  Coupling responds to oscillation frequency across the stream.")
                print("  Optimal input: high-low-high-low density alternation.")
                print("  This matches the natural structure of technical writing.")
            else:
                print(f"  ASYMMETRIC — {best} profile strongest.")
                print("  Coupling has a preferred direction in symbol space.")

    # Global
    v=[r for r in rows if r["score"] is not None]
    if len(v)>=4:
        print()
        print("─"*65)
        print("GLOBAL CORRELATIONS (all prompts)")
        G_a=[r["G"] for r in v]; M_a=[r["mean_ns"] for r in v]; P_a=[r["score"] for r in v]
        rg,rm=pearson(G_a,P_a),pearson(M_a,P_a)
        print(f"  Gradient vs Score:  r={rg:.4f}  r²={rg**2:.4f}")
        print(f"  Mean ns vs Score:   r={rm:.4f}  r²={rm**2:.4f}")

    print()
    export=[{"label":r["label"],"series":r["series"],"design":r["design"],
             "G":r["G"],"mean_ns":r["mean_ns"],"score":r["score"],
             "res":r["res"],"output":r["output"],"seq":r["m"]["seq"]} for r in rows]
    with open("coupling_results.json","w") as f:
        json.dump(export,f,indent=2)
    print("Saved: coupling_results.json")

def main():
    if len(sys.argv)<2:
        print("Usage: python tools/flexoelectric_diagnostic.py <output.txt> [--series2]")
        sys.exit(1)
    path=Path(sys.argv[1])
    if not path.exists():
        print(f"Not found: {path}"); sys.exit(1)
    text=path.read_text(encoding="utf-8",errors="replace")
    results=parse_session(text)
    if not results:
        print("No data found."); sys.exit(1)
    print(f"Parsed {len(results)} prompts.\n")
    analyze(results, series2="--series2" in sys.argv)

if __name__=="__main__":
    main()
