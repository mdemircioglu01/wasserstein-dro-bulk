import numpy as np, json, sys
from dro_bulk import generate_instance, InstanceConfig, DROConfig, DROSolver
from dro_bulk.recourse import recourse_value

def make(N, seed=7):
    return generate_instance(InstanceConfig(scale='M', seed=seed, n_train=N, n_test=4000,
                             mu_shortage_cost=6.0, supply_tightness=1.05))

def oos_costs(inst, sol):
    cs=inst.flat(inst.short_cost); co=inst.flat(inst.surp_cost)
    xbar=inst.flat(sol['x'].sum(0)); tr=float((inst.cost*sol['x']).sum())
    return tr + recourse_value(xbar, inst.test_samples, cs, co)   # array over test scenarios

part = sys.argv[1] if len(sys.argv)>1 else 'A'

if part=='A':   # price of robustness + out-of-sample, cost objective, fixed N=100
    inst=make(100)
    rows=[]
    base=None
    print("eps | wc_cost | %vsSAA | oos_mean | oos_std | oos_p95 | %tail_vsSAA")
    base_p95=None
    for eps in [0,5,10,20,40,80]:
        sol=DROSolver(inst,DROConfig(norm='l1',epsilon=eps,support='box')).solve_single('cost')
        wc=sol['cost_total']; o=oos_costs(inst,sol)
        if base is None: base=wc; base_p95=np.percentile(o,95)
        r=dict(eps=eps, wc=round(wc), pct=round(100*(wc-base)/base,1),
               oos_mean=round(float(o.mean())), oos_std=round(float(o.std())),
               oos_p95=round(float(np.percentile(o,95))),
               tail_pct=round(100*(np.percentile(o,95)-base_p95)/base_p95,1))
        rows.append(r)
        print(f"{eps:4} {r['wc']:8} {r['pct']:6} {r['oos_mean']:8} {r['oos_std']:7} {r['oos_p95']:8} {r['tail_pct']:6}")
    json.dump(rows, open('exp_A.json','w'))

if part=='B':   # value of robustness under demand-distribution shift (small N)
    inst=make(30)
    cs=inst.flat(inst.short_cost); co=inst.flat(inst.surp_cost)
    def oos_shift(sol, shift):
        xbar=inst.flat(sol['x'].sum(0)); tr=float((inst.cost*sol['x']).sum())
        return tr + recourse_value(xbar, inst.test_samples*(1+shift), cs, co)
    eps_grid=[0,100,200,300]
    shifts=[0.0,0.10,0.20]
    sols={e:DROSolver(inst,DROConfig(norm='l1',epsilon=e,support='box')).solve_single('cost') for e in eps_grid}
    print("            out-of-sample MEAN cost under demand shift")
    print("eps \\ shift |   0%      +10%     +20%")
    table={}
    for e in eps_grid:
        vals=[float(oos_shift(sols[e],s).mean()) for s in shifts]
        table[e]=vals
        print(f"  {e:4}      " + "  ".join(f"{v:8.0f}" for v in vals))
    # best eps per shift
    best=[eps_grid[int(np.argmin([table[e][j] for e in eps_grid]))] for j in range(len(shifts))]
    print("best eps   :", "      ".join(f"{b:6}" for b in best))
    json.dump({'eps':eps_grid,'shifts':shifts,'table':table,'best':best}, open('exp_B.json','w'))

if part=='C':   # tail-risk value of robustness, costly stockouts + small N + surge
    inst=generate_instance(InstanceConfig(scale='M', seed=7, n_train=25, n_test=5000,
                           mu_shortage_cost=12.0,
                           supply_tightness=1.0, coeff_variation=0.4))
    cs=inst.flat(inst.short_cost); co=inst.flat(inst.surp_cost)
    def oos(sol, shift):
        xbar=inst.flat(sol['x'].sum(0)); tr=float((inst.cost*sol['x']).sum())
        c=tr + recourse_value(xbar, inst.test_samples*(1+shift), cs, co)
        cvar=c[c>=np.quantile(c,0.90)].mean()   # CVaR@90%
        return float(c.mean()), float(cvar)
    print("Regime: costly stockouts (b=12x), tight supply, cv=0.4, N=25; test = +15% demand surge")
    print(" eps |  nominal_mean | surge_mean | surge_CVaR90 | %mean_vsSAA | %CVaR_vsSAA")
    base=None
    for e in [0,50,100,200,400]:
        sol=DROSolver(inst,DROConfig(norm='l1',epsilon=e,support='box')).solve_single('cost')
        nm,_=oos(sol,0.0); sm,sc=oos(sol,0.15)
        if base is None: base=(sm,sc)
        print(f" {e:4} | {nm:12.0f} | {sm:10.0f} | {sc:12.0f} | {100*(sm-base[0])/base[0]:10.2f} | {100*(sc-base[1])/base[1]:10.2f}")

if part=='D':   # small-sample value of robustness (SAA overfits N=10 heavy-tailed sample)
    print("Regime: N=10 training, heavy-tailed lognormal cv=0.5, b=12x; test on TRUE distribution (5000)")
    print(" eps | oos_mean | oos_CVaR90 | %mean_vsSAA | %CVaR_vsSAA   (avg over 8 training samples)")
    import numpy as np
    eps_grid=[0,100,200,400,800]
    agg={e:[] for e in eps_grid}; aggc={e:[] for e in eps_grid}
    for sd in range(8):
        inst=generate_instance(InstanceConfig(scale='M', seed=sd, n_train=10, n_test=5000,
                               mu_shortage_cost=12.0, supply_tightness=1.0, coeff_variation=0.5))
        cs=inst.flat(inst.short_cost); co=inst.flat(inst.surp_cost)
        for e in eps_grid:
            sol=DROSolver(inst,DROConfig(norm='l1',epsilon=e,support='box')).solve_single('cost')
            xbar=inst.flat(sol['x'].sum(0)); tr=float((inst.cost*sol['x']).sum())
            c=tr+recourse_value(xbar, inst.test_samples, cs, co)
            agg[e].append(float(c.mean())); aggc[e].append(float(c[c>=np.quantile(c,0.9)].mean()))
    base=np.mean(agg[0]); basec=np.mean(aggc[0])
    for e in eps_grid:
        m=np.mean(agg[e]); cv=np.mean(aggc[e])
        print(f" {e:4} | {m:8.0f} | {cv:10.0f} | {100*(m-base)/base:10.2f} | {100*(cv-basec)/basec:10.2f}")

if part=='E':   # single clean table: price vs value of robustness (small N)
    import numpy as np
    eps_grid=[0,20,50,100,200,400]
    # average over several small-N training samples for stability
    G={e:[] for e in eps_grid}; M_={e:[] for e in eps_grid}; C={e:[] for e in eps_grid}
    for sd in range(8):
        inst=generate_instance(InstanceConfig(scale='M', seed=sd, n_train=20, n_test=5000,
                               mu_shortage_cost=12.0, supply_tightness=1.0, coeff_variation=0.4))
        cs=inst.flat(inst.short_cost); co=inst.flat(inst.surp_cost)
        for e in eps_grid:
            sol=DROSolver(inst,DROConfig(norm='l1',epsilon=e,support='box')).solve_single('cost')
            xbar=inst.flat(sol['x'].sum(0)); tr=float((inst.cost*sol['x']).sum())
            c=tr+recourse_value(xbar, inst.test_samples, cs, co)
            G[e].append(sol['cost_total']); M_[e].append(float(c.mean()))
            C[e].append(float(c[c>=np.quantile(c,0.9)].mean()))
    g0=np.mean(G[0]); m0=np.mean(M_[0]); c0=np.mean(C[0])
    print("Price vs value of robustness (M, N=20, b=12x, tight supply, cv=0.4; avg of 8 samples)")
    print(" eps | guar_cost %SAA | oos_mean %SAA | oos_CVaR90 %SAA")
    rows=[]
    for e in eps_grid:
        gp=100*(np.mean(G[e])-g0)/g0; mp=100*(np.mean(M_[e])-m0)/m0; cp=100*(np.mean(C[e])-c0)/c0
        rows.append((e,round(gp,1),round(mp,2),round(cp,2)))
        print(f" {e:4} | {gp:13.1f} | {mp:13.2f} | {cp:14.2f}")
    json.dump(rows, open('exp_E.json','w'))
