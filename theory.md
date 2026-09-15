# Optimal stopping: model, derivations and numerical decisions

This note derives the mathematical objects used in the first implementation and explains the assumptions behind its validation. The results below belong to established probability, option-pricing and numerical-analysis theory. The exposition and the links to these experiments are prepared for this repository. Attribution is recorded in [SOURCES.md](SOURCES.md).

## 1. A valuation model with an explicit information structure

Work on a filtered probability space $(\Omega,\mathcal F,(\mathcal F_t)_{0\leq t\leq T},Q)$ carrying Brownian motion $W^Q$. Use its usual augmented filtration. The money-market account is $B_t=e^{rt}$, the stock pays a continuous proportional dividend yield $q$, and its ex-dividend price satisfies

$$
dS_t=(r-q)S_t\,dt+\sigma S_t\,dW_t^Q,\qquad S_0>0.
$$

The coefficients are constant and finite, $T>0$, and $\sigma>0$ for the diffusion arguments. Trading is frictionless and continuous, borrowing and lending use the same rate, and the model permits the self-financing replication described below. These are substantial financial assumptions. Risk-neutral simulation computes model values; it does not forecast actual stock returns.

Applying Itô's formula to $\log S_t$ gives

$$
\log S_t-\log S_s
=\left(r-q-\frac{\sigma^2}{2}\right)(t-s)
+\sigma(W_t^Q-W_s^Q).
$$

Independent normal increments therefore give exact transitions at every selected date. Refining the simulation calendar changes exercise opportunities without introducing an Euler error in the stock process. The implementation is [simulation.py](../../src/qf_research/optimal_stopping/simulation.py).

For every finite real $p$,

$$
E_Q[S_t^p]=S_0^p
\exp\left(p(r-q)t+\frac{p(p-1)}{2}\sigma^2t\right).
$$

In particular, the finite-date call and put payoffs are square-integrable. Also $e^{-(r-q)t}S_t$ is a martingale, as follows directly from the conditional normal increment formula. With dividends, $e^{-rt}S_t$ alone generally is not a martingale; the discounted cumulative gains include the dividend stream.

## 2. Replication and the European benchmark

The financial foundation is the work of [Black and Scholes](https://www.journals.uchicago.edu/doi/10.1086/260062) and [Merton](https://www.maths.tcd.ie/~dmcgowan/Merton.pdf). Here is the calculation for this repository's dividend convention.

Let $v(t,S)$ be a sufficiently smooth option value before expiry. A replicating portfolio holds $\Delta_t$ shares and cash $v-\Delta_t S_t$. Under a physical stock drift $\mu$, its self-financing gains are

$$
d\Pi_t=\Delta_t\,dS_t+q\Delta_t S_t\,dt
+r(v-\Delta_t S_t)\,dt.
$$

Itô's formula for $v(t,S_t)$ and equality of the Brownian coefficients imply $\Delta_t=v_S$. Matching drift coefficients then cancels $\mu$ and gives

$$
v_t+\frac12\sigma^2S^2v_{SS}+(r-q)Sv_S-rv=0,
\qquad v(T,S)=g(S).
$$

The terminal payoff has a kink at the strike, so smoothness is understood before expiry, with an appropriate growth condition. Applying Itô's formula to $e^{-rt}v(t,S_t)$ under $Q$ cancels its drift. The stochastic integral is a true martingale under the integrability conditions of this model, which yields

$$
v(t,S_t)=E_Q[e^{-r(T-t)}g(S_T)\mid\mathcal F_t].
$$

This provides two independent numerical routes: approximate the backward PDE, or estimate the discounted payoff expectation. Integrating the lognormal density for $g(S)=(S-K)^+$ gives

$$
c=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2),\qquad
d_{1,2}=\frac{\log(S_0/K)+(r-q\pm\sigma^2/2)T}{\sigma\sqrt T}.
$$

The payoff identity $(S-K)^+-(K-S)^+=S-K$ gives

$$
c-p=S_0e^{-qT}-Ke^{-rT}.
$$

This identity also checks discounting and the dividend sign. The analytical implementation handles $\sigma=0$ through deterministic discounted payoffs and $T=0$ through intrinsic value. At those singular cases, its Greek routine rejects the request rather than assigning a conventional derivative silently.

The Greek tests perturb the pricing inputs independently of the derivative formulae. Vega and rho are per unit change in volatility and rate; theta is per calendar year and has the opposite sign to the derivative with respect to remaining maturity.

## 3. Why conditional expectation is a least-squares target

Let $Y\in L^2(Q)$, let $\mathcal G\subseteq\mathcal F$ be a sub-$\sigma$-algebra, and write $X=E_Q[Y\mid\mathcal G]$. Conditional Jensen gives $E[X^2]\leq E[Y^2]$, so $X\in L^2(\mathcal G)$.

For every bounded $\mathcal G$-measurable $H$, the defining integral identity for conditional expectation gives

$$
E[(Y-X)H]=0.
$$

For general $H\in L^2(\mathcal G)$, truncate $H$ to $H_n=\max(-n,\min(H,n))$. Dominated convergence gives $H_n\to H$ in $L^2$. Cauchy-Schwarz then implies

$$
\left|E[(Y-X)(H-H_n)]\right|
\leq \|Y-X\|_2\|H-H_n\|_2\longrightarrow0.
$$

The orthogonality identity therefore extends to every such $H$. Given any competing estimator $Z\in L^2(\mathcal G)$, expand the square and use $H=X-Z$:

$$
E[(Y-Z)^2]
=E[(Y-X)^2]+E[(X-Z)^2].
$$

Thus $X$ minimises mean-square error and equality requires $Z=X$ almost surely. This is the standard projection property of conditional expectation; [Williams](https://www.cambridge.org/highereducation/books/probability-with-martingales/B4CFCE0D08930FB46C6E93E775503926/foundations/80F5CE8F44C4F29DE53486C0AA97EAB1) provides background on the underlying conditional-expectation and martingale theory.

A polynomial regression only projects onto a finite-dimensional family. Even with unlimited observations, that family need not contain the conditional expectation. Sample error, basis approximation and optimal stopping are therefore separate issues.

## 4. Exercise rights and the discrete Snell envelope

Fix the contract's positive dates $0<t_1<\cdots<t_m=T$. An admissible stopping time $\tau$ takes values in this set and satisfies $\{\tau=t_i\}\in\mathcal F_{t_i}$. Define the discounted reward $Z_i=e^{-rt_i}g(S_{t_i})$ and recursively set

$$
U_m=Z_m,\qquad
U_i=\max\{Z_i,E[U_{i+1}\mid\mathcal F_{t_i}]\},
\quad i=m-1,\ldots,1.
$$

The time-zero value is $U_0=E[U_1\mid\mathcal F_0]$. There is no time-zero exercise right in this release. Adding it would replace that value by the maximum of continuation and $g(S_0)$.

The sequence $U_i$ dominates $Z_i$ and is a supermartingale on the exercise grid. It is the smallest such process: any dominating supermartingale $A$ has $A_m\geq U_m$, and backward induction yields

$$
A_i\geq\max\{Z_i,E[A_{i+1}\mid\mathcal F_{t_i}]\}
\geq U_i.
$$

Define the stopping index $\kappa^*=\min\{i:U_i=Z_i\}$ and the calendar stopping time $\tau^*=t_{\kappa^*}$. The equality event uses current information. Before $\kappa^*$ the continuation equality holds, so $U_{i\wedge\kappa^*}$ is a martingale on the exercise grid. Finite-horizon optional sampling gives

$$
E[Z_{\kappa^*}]=E[U_1]=U_0.
$$

For any other admissible stopping index $\kappa$, domination and the supermartingale property give $E[Z_\kappa]\leq U_0$. The initial Brownian information is trivial up to null events, so $U_0$ is constant almost surely. Integrability follows from the finite number of square-integrable rewards. This proves the recursion's valuation and stopping claims for the finite calendar. It does not prove convergence to a continuous-time American contract.

Adding dates can only increase the exact value when the new calendar contains the old one, since every old stopping rule remains feasible. A denser but non-nested calendar does not have that immediate ordering argument.

## 5. A learned policy and its independent evaluation

The regression method follows [Longstaff and Schwartz (2001)](https://www.anderson.ucla.edu/documents/areas/fac/finance/file11.pdf). Training proceeds backwards. At date $t_i$, each in-the-money path supplies its current spot and the discounted future cash flow selected by the later fitted policy. A least-squares fit estimates continuation; exercise replaces that cash flow when intrinsic value is larger.

This implementation centres and scales training spots, uses monomials up to the requested degree, and solves through NumPy's least-squares routine. It reduces the degree if the design is rank deficient or its condition number exceeds $10^{10}$. Fewer than two eligible observations give a continue decision. Fitted continuation is clipped at zero. These conventions and diagnostics are part of the numerical method.

Evaluation freezes the fitted functions and walks forward along a separate sample. A decision depends on current spot, the frozen function and whether the option is still alive. Future realised prices are excluded from the decision. The test using identical current states and different future paths verifies this information restriction.

Conditional on training data $D$, the learned rule $\hat\tau_D$ is admissible for an independent evaluation path. Its model value is

$$
V(D)=E_Q[e^{-r\hat\tau_D}g(S_{\hat\tau_D})\mid D]\leq U_0.
$$

The inequality concerns a population expectation. Neither a realised Monte Carlo mean nor a finite-grid tree value is automatically a rigorous lower or upper bound for the diffusion optimum.

For independent evaluation cash flows $Y_k$, the sample mean estimates $V(D)$. A fixed European control uses

$$
\widetilde Y_k=Y_k-Y_k^{E}+V^{E},
\qquad E[\widetilde Y_k\mid D]=V(D),
$$

where $Y_k^{E}=e^{-rT}g(S_T^{(k)})$ uses the same path and $V^E$ is analytical. Coefficient one is fixed before evaluation. It is not estimated using the evaluation sample and is not claimed to minimise variance. Both raw and controlled estimates are saved.

Squared-error prediction quality and exercise quality are related but different objectives. If all later decisions were optimal, the loss from choosing the wrong action at a single state would be the absolute gap between immediate and optimal continuation value. Mistakes near indifference cost little; mistakes with a large gap cost more. A global regression fit statistic alone does not measure that financial loss.

## 6. Deterministic numerical references

### Binomial backward induction

The tree follows [Cox, Ross and Rubinstein (1979)](https://www.sciencedirect.com/science/article/pii/0304405X79900151). With step $h$, take

$$
u=e^{\sigma\sqrt h},\quad d=u^{-1},\quad
p=\frac{e^{(r-q)h}-d}{u-d}.
$$

The solver rejects $p\notin[0,1]$. Continuation is $e^{-rh}[pV_{\rm up}+(1-p)V_{\rm down}]$. The maximum with intrinsic value is taken only at permitted dates. Every exercise date must align exactly with the grid.

The tree uses $O(N^2)$ arithmetic and $O(N)$ working memory. Its price converges only under appropriate refinement assumptions; lattice placement around the strike and exercise frontier can make finite-grid errors irregular.

### Finite differences

Use $S_i=i\Delta S$ on $[0,S_{\max}]$. Away from boundary adjustments, the spatial operator has coefficients

$$
\ell_i=\tfrac12\sigma^2i^2-\tfrac12(r-q)i,\quad
u_i=\tfrac12\sigma^2i^2+\tfrac12(r-q)i,\quad
d_i=-\sigma^2i^2-r.
$$

If either off-diagonal coefficient is negative, that node instead uses positive drift-transition rates

$$
\ell_i=\tfrac12\sigma^2i^2+\max(-(r-q)i,0),\quad
u_i=\tfrac12\sigma^2i^2+\max((r-q)i,0),\quad
d_i=-\ell_i-u_i-r.
$$

These nodes use a first-order upwind drift approximation. The solver reports their count; it makes no global second-order accuracy claim for all parameters.

Marching backwards in calendar time with step $h$ solves

$$
(I-\theta hL)V^{\rm new}
=(I+(1-\theta)hL)V^{\rm old}+\text{boundary contributions}.
$$

The default is $\theta=1/2$. Two implicit Euler half steps replace the first full step after expiry and after each exercise projection. This uses the established Rannacher smoothing idea; [Giles and Carter](https://people.maths.ox.ac.uk/gilesm/files/giles_carter.pdf) analyse its convergence and payoff regularity. The selected smoothing convention is documented here and is not a reproduction of their full experiment design.

At $S=0$, the put boundary optimises the discounted strike over the remaining allowed dates. At $S_{\max}$, the put boundary is zero. The call boundary optimises the asymptotic discounted linear payoff over the remaining dates. The latter two upper boundaries neglect the chance of returning across the strike, so domain expansion at fixed spacing is a necessary diagnostic. At a decision date the PDE first computes continuation, then applies the exercise projection.

SciPy solves each tridiagonal system. Complexity is $O(MN)$ time and $O(M)$ working memory for $M$ space intervals and $N$ time steps. Smoothing helps control nonsmooth-payoff oscillations, but a Crank-Nicolson step is not unconditionally positivity-preserving. Convergence and financial checks remain necessary.

## 7. Sampling uncertainty and experimental interpretation

For iid finite-variance cash flows, $\operatorname{Var}(\bar Y_N)=\operatorname{Var}(Y)/N$. The central limit theorem motivates an approximate interval

$$
\bar Y_N\ \pm\ t_{N-1,0.975}\frac{s_Y}{\sqrt N}.
$$

Student critical values do not make this interval exact for an option payoff's non-Gaussian distribution. When antithetic paths are used, one independent observation is the average of a pair, so $N$ in the standard-error calculation is the number of pairs. Treating the two correlated members as independent would misstate uncertainty.

The $N^{-1/2}$ sampling rate has no dimension in its exponent when the payoff variance is finite. Dimension can still increase that variance, the cost of a sample and the difficulty of learning an exercise policy. It is therefore not a promise of dimension-independent computational cost.

The experiments separate the following quantities:

| Quantity | How it is assessed | What the assessment omits |
| --- | --- | --- |
| European numerical error | Comparison with an analytical value | Model misspecification |
| Bermudan reference error | Grid refinement and agreement of tree and PDE | A certified diffusion error bound |
| Evaluation noise | Independent payoff sample and its standard error | Training variability and policy approximation |
| Repeated-policy variability | New training and evaluation streams each repetition | Separation of those two variance components |
| Exercise-calendar value | Nested calendars under matching model inputs | A continuous-time limit theorem |
| Economic sensitivity | Rates and volatility changed one scenario at a time | Empirical causal identification |

The benchmark report is [results/baseline/findings.md](results/baseline/findings.md). Its numerical evidence should be read alongside these assumptions, especially when a fitted policy estimate lies above a reference or a small numerical residual appears to violate an economic inequality.
