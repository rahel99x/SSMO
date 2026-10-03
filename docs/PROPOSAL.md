# B — Singular-Sensitivity Measure Operators
## Architecture, generalized derivatives, reference data, numerical validation, and A100 40 GB implementation

**Prepared:** 2 October 2026.  
**Status:** proposed research program; no trained sensitivity operator or PDE performance result is established here.  
**Primary source:** Candidate B, §§3.1–3.7, of the supplied `FRONTIER_PROPOSALS_AND_NOVELTY_AUDIT.md`.  
**Companion:** `A_OBSTRUCTION_PRODUCING_PDE_OPERATORS.md`.  
**Execution target:** one NVIDIA A100 with 40 GB VRAM through a regular USC CARC account, with ordinary allocated CPU resources.

---

## Reading map

- [1. Copy-ready instruction to the implementing agent](#1-copy-ready-instruction-to-the-implementing-agent)
- [2. Problem definition and output types](#2-problem-definition-and-output-types)
- [3. The mathematical representation](#3-the-mathematical-representation)
- [4. Analytic references and event mathematics](#4-analytic-references-and-event-mathematics)
- [5. Weak metrics, labels, and error control](#5-weak-metrics-labels-and-error-control)
- [6. Architecture: represent support, bulk variation, and directional linearity](#6-architecture-represent-support-bulk-variation-and-directional-linearity)
- [7. Losses and training curriculum](#7-losses-and-training-curriculum)
- [8. Data, splits, and reference quality](#8-data-splits-and-reference-quality)
- [9. Software interfaces and unit tests](#9-software-interfaces-and-unit-tests)
- [10. Work packages and experiment gates](#10-work-packages-and-experiment-gates)
- [11. Baselines, statistical analysis, and inverse-problem use](#11-baselines-statistical-analysis-and-inverse-problem-use)
- [12. Main weaknesses and how to address them](#12-main-weaknesses-and-how-to-address-them)
- [13. Efficiency specific to this architecture](#13-efficiency-specific-to-this-architecture)
- [14. Compute and deployment](#14-compute-and-deployment)
- [15. Deliverables and the first executable audit](#15-deliverables-and-the-first-executable-audit)
- [16. Novelty boundaries and primary references](#16-novelty-boundaries-and-primary-references)

## 1. Copy-ready instruction to the implementing agent

Build a reusable sensitivity operator for PDE solutions with moving discontinuities. Its output must distinguish the diffuse derivative from the singular contribution carried by shock motion. Start with scalar one-dimensional conservation laws, exact Riemann and piecewise-constant reference problems, smooth output queries, and finite-dimensional parameter directions.

Do not begin with 3-D turbulence, a full Euler shock system, or arbitrary topology changes. Do not rasterize every derivative into a smooth grid tensor and then claim that the architecture represents a measure. Evaluate atoms directly in query pairings. Preserve a trusted numerical reference and strong classical front-sensitivity baseline throughout.

The first objective is to establish a real representation advantage: accurate transferable observable derivatives at different resolutions and for new smooth queries. The second is a computational advantage over classical sensitivity calculations at comparable state information and accuracy. A differentiable front tracker is a required baseline, not a discovery we can claim as new.

### 1.1 Evidence categories

- **SOURCE:** the supplied Candidate B proposes a signed-measure sensitivity output, diffuse and atomic parts, weak metrics, and classical shock-sensitivity controls.
- **DERIVED:** formulas proved below for their stated regularity/event assumptions.
- **PROPOSED:** architectures, grids, batch sizes, workload choices, and practical success thresholds.
- **VALIDATED:** comparisons with analytic references or independently converged numerical labels.
- **MEASURED:** actual runtime, memory, training, and test results.

The implementation details below expand the source; they are not reported experimental findings. The support script checks small analytic examples on CPU only. Consult its actual output before making any execution claim.

### 1.2 Non-negotiable rules

1. A moving shock derivative is generally not an ordinary bounded grid function.
2. A directional derivative may fail to exist at some events, even in the chosen generalized topology. Use a scoped status or one-sided derivative, not a fabricated two-sided gradient.
3. Signed measures are not probability distributions. Do not normalize their masses to one or feed them directly to an ordinary probability-Wasserstein loss.
4. Linear-observable pairings and nonlinear objective derivatives have different formulas at a jump.
5. If a model is a derivative operator, dependence on the input parameter direction must be linear wherever the derivative exists.
6. A measure predictor that disagrees with finite variations of its associated state map is not a valid sensitivity model.
7. Do not count a smoothed/viscous derivative as the inviscid reference without a limit study.
8. Include the state trajectory, shock localization, teacher generation, and query evaluation in performance comparisons.

---

## 2. Problem definition and output types

### 2.1 Start with a well-posed scalar conservation law

Use

\[
\partial_tu+\partial_xF(u;\alpha)=0
\]

on a declared spatial interval and time range, with an entropy solution and explicit initial/boundary conditions. Let \(\alpha\in\mathbb R^p\) parameterize initial data, flux coefficients, sources, or boundaries. Begin with initial-data parameters and a fixed smooth convex flux; add the others separately.

For a direction \(v\in\mathbb R^p\), the target is

\[
\boxed{
\mu_v(t)=D_\alpha u(\cdot,t;\alpha)[v],
}
\]

interpreted in a specified weak/distributional sense. Start with finite signed Radon measures on a bounded domain and observation times where that representation is justified. Do not claim that every hyperbolic sensitivity belongs to this class.

Use fixed physical coordinates and an explicit sign convention: each shock has left trace \(u^-\), right trace \(u^+\), and position \(s\). Let \([u]=u^+-u^-\). Positive position variation shifts the interface to the right.

### 2.2 Output contract

```text
SensitivityResult(
    parent_id, model_version, parameter_definition, direction,
    observation_time, physical_domain,
    status = regular | near_event | one_sided | unresolved,
    diffuse_representation,
    shock_positions,
    left_and_right_state_traces,
    position_directional_derivatives,
    atom_weights,
    event_metadata,
    query_capabilities,
    precision_and_validation_record
)
```

An output atom has units of state times length per parameter-direction unit. The diffuse density has units of state per parameter-direction unit. Cell widths must appear only where required by integration; an atomic weight is not multiplied by a cell width.

### 2.3 Three deployment questions

Keep these experiments separate:

- **Teacher-conditioned sensitivity:** a trusted forward state/trajectory is supplied. The method amortizes sensitivities or many query gradients, not the forward solve.
- **Joint state-and-sensitivity surrogate:** only initial data, parameters, and time are supplied. Forward prediction cost and error are part of the result.
- **Online hybrid sensitivity:** trusted numerical state steps are augmented by learned derivative evolution. Count the entire coupled runtime.

Do not give the learned model future reference shocks while comparing it with a baseline that receives only initial data.

---

## 3. The mathematical representation

### 3.1 One moving discontinuity

For a scalar parameter \(a\), assume

\[
u(x;a)=u_L(x;a)\mathbf1_{x<s(a)}
+u_R(x;a)\mathbf1_{x>s(a)},
\]

with smooth side fields near the interface and a differentiable shock position at the parameter of interest. Then

\[
\boxed{
\partial_a u
=(\partial_a u_L)\mathbf1_{x<s}
+(\partial_a u_R)\mathbf1_{x>s}
+(u_L(s;a)-u_R(s;a))s'(a)\delta_s.
}
\]

Proof: integrate against a fixed smooth test function, split the integral at \(s(a)\), and differentiate each moving integration limit. The boundary terms combine into the displayed atom.

With the jump convention \([u]=u_R-u_L\), the weight is \(-[u]s'\). Test this sign repeatedly in code.

For a translated increasing step \(u(x;a)=\mathbf1_{x>a}\), the weight is \(-1\), so \(\partial_a u=-\delta_a\). This is the first unit test.

### 3.2 Multiple shocks and parameter directions

Away from event degeneracies,

\[
\boxed{
\mu_v=g_v(x)\,dx+\sum_{j=1}^{J}w_{j,v}\delta_{s_j},
\qquad
w_{j,v}=(u_j^- -u_j^+)\,\dot s_{j,v},
}
\]

where \(\dot s_{j,v}=D_\alpha s_j[v]\). The positions and traces are features of the base solution; they must not change when only the derivative direction \(v\) changes.

For vector-valued states, \(w_j\) becomes a vector. Do not immediately generalize scalar entropy/shift regularity theorems to arbitrary systems; scope the first paper to a class with a reliable reference theory [B1,B2].

### 3.3 Why a grid-only target is badly conditioned

A difference quotient for a translated step has magnitude \(1/h\) on an interval of width \(h\), so its \(L^2\) norm is \(h^{-1/2}\). The corresponding measure has finite total variation. A neural model asked to match bounded pointwise derivative arrays across resolutions is being asked to learn an increasingly concentrated object.

This does not prove that grid-based sensitivity is useless. Appropriate weak evaluation, numerical differentiation, or classical shock treatment can work. Those are baselines. The proposed gain is an explicit resolution-compatible representation of the singular part.

### 3.4 Linear observable queries

For a smooth fixed query \(\psi\),

\[
Q_\psi(\alpha,t)=\int_\Omega\psi(x)u(x,t;\alpha)\,dx,
\]

and

\[
\boxed{
D_\alpha Q_\psi[v]
=\int_\Omega\psi(x)g_v(x)\,dx
+\sum_jw_{j,v}\psi(s_j).
}
\]

If the query itself depends on \(\alpha\), add

\[
\int_\Omega D_\alpha\psi[v]u\,dx.
\]

A point sensor at a shock and an indicator-window endpoint crossing a shock are not smooth query cases. Treat their one-sided/nondifferentiable behavior explicitly instead of hiding it with an arbitrary smoothing width.

### 3.5 Critical distinction: nonlinear objective derivatives

For

\[
J(\alpha)=\int_\Omega\ell(x,u(x;\alpha),\alpha)\,dx,
\]

split the integral across all smooth regions. For one interface,

\[
\boxed{
D J[v]
=\int_{x<s}\big(\ell_u(x,u_L,\alpha)g_{L,v}+\ell_\alpha[v]\big)dx
+\int_{x>s}\big(\ell_u(x,u_R,\alpha)g_{R,v}+\ell_\alpha[v]\big)dx
+\big[\ell(s,u_L,\alpha)-\ell(s,u_R,\alpha)\big]\dot s_v.
}
\]

The boundary contribution uses the **jump in the objective density**, not an arbitrarily chosen value of \(\ell_u\) multiplied by the state-derivative atom.

For \(\ell=\tfrac12(u-d(x))^2\), the interface term is

\[
\frac12\big[(u_L-d(s))^2-(u_R-d(s))^2\big]\dot s_v.
\]

Using the left trace of \(u-d\) times \((u_L-u_R)\dot s_v\) is generally wrong. A signed state-derivative measure alone is insufficient for arbitrary nonlinear objectives; return base traces and shock motion as well. This is one reason to couple the state and sensitivity representations.

### 3.6 Higher-dimensional extension

For a smooth moving interface \(\Gamma(\alpha)\) with a declared normal and normal parameter displacement \(V_{n,v}\),

\[
D_\alpha u[v]
=g_v\,dx+(u^- -u^+)V_{n,v}\,
\mathcal H^{d-1}\!\restriction_{\Gamma},
\]

with signs fixed by the chosen interior/exterior convention. Query evaluation uses surface quadrature. A level-set representation must not divide by a nearly zero gradient without detection.

This is a later extension. Surface topology changes, self-intersections, and systems shocks require additional analysis. Do not build a 2-D surface model before the 1-D event and query semantics are reliable.

---

## 4. Analytic references and event mathematics

### 4.1 Burgers single shock

Use \(F(u)=u^2/2\), with \(u_L>u_R\), initial shock location \(a\), and no boundary interaction over the horizon. The shock speed and position are

\[
c=\frac{u_L+u_R}{2},\qquad s(t)=a+ct.
\]

For direction \((v_L,v_R,v_a)\),

\[
\dot s_v=v_a+\frac t2(v_L+v_R),
\]

\[
g_v(x)=v_L\mathbf1_{x<s}+v_R\mathbf1_{x>s},
\qquad
w_v=(u_L-u_R)\dot s_v.
\]

The observable derivative is therefore

\[
D Q_\psi[v]
=v_L\int_{x_{\min}}^{s}\psi(x)dx
+v_R\int_s^{x_{\max}}\psi(x)dx
+(u_L-u_R)\dot s_v\psi(s).
\]

This is an exact target independent of a raster resolution. Standard shock and rarefaction solutions are documented in the Clawpack Riemann materials [B3]. The derivative formulas here follow by differentiation of those solutions, not by a learned result.

### 4.2 Rarefaction control

For \(u_L<u_R\), the Burgers Riemann solution has a continuous fan between \(a+u_Lt\) and \(a+u_Rt\), with interior value \((x-a)/t\). Because the state is continuous at the fan boundaries, moving those boundaries produces no state-jump atom there.

This is a mandatory negative control for false atom prediction. Sensitivities can still have discontinuous diffuse densities. Restrict \(t\) away from zero when using formulas with \(1/t\).

### 4.3 Two merging Burgers shocks

For constant states \(u_L>u_M>u_R\) and initial positions \(a<b\),

\[
c_1=(u_L+u_M)/2,\quad c_2=(u_M+u_R)/2,
\quad \tau=\frac{b-a}{c_1-c_2}.
\]

Before \(\tau\), there are two shocks. After collision, the speed is \(c_3=(u_L+u_R)/2\) and

\[
s(t)=a+c_1\tau+c_3(t-\tau).
\]

For a direction, differentiate

\[
\dot\tau_v=
\frac{(v_b-v_a)(c_1-c_2)-(b-a)(\dot c_1-\dot c_2)}
{(c_1-c_2)^2},
\]

\[
\boxed{
\dot s_v=v_a+\dot c_1\tau+\dot c_3(t-\tau)
+(c_1-c_3)\dot\tau_v.
}
\]

Ignoring the event-time derivative omits the last contribution and generally gives the wrong post-collision sensitivity.

Evaluate fixed times separated from \(\tau\) first. At or near an event, check the existence and equality of one-sided derivatives. A change in representation cardinality does not automatically imply failure of every weak derivative, but neither does it guarantee differentiability.

### 4.4 General event-time and reset derivatives

For a finite-dimensional front/event state \(z\), suppose an event satisfies

\[
h(z^-(\tau,\alpha),\alpha,\tau)=0.
\]

With pre-event tangent \(\eta^-\) at fixed time and flow \(f^-\),

\[
\dot\tau_v=-\frac{h_z\eta^-+h_\alpha v}
{h_zf^-+h_t}.
\]

For reset \(z^+=R(z^-,\alpha,\tau)\) and post-event vector field \(f^+\), the fixed-time tangent after reset is

\[
\boxed{
\eta^+=R_z\eta^-+R_\alpha v
+(R_zf^-+R_t-f^+)\dot\tau_v.
}
\]

This formula is a derivation for a transverse differentiable event/reset system. It is not a claim that every PDE event admits this finite-dimensional description. A small denominator flags grazing or degeneracy; do not clip it and call the result an exact derivative.

### 4.5 Smooth-region tangent equation

Where the base state is smooth,

\[
g_t+\partial_x\big(F_u(u;\alpha)g+F_\alpha(u;\alpha)v\big)=0.
\]

For a balance law, include the differentiated source and parameter/boundary terms. Coupling to moving shocks needs the correct interface relations. Evolving a tangent grid PDE and ignoring shock motion is not a reference method for the proposed task.

### 4.6 Temporal versus spatial singularities

A jump of an objective as a function of a parameter creates a measure in **parameter space**. A moving spatial discontinuity creates the spatial measure described here. They are different objects and require different queries and normalization. The first paper should not combine them under one ambiguous “measure derivative” label.

---

## 5. Weak metrics, labels, and error control

### 5.1 Bounded-Lipschitz distance

For finite signed measures on a bounded nondimensional domain, define

\[
d_{BL}(\mu,\nu)=
\sup_{\|\psi\|_\infty\le1,\;\operatorname{Lip}(\psi)\le1}
|\langle\mu-\nu,\psi\rangle|.
\]

Normalize the domain length first; otherwise the Lipschitz constraint changes physical meaning across examples. For a query with different amplitude/Lipschitz constants, scale the resulting bound accordingly.

For matched atoms,

\[
|\langle w\delta_s-\widehat w\delta_{\widehat s},\psi\rangle|
\le |w-\widehat w|\|\psi\|_\infty
+|\widehat w|\operatorname{Lip}(\psi)|s-\widehat s|.
\]

A useful upper bound adds diffuse \(L^1\) error and total variation of unmatched atoms. Matching is a diagnostic device; the actual measure comparison does not require a permanent atom identity.

### 5.2 Finite-query training is not the full weak norm

Train with a collection of smooth test functions, including constants, low-order polynomials, bounded trigonometric functions, and compact smooth bumps at several physical widths. Normalize amplitude and derivative consistently.

Use independent query families and locations for validation. If the model only matches a handful of training moments, very different measures can look identical. Include bounded total-variation regularization, held-out localized queries, and worst-case query search.

### 5.3 A useful one-dimensional LP diagnostic

For a finite signed atomic difference \(\sum_i m_i\delta_{x_i}\), sort the support and solve

\[
\max_{\psi_i}\sum_i m_i\psi_i,
\quad |\psi_i|\le1,
\quad |\psi_{i+1}-\psi_i|\le x_{i+1}-x_i.
\]

The feasible query set is symmetric, so maximizing the signed objective also captures the absolute supremum. Piecewise-linear extension gives an exact bounded-Lipschitz diagnostic for this discrete measure on the interval. A numerical LP solution still has solver tolerances; use primal/dual residual checks and appropriate labels rather than calling it a rigorous certificate by default.

For a diffuse component approximated by cell masses, add or estimate the quadrature/discretization error separately. Replacing the density with a discrete mass distribution changes the measure being tested. If exact cell masses are available, moving each cell's mass to its representative point costs at most the cell radius times that cell's total variation.

### 5.4 Finite-difference teacher validation

For a smooth query,

\[
D Q_\psi[v]\approx
\frac{Q_\psi(\alpha+hv)-Q_\psi(\alpha-hv)}{2h}.
\]

Sweep \(h\) and spatial resolution together. If each forward query has error at most \(\delta_Q\), the quotient error contributed by the two solves can be as large as \(\delta_Q/h\). A shrinking \(h\) with a fixed noisy solver can make labels worse.

Use exact or front-tracking teachers where available. For numerical teachers, require a refinement plateau and forward tolerances small enough relative to \(h\). Store the direction, perturbation size, event proximity, and reference uncertainty.

### 5.5 Do not use a vanishing smoothing width as unexamined ground truth

For a viscous regularization, derivatives describe that viscous numerical model. Their weak limit may agree with the inviscid target in a specified regime, but the limits in viscosity, grid spacing, timestep, and parameter perturbation need not be interchangeable. Perform a staged convergence study and label the regularization used.

---

## 6. Architecture: represent support, bulk variation, and directional linearity

### 6.1 Recommended first architecture: chart-consistent derivatives

Build a piecewise-smooth state chart

\[
\widehat u(x,t;\alpha)=
\sum_{r=0}^{J}\widehat u_r(x,t;\alpha)
\mathbf1_{s_r(t;\alpha)<x<s_{r+1}(t;\alpha)}.
\]

Predict the region states and shock positions. Compute derivatives of the smooth chart outputs with respect to \(\alpha\) or its direction, then assemble the measure analytically:

\[
\widehat g_v=D_\alpha\widehat u_r[v]
\quad\text{within region }r,
\qquad
\widehat w_{j,v}=(\widehat u_j^--\widehat u_j^+)D_\alpha\widehat s_j[v].
\]

Do not differentiate the hard indicator numerically and expect autograd to produce a Dirac measure. Autodiff computes chart derivatives; the distributional assembly is explicit.

This architecture guarantees internal consistency with the represented chart away from event boundaries. It does not establish that the chart approximates the true PDE well. Joint state and derivative validation remain necessary.

### 6.2 Input encoder

For 1-D regular grids, start with a modest convolutional/multiscale encoder. Inputs include initial state samples or coefficients, flux/source parameters, time, domain scaling, and boundary conditions. Use actual spacing and quadrature-aware pooling for resolution transfer.

For piecewise-constant development data, a front graph is a strong and more efficient input representation: region states, discontinuity locations, physical parameters, and a sparse adjacency graph. This is a legitimate baseline and may be sufficient.

Do not claim a neural operator merely because one tensor dimension is large. Demonstrate consistent physical evaluation across input resolutions and new query functions.

### 6.3 Output heads

Use three small heads:

1. **State-region head:** coefficients of a low-order continuous basis within each region, or coordinate-query outputs.
2. **Support head:** shock positions and base event metadata.
3. **Derivative representation:** chart JVPs, or a separately tested factored linear sensitivity head.

Suggested initial network widths are 64 or 128 with 3–4 blocks. Suggested maximum shock slots are 4, 8, or 16 for separate size buckets. These are proposed pilot defaults, not benchmark claims.

A global front-order representation matters. During fixed-cardinality training, positive gaps normalized to the domain can enforce ordering. Variable-cardinality cases should be bucketed or use an explicit cardinality head and masks. Sorting is piecewise differentiable and becomes nonsmooth at collisions; do not hide event behavior inside an unexplained sort operation.

### 6.4 Smooth-region basis

For controlled examples, constant or low-degree piecewise polynomials may suffice. For more complex smooth regions, use a coordinate network or a compact basis whose integrals against queries are available analytically or by stable quadrature.

Map each region to a reference interval but include the region-length derivative when differentiating the representation. Derivatives at fixed physical \(x\) are not the same as derivatives at fixed reference coordinate. Derive the chain rule for the chosen map and test it against direct physical-coordinate finite differences.

Avoid region-local bases that become ill-conditioned as a region width tends to zero. Detect collapsing regions and use the event update rather than amplifying \(1/\text{width}\) factors indefinitely.

### 6.5 Optional directly learned tangent head

For large parameter dimension, a direct factorization can be

\[
\mu_v=\sum_{k=1}^{r}(R_k(\alpha)^\top v)\,\nu_k(\alpha),
\]

where each \(\nu_k\) is a diffuse-plus-atomic measure. This makes directional linearity explicit and reduces storage from \(O(pN)\) toward \(O(r(N+p))\), but assumes useful derivative rank structure.

The base supports must be shared across directions. A generic nonlinear network receiving \(v\) can violate additivity. Test

\[
\mu_{v+w}\approx\mu_v+\mu_w,\quad
\mu_{cv}\approx c\mu_v,\quad
\mu_0=0.
\]

A direct head may not be integrable into the predicted state map. Require weak finite-difference consistency and, in regular parameter regions, small-loop tests of observable gradient integrals. Compare with the chart-derived head before adopting it.

### 6.6 State-conditioned versus initial-condition-only heads

Teacher-conditioned models may be much easier and useful for many sensitivities, but they require the forward state. Label that cost and information access. For a fair initial-condition-only comparison, all models must infer the state and shock support themselves.

### 6.7 Event module

Begin with exact classical event logic for the known scalar flux. Learn only after classical event calculations are a measured cost or generalization bottleneck.

An optional learned event module predicts event time, post-event chart initialization, and tangent reset parameters, subject to conservation and a validated derivative calculation. Always compare with the same state architecture plus exact event updates.

Near-event outputs should include a status and event-distance diagnostic. Predicted event probabilities are not a proof of differentiability. Keep one-sided outputs explicit when appropriate.

### 6.8 KANs are an optional parameterization

A KAN may replace a small coefficient/rate head after the representation works. It must use the same inputs and output semantics as the MLP and classical spline controls. Do not combine a new measure representation, new event handling, and a KAN change in one unablated comparison.

---

## 7. Losses and training curriculum

### 7.1 Query-pairing loss

For directions \(v_k\) and test functions \(\psi_\ell\),

\[
\mathcal L_{\mathrm{weak}}
=\frac1{KL}\sum_{k,\ell}
\left|
\langle\widehat\mu_{v_k},\psi_\ell\rangle
-q^{\mathrm{ref}}_{k\ell}
\right|^2.
\]

Use scales fixed from the training protocol. Do not normalize by a near-zero target derivative; report absolute errors and normalized errors with a declared floor.

A relative loss alone can ignore small but physically important gradients or explode at exact zeros. Include zero-gradient and zero-jump controls.

### 7.2 State and structure losses

A candidate objective is

\[
\mathcal L=
\lambda_q\mathcal L_{\mathrm{weak}}
+\lambda_u\mathcal L_{\mathrm{state}}
+\lambda_s\mathcal L_{\mathrm{support}}
+\lambda_w\mathcal L_{\mathrm{weights}}
+\lambda_c\mathcal L_{\mathrm{consistency}}
+\lambda_{TV}\mathcal R_{TV}.
\]

State loss can use integral, conservative cell-average, or weak norms rather than only pointwise values at discontinuities. Support and weight losses require matching or known event identities; use the measure loss as the representation-invariant primary target.

Total variation regularization discourages huge cancelling atom pairs and oscillatory diffuse/atomic compensation. It is a regularizer, not an exact physical law.

### 7.3 Finite-variation consistency

For a joint state/sensitivity model, compare

\[
\langle\widehat\mu_v,\psi\rangle
\quad\text{with}\quad
\frac{\widehat Q_\psi(\alpha+hv)-\widehat Q_\psi(\alpha-hv)}{2h}.
\]

Use perturbations that remain in the same regularity stratum for a two-sided derivative test. Also compare to the trusted state map, since two wrong learned heads can agree with each other.

Do not backpropagate through numerical discontinuity masks as a substitute for analytic measure assembly. For chart-derived sensitivities, weak sensitivity training requires mixed parameter/weight derivatives; profile their memory and validate them.

### 7.4 Direction and observable curricula

1. Train exact single-shock cases with a small parameter vector and analytic query integrals.
2. Add rarefactions and smooth solutions, where singular mass should vanish.
3. Add mixed diffuse and atomic sensitivities.
4. Add multiple noninteracting shocks.
5. Add collision cases with exact event derivatives and times separated from the event.
6. Add near-event one-sided or unresolved behavior.
7. Add numerical families and inverse/design tasks.

Use novel queries at every evaluation stage. Keep physically short-wavelength queries separate: their large Lipschitz constants make the sensitivity task harder, not “equally accurate” under the same weak norm.

### 7.5 Training budget

A proposed initial study uses three initialization seeds, a limited architecture grid, and equal wall-time or optimizer-budget tuning across methods. Begin around 10,000 optimization steps for a modest analytic dataset only if validation has not saturated; the number is a budget, not a scientific requirement. Use a smaller smoke run first.

Freeze query normalization, loss weights, checkpoint rule, and architecture search budget on validation. Do not select a different checkpoint for each test objective.

### 7.6 Efficient query accumulation

Compute a state/support representation once per parent, time, and direction. Evaluate query chunks without storing a tensor for all \(B\times N\times Q\) combinations. For basis-represented diffuse terms, contract precomputed query integrals directly.

When several chunk losses share an encoder graph, either accumulate the scalar loss before one backward pass or deliberately recompute/checkpoint the representation. Repeated `backward(retain_graph=True)` can retain unexpected memory and must be measured rather than assumed efficient.

---

## 8. Data, splits, and reference quality

### 8.1 Parent definition

A parent includes initial-condition family and coefficients, flux/source parameters, physical domain, boundary conditions, and forcing history. All resolutions, query functions, parameter directions, times, and finite-difference pairs from the same parent stay in one split.

An apparently new shock location drawn from the same held-out parent is not an independent test. An entire collision topology or flux family should be held out for the corresponding generalization claim.

### 8.2 Proposed first dataset

Generate exact analytic data on the fly or as immutable compact parameter manifests. A starting allocation can be 4,096 training parents, 512 validation parents, and a separately sealed test pool, with 4–8 directions and 8–16 queries sampled per training item. These are proposed compute allocations; determine confirmatory sample count from pilot variability.

Analytic data need not be stored as dense trajectories. Store parameters and reference formula/version. Numerical teachers need checkpoints, flux/discretization metadata, mesh refinement information, and uncertainty estimates.

### 8.3 Input and label separation

Inference input objects must not contain teacher shock positions or future event times unless the experiment is explicitly teacher-conditioned. Store labels in a separate record/class. Check accidental access with a unit test that removes labels before model invocation.

### 8.4 Reference tiers

| Tier | Source | Label claim |
|---|---|---|
| 0 | Analytic step/Riemann formulas | Exact mathematical formula, floating evaluation error recorded |
| 1 | Classical front tracking plus differentiated events | Reference under validated implementation and regime |
| 2 | Independently refined finite-volume/adjoint calculations | Numerical sensitivity with convergence/uncertainty evidence |
| 3 | Viscous/smoothed model | Sensitivity of that declared regularized problem |

Do not mix tiers without reporting them. Automatic differentiation of a particular discrete scheme is a reference for that discrete program, not necessarily for the entropy-solution derivative.

### 8.5 Metrics to store per example

```text
parent_id, split, parameter_vector, direction, direction_norm
physical_domain, observation_time, base_event_structure
reference_tier, solver_version, reference_resolution
fd_steps_and_errors, near_event_distance, derivative_status
state_error, diffuse_error, support_error, atom_weight_error
held_out_query_errors, weak_LP_diagnostic, TV_norms
nonlinear_objective_gradient_error, constraint_diagnostics
encoder_ms, direction_ms, query_ms, total_ms
peak_allocated_bytes, peak_reserved_bytes, host_RSS
```

Store raw query values and differences as well as averages. Wrong signs and occasional catastrophic event errors can disappear in mean-square summaries.

---

## 9. Software interfaces and unit tests

Suggested layout:

```text
singular_sensitivity/
  problems/        # scalar laws, parameter maps, boundaries
  reference/       # analytic Riemann, front tracking, event derivatives
  representation/  # diffuse charts, atoms, surfaces later
  queries/         # linear pairings and nonlinear objective formulas
  events/          # event graph, transverse derivatives, status handling
  models/          # state charts, encoders, tangent heads
  data/            # parent splits and numerical reference audit
  train/           # query losses, curricula, memory-aware loops
  validate/        # weak metrics, FD ladders, inverse tests
  benchmark/       # complete runtime and query-amortization studies
  cli.py
```

### 9.1 Portable output structure

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class MeasureBatch:
    diffuse_coefficients: object
    atom_positions: object
    atom_weights: object
    atom_mask: object
    left_traces: object
    right_traces: object
    position_jvps: object
    status: Literal["regular", "near_event", "one_sided", "unresolved"]
    coordinate_and_basis_metadata: dict
```

Use explicit tensor shapes in the implementation. Example: `positions[B,J]`, `weights[B,V,J]`, and `diffuse_coefficients[B,V,R,K]`. Do not require a dense `[B,V,N,Q]` product. Keep positions independent of derivative direction.

### 9.2 Query interfaces

```text
linear_pairing(measure, query) -> directional_derivative
nonlinear_objective_derivative(state_chart, tangent_chart, objective)
validate_direction_linearity(model, parent, directions)
compare_weak_finite_differences(parent, direction, query, h_values)
```

A generic `loss.backward()` over a rasterized step is not the nonlinear-objective implementation. Test the analytic moving-boundary term directly.

### 9.3 Required tests

- Translated increasing and decreasing step signs.
- Atomic weight does not receive an extra grid-spacing factor.
- Single-shock position and state-amplitude parameter derivatives.
- Rarefaction endpoints do not create false state-jump atoms.
- Constant query obeys the corresponding mass derivative with boundary/source terms.
- Nonlinear squared-error objective uses payoff jumps, not an arbitrary trace multiplier.
- Zero direction gives zero; scaling/additivity hold for regular directions.
- Two-shock post-collision derivative includes event-time sensitivity.
- One-sided tests near grazing/collision degeneracy are labeled correctly.
- Grid refinement preserves analytic atom pairings.
- Permuting atom slots does not change the represented measure.
- Collapsing zero-jump atoms has negligible effect.
- Signed positive/negative masses are retained without probability normalization.
- Numerical quadrature converges for diffuse terms and smooth queries.
- Model/parameter state survives checkpoint/resume without silently changing the direction convention.

### 9.4 Mixed derivative tests

For the chart-derived architecture, validate the gradient of a weak derivative loss with respect to network weights on a tiny FP64 case. Check it with centered finite differences of the scalar loss. This catches detached JVPs, missing region-map derivatives, and unsupported higher-order autograd paths.

Official `torch.func` JVP/VJP facilities can support these operations, but the installed version and individual operation coverage must be tested. Prefer pure functions; isolate in-place updates and external solver calls from the differentiated graph [H6].

---

## 10. Work packages and experiment gates

### WP0 — Target and semantics

Freeze scalar PDE class, parameterization, direction norm, time region, entropy selection, query class, and derivative status policy. Write the source/novelty ledger. Decide teacher-conditioned versus initial-condition-only inference.

### WP1 — Exact reference and query engine

Implement the translated step, one shock, rarefaction, and two-shock collision cases. Implement linear and nonlinear query derivatives. Run the required tests before any learning.

### WP2 — Representation separation

Compare exact measure evaluation against grid difference quotients across resolution and parameter-step ladders. Demonstrate both where weak pairings behave well and where the chosen derivative is undefined. This is a diagnostic, not yet a learned-method result.

### WP3 — Minimal learned chart

Train on single-shock and mixed smooth/shock parents. Compare field-only state/sensitivity models, chart autodiff, classical front sensitivities, and the proposed explicit measure query head.

### WP4 — Generalization beyond interpolation

Hold out query families, grid resolutions, parameter ranges, and multi-shock arrangements. Use exact event logic first. Test state/sensitivity consistency and directional linearity.

### WP5 — Events and inverse tasks

Add collision derivatives and declared one-sided cases. Evaluate parameter estimation or design with smooth integral objectives. Re-evaluate final candidates using the trusted entropy solver and objective—not the learned surrogate alone.

### WP6 — Numerical and higher-dimensional extension

Only after a benefit exists, use more complex scalar balance laws or 2-D interface-supported measures. Audit which assumptions survive. Do not claim general systems theory from scalar experiments.

### WP7 — Complete performance and publication audit

Measure amortization over queries/directions, host and GPU cost, teacher cost, transfer, and failure rates. Audit direct neural shock-adjoint and shape-sensitivity literature before asserting novelty.

### Proposed practical gates

- Analytic weak derivatives and nonlinear objective formulas pass at the declared floating tolerance.
- No systematic loss of direction linearity or forward/tangent consistency in regular regimes.
- Improvement over a matched classical front/adjoint baseline in a nontrivial use case, not only over rasterized autograd.
- At least a material reduction in weak derivative error or 20% lower complete cost at the required downstream tolerance on held-out parents. The 20% value is a proposed practical target, not an expected outcome.
- Generalization to new queries and resolution changes without changing physical smoothing widths to hide errors.

---

## 11. Baselines, statistical analysis, and inverse-problem use

### 11.1 Mandatory comparisons

1. Exact/classical front sensitivity where available.
2. A classical front representation with interpolation/regression of coefficients.
3. Autodiff through an ordinary learned state operator, evaluated weakly rather than unfairly only in pointwise norm.
4. Direct smooth-grid sensitivity prediction.
5. A state-chart model with analytic derivative assembly but no special derivative training.
6. The proposed measure model with weak/sensitivity training.
7. Viscous/smoothed sensitivity with an explicit regularization study.

Match access to forward states, event metadata, tuning budget, and reference fidelity. An easy analytic Burgers formula is expected to beat a neural model on its own toy family; it supplies correctness tests, not the main performance claim.

### 11.2 Query and direction amortization

Separate costs:

\[
C_{\mathrm{total}}=C_{\mathrm{state}}+C_{\mathrm{representation}}
+D C_{\mathrm{direction}}+Q C_{\mathrm{query}},
\]

with a more detailed product cost when all direction–query pairs are needed. An adjoint may already answer one scalar objective cheaply regardless of parameter count. Compare the correct workload: many objectives, many directions, or repeated parameter instances.

Do not report a speedup for a thousand queries when the application needs only one, without reporting both regimes.

### 11.3 Nonlinear inverse/design task

Use a smooth tracking objective or sensor-kernel objective with explicit parameter bounds. If the objective is nonlinear in the field, use the payoff-jump formula from §3.5.

Freeze initializations and optimization budgets. Monitor descent using trusted objective reevaluation. If event crossings produce nondifferentiability, use a declared line search, one-sided strategy, or fallback rather than silently clipping bad gradients.

Report final trusted objective, recovered parameters when identifiable, number of trusted solves, query errors, optimization failures, and total time. A low learned objective is not an independent validation.

### 11.4 Statistics

Use physical parents as independent units. Aggregate times, shocks, directions, and queries within a parent or use a declared hierarchical analysis. Report model-seed variability separately. Use paired log-cost differences for positive runtimes and paired error differences at a predeclared tolerance.

Give quantiles and worst observed strata, not only global means. Near-event and zero-gradient cases deserve separate tables. Final configuration selection uses validation only.

---

## 12. Main weaknesses and how to address them

| Weakness | Why it matters | Remedy | Stop/narrow condition |
|---|---|---|---|
| Missing singular term | Smooth sensitivity can misrepresent moving shocks | Explicit support/weight output and weak tests | Measure representation gives no benefit over proper classical baseline |
| Incorrect nonlinear chain rule | Wrong optimization gradients despite good linear queries | Return traces and shock motion; use payoff jumps | Objective outside supported regularity class |
| Genuine nondifferentiability | A two-sided target may not exist | Event statuses, one-sided analysis, scoped claims | Claimed operator lacks a defined target |
| Wrong event-time derivative | Post-collision gradients drift | Analytic event calculus and exact multi-shock tests | Model cannot handle even transverse events reliably |
| Direction nonlinearity | Output not a derivative operator | Chart JVP or linear-in-direction factorization | Persistent additivity violations |
| Nonintegrable tangent head | Predicted gradients not tied to any state map | Weak finite differences and loop tests; prefer chart derivatives | Inconsistent objective optimization |
| Fake atom/diffuse cancellation | Training moments do not identify representation | TV control, richer held-out queries, state coupling | Large hidden cancellations remain |
| Poor atom matching | Labels swap around collisions | Measure loss primary; event-aware matching diagnostics | Matching dominates the claimed accuracy |
| Grid-label contamination | FD error explodes as h shrinks | Exact teachers, coupled grid/h refinement | Reference uncertainty exceeds target gain |
| Small jump or collapsing region | Parameterization becomes ill-conditioned | Zero-jump controls, status handling, event merge | Apparent gains require arbitrary clipping |
| Spectral/global input bottleneck | Finite receptive field misses influencing features | Audit observation support; multiscale inputs | Local architecture lacks necessary information |
| Classical method is already cheap | Neural overhead erases benefit | Many-query workload, harder validated family | No complete-cost advantage |
| Nonlinear system extension invalid | Scalar theory does not transfer automatically | Add one rigorously scoped system at a time | No trustworthy reference sensitivity |
| Signed measure treated as probability | Loss/object interpretation is wrong | Signed weak metrics and TV accounting | Cannot define meaningful query metric |
| 2-D surface cost explosion | Surface geometry and quadrature dominate | Delay extension; compact patches with validated quadrature | Surface approximation cannot meet tolerance affordably |

---

## 13. Efficiency specific to this architecture

### 13.1 Do not allocate dense Jacobians by default

For a field with \(N\) values and \(p\) parameters, a dense FP32 Jacobian needs \(4Np\) bytes before autograd storage. Use JVPs for a few directions and VJPs/adjoints for a few objectives. A low-rank tangent head requires an empirical rank study; it is not a guaranteed compression.

### 13.2 Query-first evaluation

For polynomial/spline diffuse charts, cache basis integrals against a query bank. Evaluate atoms at their continuous coordinates directly. Use batched contractions to compute many pairings without materializing per-grid/per-query tensors.

For nonlinear objectives, evaluate smooth-region quadrature and payoff jumps. Do not rasterize a narrow mollifier for each atom unless it is an explicitly labeled baseline or visualization path.

### 13.3 Bucket rather than compile every shape

Use a small number of resolution, region-count, and shock-count buckets, with masks for padding. Recompiling for every front count can erase gains. Start with eager FP32; benchmark compiled static buckets only after math tests pass.

GPU graph capture is poorly suited to Python event queues and dynamically growing front lists. Keep a control plane on CPU and a batched numeric plane on GPU, or use bounded static representations where meaningful. Do not force invalid fixed topology merely to capture a graph.

### 13.4 Mixed precision

The encoder may use BF16 after validation. Keep coordinates, small displacement differences, atom weights, event denominators, weak reductions, and derivative checks in FP32 or FP64 as required.

BF16 can quantize nearby positions to the same value over some scales. A model with accurate-looking fields can then have incorrect position derivatives. Test a displacement ladder before enabling low precision in support heads.

For the tiny exact and mixed-derivative tests use FP64 with reduced-precision matrix multiplication disabled. Numerical testing in FP64 is not a rigor certificate for arbitrary PDEs.

### 13.5 Teacher generation and storage

Analytic/front-tracking data are compact parameter records. Generate query values on demand or in consolidated shards. Cache expensive teacher states and event derivatives on persistent storage with manifests; use scratch only for regenerable working copies.

Use CPU job arrays for independent front/reference problems when that is more efficient than reserving a GPU. GPU utilization is not the scientific objective; reliable throughput per allocated resource is.


---

## 14. Compute and deployment

### A100 40 GB engineering plan

This is a proposed execution budget, not a measured fit or throughput claim. Current NVIDIA documentation describes Ampere's tensor-core formats and memory architecture; the actual job must record the allocated model, total memory, driver, CUDA runtime, and power/clock conditions rather than assuming a specific A100 variant's sustained performance [H1].

#### Hardware and memory contract

Query `torch.cuda.get_device_properties(0).total_memory` and `nvidia-smi` inside the allocation. Manufacturer GB and binary GiB are different. A Slurm `--mem=64G` request is **host RAM**, not VRAM. Detect unexpected GPU model, partitioning/MIG configuration, or device count and stop or explicitly revise the benchmark protocol; do not silently compare heterogeneous hardware.

Use a development peak-reserved-memory target of at most **80% of the allocated device's reported total memory**. Treat this as a safety budget, not a guarantee. Measure both PyTorch allocated/reserved peaks and whole-process GPU usage where possible; external libraries can allocate outside the PyTorch allocator [H5].

A conservative planning ledger is:

| Component | Initial policy |
|---|---|
| Parameters, gradients, optimizer | Target below 2 GiB for the first small models |
| Activations and derivative graphs | Largest adjustable pool; profile one real forward/backward |
| Inputs, query/certificate tensors | Stream or chunk; never preload the campaign on device |
| Compiler/FFT/sparse workspaces | Reserve explicitly; measure after warmup |
| Non-PyTorch buffers and fragmentation | Leave headroom rather than consuming the nominal limit |
| Persistent checkpoints/datasets | Host/persistent filesystem, not GPU memory |

For ordinary FP32 Adam training, a useful first approximation is about \(16P\) bytes for parameters, gradients, and two optimizer-state tensors, before activations and implementation temporaries. Mixed precision changes parts of the ledger but does not remove all FP32 state. Optimizer implementations can have additional peak allocations. Profile rather than infer capacity from parameter count.

Do not use giant global attention for a local-grid task without a measured reason. Attention-logit storage scales quadratically with tokens. A small multiscale encoder, sparse graph, or compact global token set is the initial design.

#### Architecture-specific memory hazards

**Candidate A:** exact symbolic algebra and interval trees may consume host RAM while the GPU is nearly empty. An SOS polynomial of degree \(2d\) in \(n\) variables has a dense monomial basis of size \(\binom{n+d}{d}\); a dense Gram matrix scales with its square. Low variable count, sparse structure, bounded-degree templates, and CPU verification are essential. A large GPU does not fix combinatorial certificate growth.

**Candidate B:** never materialize a full parameter-by-grid derivative for convenience. Use JVPs, VJPs, chart derivatives, or a justified low-rank form. Avoid a batch-by-grid-by-query tensor when contractions can stream over queries. Support positions, small shifts, and cancellation-sensitive pairings require higher precision than a generic encoder activation.

#### Optimization order

1. Establish a deterministic small FP64 reference for the mathematical core.
2. Build a correct eager FP32 training path and validate gradients.
3. Profile input staging, encoder, derivative/candidate formation, reduction, and backward separately.
4. Enable BF16 autocast in the neural encoder only; retain critical calculations at FP32/FP64 and revalidate.
5. Use static shape buckets and benchmark `torch.compile` on the numerical neural blocks.
6. Add activation checkpointing only if activation memory, rather than data or an algorithmic bottleneck, dominates.
7. Fuse kernels or write Triton/CUDA only for an identified dominant operation with a tested reference implementation.
8. Re-run end-to-end accuracy and gradient/certificate checks after each optimization.

PyTorch documents operation-specific mixed-precision behavior. Do not cast the entire pipeline to BF16 by default [H2]. The compiler's `reduce-overhead` mode can use CUDA graphs and additional workspace memory; it is neither universally faster nor universally applicable [H3]. PyTorch recommends explicit `use_reentrant=False` for checkpointing; stateful recomputation that differs from the forward pass can produce incorrect gradients [H4].

Pin the installed versions after a passing smoke test. The version displayed by an online documentation page is not evidence of the version available at CARC. Keep an eager fallback and record compilation startup separately from steady-state timing.

#### Candidate A throughput pipeline

Run batched proposal generation on GPU and verification in bounded CPU workers. Use `spawn` or clean subprocesses rather than forking a CUDA-initialized process. Start with a small worker count within `SLURM_CPUS_PER_TASK`; benchmark 1, 2, and 4 workers instead of spawning one per candidate without limit. Cap thread counts inside BLAS/SDP libraries to avoid oversubscription.

Apply backpressure: a finite candidate queue, maximum certificate byte size, per-check memory budget, and timeout. Rejecting an unverified certificate at timeout leaves the task unresolved. A timeout does not create an accepted negative answer.

If checking dominates and proposal batches are infrequent, export immutable candidates and release the GPU. Run checking as a CPU batch stage. Higher GPU utilization is not necessarily more efficient or cheaper.

#### Candidate B throughput pipeline

Compute the base representation once, then stream directions and query blocks. Prefer analytic basis-query contractions for polynomial pieces and direct evaluation at atom coordinates. Keep the event/control plane separate from static numeric batches.

Mixed-derivative losses can be expensive even when the forward encoder is small. Test one actual gradient-bearing batch before estimating memory. Shorten the derivative unroll or use analytic chart tangent formulas before shrinking the physical problem indiscriminately.

#### Runtime measurement

Use warmup, explicit synchronization around GPU timing boundaries, repeated paired runs, and equal scientific tolerances. Report cold-start compilation separately. Record wall time for the entire requested task, not just the neural forward pass.

For A, the endpoint includes candidate generation, rationalization, failed attempts, verification, and fallback. For B, it includes the forward state where required, support reconstruction, derivative work, and the requested query workload. Baselines receive the same information and resource budget.

A speed claim requires actual measurements on the declared hardware. Analytical operation counts and memory estimates are planning aids only.

### USC CARC and Slurm: regular-user deployment

#### Verify access before using templates

CARC documents typed GPU requests and an `a100-40gb` feature constraint. Its running-jobs guide uses `myaccount` to inspect authorized associations. Those pages do not establish your specific account permissions or current availability [H7,H8].

Run read-only discovery in the authorized login environment:

```bash
myaccount
myquota
sinfo -o '%P %a %l %D %G %f'
scontrol show partition gpu
module avail conda 2>&1
module avail cuda 2>&1
```

Choose writable existing project, scratch, and environment paths from live account information. Do not invent a project allocation name. Do not install drivers, use `sudo`, assume Docker privileges, or start heavy training on a login node. Follow the site's permitted environment-installation route.

CARC documents `/home1`, `/project2`, and `/scratch1` storage. Scratch is temporary and not backed up. Its `/tmp` and `/dev/shm` are RAM-backed, so copying a large dataset there consumes host memory rather than presumed NVMe capacity [H9]. Use confirmed scratch for regenerable temporary shards and compiler caches, and persistent storage plus an additional backup/regeneration route for essential artifacts.

CARC's Conda guide documents module-based user environments. Inspect the available module and installed CUDA/PyTorch compatibility rather than copying an old package command uncritically [H10]. A direct path to a tested environment's Python executable makes batch behavior explicit.

#### Submission templates in the bundle

`slurm/train_a100.sbatch` requests one task, one A100 40 GB, eight CPUs, 64 GB host RAM, and four hours as **proposed starting resources**. Verify partition/account limits before use. `slurm/submit.sh` is dry-run by default and checks that the project, Python executable, entry point, config, scratch, and output roots exist.

The templates target a training program you must implement. That entry point must support:

```text
--config <absolute YAML path>
--run-dir <absolute output directory>
```

Do not call these files a runnable research pipeline before that implementation exists. The scripts have no embedded private account information and do not submit automatically.

Slurm reads `#SBATCH` directives before the shell executes; shell variables there are literal. The wrapper passes variable paths and account options on the `sbatch` command line and creates log directories before submission [H11].

#### Required environment variables

```bash
export PROJECT_ROOT="/absolute/existing/repository"
export SCRATCH_ROOT="/absolute/existing/authorized/scratch"
export RUN_ROOT="/absolute/existing/persistent/run_root"
export PYTHON_EXE="/absolute/existing/environment/bin/python"
export ENTRYPOINT="/absolute/existing/repository/train.py"
export CONFIG="/absolute/existing/implemented_config.yaml"
export CARC_ACCOUNT=""  # set an authorized account when required

bash slurm/submit.sh            # print the reviewed command only
bash slurm/submit.sh --submit   # explicit submission after implementation/tests
```

The displayed paths are placeholders, not discovered paths for your account. The included `configs/*_PLAN.yaml` files are planning manifests and are intentionally rejected by the wrapper. Convert them into validated implementation configs first.

#### Pre-time-limit checkpoints

The template requests `--signal=USR1@180` and starts the process with `srun`. Implement and test a Python signal handler that sets a flag only; checkpoint at the next safe point. Slurm's signal timing and propagation must be tested in a short allocation before relying on it. Do not add `B:` unless also changing the signal-forwarding design; batch-shell-only signaling is a different contract [H11].

```python
import signal
checkpoint_requested = False

def request_checkpoint(signum, frame):
    global checkpoint_requested
    checkpoint_requested = True

signal.signal(signal.SIGUSR1, request_checkpoint)
signal.signal(signal.SIGTERM, request_checkpoint)
# Main loop: at a safe boundary, serialize a consistent checkpoint and exit.
```

A regular periodic checkpoint remains necessary because abrupt node failure may not deliver a useful signal. Save model, optimizer, random states, split manifest, sampler position, and proposal/event queues as applicable. For A, also save unresolved coverage trees and certificate statuses. For B, preserve event/chart metadata and direction conventions.

Write a temporary file in the same persistent directory, flush/close it, then atomically replace the target. Keep a previous known-good checkpoint. A rename across filesystems is not an atomic checkpoint protocol. Verify loading in a fresh process; never trust untrusted pickle files.

#### Job completion and accounting

Capture actual job IDs returned by `sbatch`. Check scheduler state and recorded exit codes; a printed command is not a submission. Archive run manifests, package versions, GPU properties, config hashes, git state, elapsed time, peak memory, and failure reason. Use site-supported accounting commands and do not assume a particular optional Slurm plugin is installed.

Use CPU jobs for exact arithmetic and small references when appropriate. A multi-node SC performance claim requires a separate actual distributed experiment. One-GPU timing establishes only the one-GPU result.

---

## 15. Deliverables and the first executable audit

### 15.1 Required research artifacts

Deliver the scoped mathematical target, analytic and numerical teacher audits, state/measure schemas, linear and nonlinear query engines, classical front-sensitivity baseline, learned architecture, event/status policy, held-out parent/query splits, and full-cost evaluation.

The final paper should demonstrate more than shock detection. It should establish a benefit from the derivative representation, valid transfer to new queries or resolutions, internal state/tangent consistency, and a useful downstream task. A classical method that already returns the same measure is the benchmark to beat or complement.

### 15.2 Worked correctness example

Take Burgers states \(u_L=2\), \(u_R=-1/2\), location \(a=0.1\), time \(t=0.6\), and direction \(v=(0.2,-0.3,0.4)\). The shock lies at

\[
s=0.1+\tfrac12(2-0.5)0.6=0.55,
\]

and its directional displacement is

\[
\dot s_v=0.4+\tfrac{0.6}{2}(0.2-0.3)=0.37.
\]

The atom weight is \((2-(-0.5))0.37=0.925\). The diffuse density is \(0.2\) to the left and \(-0.3\) to the right. Query this representation directly; do not smear the 0.925 mass over a grid cell without including the resulting approximation in the evaluation.

For a squared tracking objective, compute the objective-density jump independently. This example is included in the audit and detects the incorrect choice of a left-trace pointwise chain rule.

### 15.3 Included support code and status

From the extracted bundle root:

```bash
python support/math_audit.py --output support/math_audit_results.json
python support/torch_math_audit.py --output support/torch_math_audit_results.json
```

The first script uses only the standard library. Its eleven groups cover the A certificate mechanics and B analytic step, shock, direction-linearity, nonlinear-objective, collision-time, and atomic-bound checks. All passed in the preparation environment.

The optional PyTorch script checks that a smooth chart query's JVP equals the explicit measure formula, that a mixed derivative loss passes a tiny FP64 gradient check, and that a batched atomic contraction matches direct summation. Its three checks passed on CPU. None is a neural training or PDE performance result.

The mathematical tests are necessary implementation seeds, not a validation of the whole proposed architecture. Full reference/teacher convergence, event handling, and application experiments remain to be implemented.

### 15.4 Immediate work order

Implement the analytic reference/query engine, then test the weak representation on different raster resolutions. Build the smallest chart-consistent neural baseline next. Do not add learned events, a large backbone, a KAN head, or 2-D surfaces until the current stage has passed its classical and consistency controls.

## 16. Novelty boundaries and primary references

**Source provenance:** Candidate B in the supplied `FRONTIER_PROPOSALS_AND_NOVELTY_AUDIT.md` proposes the signed-measure target and weak-observable evaluation. This guide expands its implementation, including the essential nonlinear-payoff and event-time caveats; those additions are derivations and design instructions, not new reported experiments.

- **[B1] Bianchini.** [On the Shift Differentiability of the Flow Generated by a Hyperbolic System of Conservation Laws](https://doi.org/10.3934/dcds.2000.6.329), DCDS 6(2), 329–350, 2000. Foundational shift-tangent theory.
- **[B2] Ulbrich.** [A Sensitivity and Adjoint Calculus for Discontinuous Solutions of Hyperbolic Conservation Laws with Source Terms](https://doi.org/10.1137/S0363012900370764), SIAM J. Control Optim. 41(3), 740–797, 2002. Classical shock sensitivity and adjoint results; cite the volume year rather than confusing a later online timestamp with the work's origin.
- **[B3] Ketcheson, LeVeque, and del Razo / Clawpack Riemann materials.** [Burgers' equation](https://www.clawpack.org/riemann_book/html/Burgers.html). Primary computational reference for the Riemann and interaction controls used here.

The jump-derivative, payoff-jump, and event-time formulas are derived or specialized explicitly in this guide. They are not claimed as new calculus. The proposed learning contribution is a reusable, resolution-consistent, computationally advantageous sensitivity object with event-aware semantics. A focused search for neural shock adjoints, shape sensitivities, and measure-output learning remains required before any priority claim.

### Official implementation and deployment references

The following documentation was checked during preparation. Verify behavior against the installed versions and live CARC allocation before a campaign; these links do not establish account authorization or measured performance.

- **[H1] NVIDIA.** [Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html). Architecture and optimization constraints, not a workload speed prediction.
- **[H2] PyTorch.** [Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/accelerator/amp.html). Operation-specific precision behavior.
- **[H3] PyTorch.** [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile). Compiler modes and possible memory overhead.
- **[H4] PyTorch.** [Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint). Explicit non-reentrant configuration and recomputation caveats.
- **[H5] PyTorch.** [CUDA semantics and memory management](https://docs.pytorch.org/docs/stable/notes/cuda). Allocated versus reserved memory and allocator behavior.
- **[H6] PyTorch.** [`torch.func.jvp`](https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html). Forward-mode differentiation interface and operation-coverage caveat.
- **[H7] USC CARC.** [GPU Programming](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming.html). Typed GPU request and A100 memory-variant constraints.
- **[H8] USC CARC.** [Running Jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs). Slurm submission and account-association checks.
- **[H9] USC CARC.** [Storage File Systems](https://www.carc.usc.edu/user-guides/research-data-management/storage-file-systems). Current documented storage names, scratch policy, and RAM-backed temporary storage.
- **[H10] USC CARC.** [Using Conda](https://www.carc.usc.edu/user-guides/hpc-systems/software/conda). User-managed environments through the site module system.
- **[H11] SchedMD.** [`sbatch` manual](https://slurm.schedmd.com/sbatch.html). Directive parsing, signals, resource options, and job submission semantics.

Do not infer that a supported command or a permitted request is available to every account. Record the actual site/module/runtime behavior during the preflight.

---

**Final status:** implementation instructions and tested small support identities only. Research efficacy, runtime gains, full-model correctness, and venue suitability remain experimental questions.
