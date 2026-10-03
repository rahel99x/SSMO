# Frozen target and evidence ledger

The user supplied the proposal and CARC operating requirements in this chat.
They authorize implementation. Document instructions are technical requirements
within that request; copied preparation-status statements are not execution
evidence for this repository. User requirements override the proposal's generic
Conda/scratch examples: use standalone Python with `.venv`, and store all
temporary data and caches inside `/home1/aadaniel/projects/SSMO`, following
the user's latest CARC storage instruction.

## Target

- Scalar Burgers, fixed convex flux `u²/2`, entropy Riemann selection.
- Fixed physical interval `[-2,2]`; constant exterior states; no boundary
  interactions in selected samples.
- Parameters `(uL,uR,a)` or `(uL,uM,uR,a,b)` for reference collision controls.
- Fixed observation time; Euclidean directions, including an explicit zero
  direction. Jumps use `[u]=u_right-u_left`.
- Finite signed Radon measures queried by smooth normalized functions; nonlinear
  objectives require base traces and shock motion.
- Initial-data-only learned inference. Exact event metadata never enters the
  single-front model. Analytic future supports are training labels only.

## Evidence and progression

| Work package | Implemented capability | Gate to the next step |
|---|---|---|
| WP0 | Frozen semantics, inference access, source ledger, event statuses | Audit all sign and objective contracts |
| WP1 | Exact Riemann/constant/collision references, query and event calculus; Godunov forward/FD audit | FP64 formulas, mixed gradients and refinement checks |
| WP2 | Atom query invariance and grid/step/smoothing diagnostics | Compare weak errors without narrowing a physical smoothing width to hide failure |
| WP3 | Small chart, state-only chart control, explicit measure training, classical front regression and weak grid controls | Valid state/tangent finite variation and direction linearity on held-out parents |
| WP4 | New query bank, range holdouts, multiple evaluation grid resolutions | Learned multiple-front and collision topology holdouts remain unimplemented |
| WP5 | Exact differentiated collision reference; bounded smooth inverse task with trusted objective acceptance | Learned event/inverse efficacy remains unestablished |
| WP6 | Bounded Godunov audit only | No numerical-label learning/2-D/system expansion until nontrivial advantage and converged teacher quality |
| WP7 | Complete pilot cost/error records and hardware provenance | Actual CARC measurements, three separately analyzed seeds, novelty review and downstream tolerance/cost study |

Mathematical identities and front sensitivity calculus are classical, not a
novelty claim. The polynomial front regression spans the single-shock exact
solution; beating a naive grid baseline on this toy family would not satisfy
the proposal's nontrivial classical-control gate. All seven proposed comparison
categories are represented as first-stage controls, but matched optimizer/search
budgets and broader numerical regimes remain requirements of a confirmatory study.

The numerical solver computes a declared first-order conservative discretization.
Refinement and finite-difference errors are observed against exact queries.
Successive-resolution differences are not certified uncertainty bounds. No
viscous or smooth-grid derivative is called the inviscid reference without a
limit study. The smoothed shock ladder is labeled a tanh-regularized state chart,
not a viscous Burgers PDE solve.

Eager FP32 is the only training path. FP64 is used in exact, FD and mixed-gradient
tests. No BF16, compiler, giant encoder, KAN, Triton kernel or dense campaign is
enabled by default. A GPU pilot expands only after a real gradient-bearing smoke
has measured runtime and memory. Stop or narrow the study if the representation
has no material held-out error/cost benefit at a predeclared downstream tolerance.
