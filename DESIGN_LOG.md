# PushT-PO — Design & Implementation Log

This file is the running record for building the environment specified in `SPEC.md`.
It holds the phased plan, decisions/deviations from the spec, and a dated log of
progress. Update it as work happens — check off phases, append log entries, record
any deviation from `SPEC.md` with a reason.

Conventions for entries below:
- Phase checkboxes reflect current status only; don't rewrite history in them.
- The dated log is append-only, most recent last.
- "Deviation" entries must state what SPEC.md says, what we did instead, and why.

---

## 0. Project mapping

SPEC.md's file layout (§15) assumes a package named `pusht_po/`. This repo was
already scaffolded by `uv` with package `src/push_me/`. Decision: **keep
`push_me` as the package name**, do not rename to match the spec — map spec
module names 1:1 into `src/push_me/`:

| Spec module | Actual path |
|---|---|
| `pusht_po/__init__.py` | `src/push_me/__init__.py` |
| `pusht_po/config.py` | `src/push_me/config.py` |
| `pusht_po/shapes.py` | `src/push_me/shapes.py` |
| `pusht_po/geometry.py` | `src/push_me/geometry.py` |
| `pusht_po/goals.py` | `src/push_me/goals.py` |
| `pusht_po/env.py` | `src/push_me/env.py` |
| `pusht_po/lidar.py` | `src/push_me/lidar.py` |
| `pusht_po/render.py` | `src/push_me/render.py` |
| `pusht_po/vec.py` | `src/push_me/vec.py` |
| `pusht_po/tests/*` | `tests/*` (top-level, sibling to `src/` — not nested inside the package; see §3 deviation) |
| `scripts/*` | `scripts/*` (unchanged) |

Gymnasium env IDs (e.g. `PushTPO-Full-Single-v0`) are registration strings, not
paths — those stay exactly as specified.

**Arena coordinate convention (decided in Phase 4, binding for Phase 6/10):**
SPEC.md never states where the origin sits. Adopted: the arena spans
`[0, arena_size] × [0, arena_size]` (origin at a corner), matching §12's
pygame canvas (`arena at 512×512`) and the original PushT's convention,
rather than centering it at `[-arena_size/2, arena_size/2]`. `goals.py`'s
rejection sampling (`sample_goal_rects`) already samples poses in
`[0, arena_size)` on this basis — `env.py` (Phase 6) and `render.py` (Phase
10) must place walls/pusher/objects consistently with it.

Dependencies to add via `uv add` when each phase needs them (not all up front):
`pymunk`, `gymnasium`, `numpy`, `scipy` (Hungarian algorithm), `pygame` (render,
optional/lazy import per §12), `zarr` or `h5py` (demo storage), `pytest` +
`hypothesis` (dev).

---

## 1. Testing strategy

Four tiers, not one undifferentiated "tests" bucket. Pick the tier by what the
spec is actually claiming: a universal statement ("all n symmetric
orientations", "matches brute force for k≤5", "bit-identical for any seed")
gets a property test, not a hand-picked example standing in for it.

**Test-first within a phase, wherever the API is pinned down.** Where
`SPEC.md` already gives an exact signature, dataclass shape, or worked example
— `contains()`, `containment_error()`, `ShapeDef`, `GoalRect`, the `info`
dict schema, the reward formula — write that phase's test file(s) before the
implementation, then implement to make them pass. Where the API is still
being discovered as the phase is built (e.g. the internal helper structure
inside `env.py`'s pymunk wiring), it's fine to implement first and backfill —
don't force test-first onto exploratory work with nothing pinned down yet to
assert against. Each phase bullet in §2 notes which case it is.

### 1.1 Tiers

1. **Property-based (`hypothesis`)** — for anything with a universal
   quantifier or an invariant that should hold across a generated space of
   inputs. This covers most of the geometry/goals/assignment surface, because
   SPEC.md's acceptance criteria are themselves phrased as universal claims.
2. **Example-based (`pytest`)** — known-answer cases, edge cases
   (axis-aligned rect, zero margin, degenerate/collinear outlines), anything
   cheaper to hardcode than to generate.
3. **Determinism / golden-trajectory** — fixed seed + fixed action sequence →
   recorded `(obs, reward, terminated, truncated, info)` sequence, replayed
   and compared bit-exact per SPEC.md §14. This pins pymunk's actual behavior;
   it isn't a property to re-derive, it's a regression fence.
4. **Visual snapshot** — pixel-level regression for `render.py`, starting
   Phase 10. See §1.3.

### 1.2 Property-based test catalog

Add `hypothesis` as a dev dependency in Phase 0. `conftest.py` registers a
settings profile: pymunk-touching properties are slow per-example, cap those
at `max_examples=50, deadline=None`; pure-numpy geometry properties run the
default 100+.

| Property | Module under test | Strategy sketch | Test file |
|---|---|---|---|
| Rotating a shape's outline by `2π/symmetry_order` maps the vertex set onto itself within tolerance, for all registered shapes | `shapes.py` | `st.sampled_from(SHAPES)` | `test_shapes.py` |
| `make_shape(name, area).outline` has area == `area` within tolerance, for any positive area | `shapes.py` | `st.sampled_from(SHAPES.keys())`, `st.floats(min=1, max=1e6)` | `test_shapes.py` |
| Convex decomposition parts are each individually convex, and their union area equals the outline area | `shapes.py` | same | `test_shapes.py` |
| `containment_error(rect, outline) <= 0` iff `contains(rect, outline)` is `True` | `geometry.py` | random rect (center/angle/half-extents) + random outline | `test_containment.py` |
| Containment is monotonic in margin: contained at margin `m` ⇒ contained at margin `m' > m` | `geometry.py` | random rect/outline + ordered margin pair | `test_containment.py` |
| `minAreaRect` is invariant to the outline's starting vertex / winding restated CCW | `geometry.py` | permute vertex array | `test_containment.py` |
| All `n = symmetry_order` rotations of a shape about its own centroid, placed at a goal rect, are contained with *equal* containment error (within tolerance) | `goals.py` + `shapes.py` | `st.sampled_from(SHAPES)`, random rect | `test_modes.py` — this generalises acceptance test #1 from "hexagon at margin 24" to all shapes/margins |
| Hungarian assignment cost ≤ every permutation's cost, for random `k×k` cost matrices, `k` ≤ 5 | `goals.py` | `st.integers(1,5)` for k, random cost matrix | `test_assignment.py`, oracle = brute force via `itertools.permutations` |
| Dense reward (`sum_i min_assignment E_i`) is invariant under permuting identical objects' labels, under `assignment_mode="free"` | `env.py` | random poses for k identical-shape objects, shuffle index order | `test_assignment.py` or `test_reward.py` |
| Bit-identical replay: for random seeds and random (valid) action sequences, `reset(seed=s)` + a fixed action trace produces identical `(obs, reward, info)` across two independent runs | `env.py` | `st.integers()` seed, `st.lists` of actions | `test_determinism.py` — hypothesis here is only the input generator; the check itself is tier 3's bit-exact equality, not a derived invariant |

### 1.3 Snapshot testing (`render.py`, Phase 10)

Nothing to snapshot before `render.py` exists, so this tier starts at Phase 10.

- **No new dependency.** Hand-roll rather than pull in `pytest-mpl` or
  similar: pygame already gives `rgb_array` output and the comparison logic is
  ~15 lines. One less dependency to pin.
- **Headless rendering.** Set `SDL_VIDEODRIVER=dummy` in the snapshot fixture
  only (not globally, so other tests can still use a real display for manual
  debugging) — `pygame.display.set_mode` needs it to succeed without an X
  server, even though `render_mode="rgb_array"` never opens a visible window.
- **Tolerance, not bit-exactness.** Font rasterisation and anti-aliasing vary
  a few pixel values across SDL/font versions and platforms; bit-exact compare
  would be flaky for reasons unrelated to real regressions. Compare mean
  absolute pixel difference against a threshold (e.g. mean < 1.0/255) *and*
  the fraction of pixels differing by more than 10/255 (catches localized
  regressions, like a missing lidar ray, that a mean alone can hide).
- **Goldens stored under `tests/snapshots/*.png`**, checked into
  git. Update workflow: a `--snapshot-update` pytest flag (custom
  `conftest.py` option) that overwrites goldens instead of asserting —
  mirrors the syrupy/jest-image pattern without the dependency.
- **Canonical scenes** (each from a fixed seed + fixed step count, so the
  underlying physics state is deterministic per tier 3):
  1. Fresh `reset()`, default preset, no toggles — draw-order baseline.
  2. Lidar rays on (`L`), single object partially occluding a ray.
  3. A goal rect currently satisfied — confirms green-fill-on-satisfied.
  4. `PushTPO-Lidar-Multi3-v0` mid-episode — confirms per-object recoloring
     under the live best assignment (SPEC.md §12's "genuinely informative"
     behavior).
  5. Belief overlay on (`B`), fed a dummy `fn` returning ground truth + fixed
     Gaussian noise — this doubles as acceptance test #6, made repeatable
     instead of eyeballed once.
- **Not a substitute for**: the manual `play.py` human-solvability check
  (acceptance test #3) has no automated stand-in and stays manual.

### 1.4 Tooling summary

Phase 0 adds `pytest`, `hypothesis`. Phase 10 adds nothing new — snapshot
comparison is hand-rolled numpy against the `pygame` dependency already in
place by then. New file `tests/conftest.py` holds: the hypothesis
settings profile, the `SDL_VIDEODRIVER=dummy` fixture, and the
`--snapshot-update` CLI option.

---

## 2. Phased plan

Phases are ordered by dependency, not by spec section number. Each phase should
land with its corresponding test(s) passing before moving on.

- [x] **Phase 0 — Scaffolding**
  Add core deps (`numpy`, `pymunk`, `gymnasium`, `scipy`, `pytest`,
  `hypothesis`). Create top-level `tests/` directory with `conftest.py`
  (hypothesis profile, per §1.4). No behavior yet.

- [x] **Phase 1 — Shapes registry** (`shapes.py`, `tests/test_shapes.py`) —
  **test-first**: §4's `ShapeDef`/`SHAPES`/`make_shape` signatures are exact.
  Write `test_shapes.py` (property-based per §1.2 — symmetry,
  area-normalisation, convex-part validity — plus hand-picked examples)
  against the not-yet-written module, then implement all 9 required shapes,
  numeric symmetry verification at import time, convex decomposition, area
  normalisation, until it passes.

- [x] **Phase 2 — Geometry primitives** (`geometry.py`, `tests/test_containment.py`)
  — **test-first**: §5 gives `contains()` and `containment_error()` verbatim.
  Write `test_containment.py` (property-based per §1.2 — containment/error
  agreement, margin monotonicity, vertex-order invariance — plus known-answer
  examples for axis-aligned/degenerate cases) first, then implement
  `minAreaRect` (rotating calipers), the transform helper, and the two
  functions from spec. Independent of pymunk — pure numpy, testable in
  isolation.

- [x] **Phase 3 — Config** (`config.py`) — **test-first**: §10's dataclass is
  given field-for-field. Write a small test asserting the default field
  values before writing the dataclass (there's little to discover here, but
  it still catches transcription slips against §10).

- [x] **Phase 4 — Goals & assignment** (`goals.py`, `tests/test_assignment.py`,
  `tests/test_modes.py`) — **test-first**: `GoalRect`'s shape and the
  Hungarian-assignment requirement are both spelled out in §5/§6. Write
  `test_modes.py` (generalised, property-based form of acceptance test #1,
  §1.2) and `test_assignment.py` (Hungarian-vs-brute-force, reward
  permutation-invariance, both property-based) first, then implement
  `GoalRect`, rejection sampling of goal poses, and the assignment resolution.

- [x] **Phase 5 — Lidar** (`lidar.py`) — **test-first** (reassessed from the
  original **implement-first** call above): on closer look, `cast_rays`
  doesn't actually need `env.py`'s real arena — it just needs *some* pymunk
  `Space` with shapes tagged by a hit-class contract, which a test can build
  standalone via a couple of static `pymunk.Segment`s. §7 already pins down
  the per-ray output (`[normalised_distance, hit_class_onehot]`), n_rays
  spacing, and "first hit only" semantics precisely enough to write tests
  against before writing the function.

- [x] **Phase 6 — Core env** (`env.py`) — **mixed**: the outward contract
  (`reset`/`step`, reward formula §6, `info` dict §11) is pinned down, so
  write tests against that contract first — a test asserting `info`'s exact
  key set, and a reward test using known object/goal poses with a hand-computed
  expected value. The internal pymunk wiring (space/walls/pusher
  PD-controller/object-body construction) is exploratory — implement that
  part first, then backfill. Pymunk space/arena/walls, pusher PD controller,
  object bodies from convex decomposition, both obs modes, reward
  (min-over-assignment, success hold, sparse_only), traps (§9), occluder
  walls.

- [x] **Phase 7 — Gymnasium registration & presets** (`__init__.py`) —
  **test-first**: the 6 presets and their configs are a literal table in
  §10. Write a test that `gymnasium.make()` succeeds for each registered ID
  and yields the expected config values, then register them.

- [x] **Phase 8 — Determinism** (`tests/test_determinism.py`) —
  **test-first**: the bit-identical-replay requirement (§14) is a direct,
  already-known test to write against the Phase 6 env. Write it (hypothesis-
  generated seeds/action sequences over a tier-3 bit-exact check per §1.2's
  last row), then audit `env.py`/`goals.py` for stray global `random`/
  `np.random` calls until it passes.

- [x] **Phase 9 — Vectorisation & speed** (`vec.py`, `scripts/benchmark_speed.py`)
  — **test-first** for the throughput bar: write `benchmark_speed.py` against
  the ≥2000 steps/sec/core target (`n_rays=64`, `n_objects=1`, acceptance
  test #5) before optimising anything, then implement `make_vec_env` over
  `AsyncVectorEnv` and iterate against that number.

- [x] **Phase 10 — Renderer** (`render.py`, `tests/snapshots/`) —
  **implement-first**: there's no golden image to assert against until a
  first render exists. Build the pygame visualiser (draw order per §12,
  belief overlay hook + `BeliefMarker`, keyboard controls) to a working
  state, capture the five canonical scenes from §1.3 as goldens, then those
  goldens become the regression tests for everything after.

- [x] **Phase 11 — Demo collection** (`scripts/collect_demos.py`, `scripts/play.py`)
  — **implement-first** for the two CLI scripts themselves (interactive
  operator tools; correctness judged by using them), but **test-first** for
  everything the scripts are built out of: `demo_storage.py`,
  `demo_collection.py`, `teleop.py`, and `env.py`'s new `compute_observation`
  all have clear, testable contracts, so tests came before implementation
  for each of those. Mouse teleop, episode discard (`N`), dual obs recording
  (full + lidar), zarr store in Diffusion Policy layout, mode-distribution
  reporting.

- [x] **Phase 12 — Acceptance pass** (§16 checklist end-to-end) — 5 of 6
  criteria verified; #3 (human solvability) genuinely requires a human and
  is flagged in §4, not something this session can self-certify. Found and
  fixed a real, significant physics-calibration bug along the way — see log.

---

## 3. Deviations from SPEC.md

- **`geometry.py`'s `contains`/`containment_error` take a `RectLike` Protocol,
  not `GoalRect`.** SPEC.md §5 writes their signature as
  `contains(rect: GoalRect, outline_world: np.ndarray)`, but `GoalRect` is
  assigned to `goals.py` in §15's file layout, and `goals.py` is built in
  Phase 4 — after `geometry.py` (Phase 2), which `goals.py` itself depends on
  for the containment functions. Importing `GoalRect` from `geometry.py`
  would invert that dependency (a cycle, or a forward-reference to a module
  that doesn't exist yet at Phase 2). Instead, `geometry.py` defines a local
  `typing.Protocol` (`RectLike`, with `.center`/`.angle`/`.half_extents`) and
  type-hints against that. No behavioural difference: Python's structural
  typing means Phase 4's `GoalRect` dataclass satisfies `RectLike`
  automatically as long as it has those three fields, with zero coupling code
  needed on either side.
- **`minAreaRect` named `min_area_rect`.** SPEC.md's prose uses camelCase
  (`minAreaRect`) but never gives it as a literal code signature (unlike
  `contains`/`containment_error`, which are given verbatim) — snake_case
  matches the naming convention of every other function SPEC.md *does* spell
  out (`make_shape`, `containment_error`, etc.), so `min_area_rect` was used.
  Purely cosmetic, noted only so a future reader isn't confused why one name
  doesn't match the prose 1:1.
- **Tests live at top-level `tests/`, not nested inside the package.**
  SPEC.md §15 literally nests them at `pusht_po/tests/*`, and Phases 0–4 were
  built that way (`src/push_me/tests/`) before this was corrected. User
  feedback: that layout is non-standard for a Python package built for
  distribution — a package's installed/built wheel shouldn't carry its test
  suite inside the importable module. Moved to `tests/` at the repo root
  (sibling to `src/`), dropped the now-unnecessary `tests/__init__.py`
  (top-level test directories don't need to be a package), and added
  `[tool.pytest.ini_options] testpaths = ["tests"]` to `pyproject.toml` so
  bare `uv run pytest` still finds them without an explicit path. No test
  content changed — every reference to `src/push_me/tests/...` elsewhere in
  this doc for *future* phases has been updated to `tests/...`; historical
  log entries below are left describing what was actually done at the time.

---

## 4. Open questions / risks

- **SPEC.md acceptance test #3 needs a human.** "A human can reliably solve
  `PushTPO-Lidar-Multi3-v0` via `play.py` — if a human cannot solve it with
  the same observations, the task is broken, not hard." A scripted oracle
  (Phase 12's log entry) shows the task makes real, substantial progress
  under a merely-competent heuristic (1–2 of 3 objects reliably contained)
  but doesn't finish within 300 steps — consistent with "hard by design,"
  but only an actual human playing `play.py` can confirm it isn't "broken."
  This is the one remaining item to close out SPEC.md's acceptance
  checklist in full.
- **Trap inescapability is asserted by geometry, not verified by simulation.**
  `env.py`'s traps are built so the throat is narrower than the pusher's
  diameter (§9's requirement), and `n_objects_trapped` is a point-in-pocket
  geometric test — but nothing actually drives a policy/pusher toward a
  trapped object and confirms it truly can't be extracted. Acceptance test
  #6-adjacent risk: if this matters for the Claim-A/B experiments later
  (§9's stated purpose — "a direct test of whether a least-action planner
  walks into an absorbing state"), consider adding a scripted-escape-attempt
  test before relying on it.
- **`min_area_rect` doesn't return a center.** For shapes without central
  symmetry (triangle, pentagon, T/L tetromino), the true minimal-bounding-
  rectangle center can be offset from the shape's centroid. Not needed for
  any current production code path (goals.py places rects at the sampled
  pose directly, per spec), but if a later phase wants to compute an
  "ideal fit" object pose — e.g. demo-collection tooling, or a debug/render
  affordance — this will need revisiting. See Phase 4's log entry for the
  numeric reasoning on why it's safe to skip for now.
- **Convex decomposition for the 4 curved-ish shapes (pentagon/hexagon/octagon)**:
  these are already convex as regular polygons, so decomposition is trivial
  (single part). Only the tetrominoes (T/L/S/Z) need real splitting. Spec's
  "ear-clipping plus merging" is over-general for our fixed shape set — plan is
  to hand-specify convex parts for the 4 tetrominoes and single-part for the
  rest, per the spec's own suggestion in §4.
- **zarr vs HDF5** for demo storage — spec allows either; decide in Phase 11
  based on what's easiest to get a Diffusion-Policy-compatible layout from.
- **Ray-cast batching**: §14 requires ≥2000 steps/sec/core; if
  `segment_query_first` per-ray in a Python loop doesn't hit that at 64 rays,
  will need to profile before reaching for a fix (batching, reduced default
  rays only as a config choice, never dropping substeps).

---

## 5. Log

### 2026-08-07
- Repo scaffolded (empty `push_me` package, `pyproject.toml`, no deps, no
  commits yet). Read `SPEC.md` in full. Created this log with phased plan
  mapping spec's `pusht_po/` layout onto the existing `src/push_me/` package.
  No code written yet — next up is Phase 0 (add deps) then Phase 1 (shapes
  registry), since geometry/goals/env all depend on it.
- Expanded the placeholder "tests" references into a concrete testing
  strategy (new §1): four tiers (property-based via `hypothesis`,
  example-based, determinism/golden-trajectory, visual snapshot), a property
  catalog mapping each of SPEC.md's universal claims to a hypothesis strategy,
  and a hand-rolled (no new dependency) pixel-snapshot plan for `render.py`
  with tolerance-based comparison and `SDL_VIDEODRIVER=dummy` for headless CI.
  Updated Phase 0/1/2/4/8/10 bullets to point at it instead of leaving "tests"
  unspecified.
- Added test-first-within-a-phase as a cross-cutting rule (new intro
  paragraph in §1) per user feedback, and annotated each phase in §2 as
  **test-first**, **implement-first**, or **mixed** depending on whether
  SPEC.md pins the API down before the phase starts.
- **Phase 0 complete.** Added `numpy`, `pymunk`, `gymnasium`, `scipy` as core
  deps and `pytest`, `hypothesis` as a dev group via `uv add`/`uv add --dev`.
  (First attempt added `gymnasium[classic-control]`, which pulls in `pygame`
  as a side effect of the extra — corrected via `uv remove`/`uv add
  gymnasium` with no extra, since pygame was deliberately deferred to Phase
  10 per §0's dependency table.) Created `src/push_me/tests/__init__.py` and
  `conftest.py` registering two hypothesis profiles (`default`,
  `max_examples=100`; `pymunk`, `max_examples=50, deadline=None`, exposed as
  `pymunk_settings` for physics-touching test modules to import, per §1.2/§1.4).
  Verified: all five packages import cleanly under `uv run python`, and
  `uv run pytest src/push_me/tests -q` runs (0 tests collected, as expected —
  no test files yet).
- **Phase 1 complete.** Wrote `test_shapes.py` first (19 tests: registry
  completeness, symmetry-order table, property-based rotation-onto-self
  check, centroid-at-origin, CCW winding, convex-part validity, convex-part
  area summing to outline area, and `make_shape` area normalisation /
  immutability of the base registry) — confirmed it fails on
  `ModuleNotFoundError` before writing any implementation. Then implemented
  `shapes.py`: `ShapeDef`, hand-specified outlines for the 4 tetrominoes
  (T/L/S/Z as two-rectangle convex decompositions, per §4's own suggestion),
  regular-polygon generation for triangle/square/pentagon/hexagon/octagon
  (already convex, single-part decomposition), CCW enforcement, and
  `_verify_symmetry` run at import time over every registry entry via
  Hungarian vertex-matching (not hand-asserted per-shape). All 19 tests
  passed on the first implementation attempt; separately confirmed the
  import-time verification isn't a no-op by calling it with a deliberately
  wrong symmetry order (`square` claimed as order 3) and observing it raise.
  No deviations from SPEC.md in this phase.
- **Phase 2 complete.** Wrote `test_containment.py` first (11 tests: a
  known-answer `min_area_rect` case, self-consistency and rotation-invariance
  properties for `min_area_rect` against all 9 Phase-1 shapes, an
  optimality-vs-AABB property, `transform_to_rect_frame` known cases, and for
  `contains`/`containment_error` — sign agreement, margin monotonicity,
  vertex-order invariance, and rigid-transform invariance, all property-based)
  — confirmed `ModuleNotFoundError` before implementing. Implemented
  `geometry.py`: `min_area_rect` via convex hull (`scipy.spatial.ConvexHull`)
  + rotating calipers over hull edges, `transform_to_rect_frame`,
  `contains`/`containment_error` verbatim per §5's pseudocode. All 11 passed
  on the first implementation attempt in isolation; running the *full* suite
  turned up one hypothesis-shrunk failure in
  `test_contains_is_rigid_transform_invariant` (s_tetromino, `hh=1.0` exactly
  matching the shape's exact half-extent — a genuine floating-point edge
  case, not a geometry bug: proved algebraically that `contains()`'s
  rigid-transform invariance is exact, but as a boolean it's a step function
  of a continuous quantity, so independently-rounded rotation matrices for
  `angle` vs `angle+dtheta` can flip the sign bit right at an exact boundary).
  Fixed by asserting only the continuous `containment_error` invariant
  (tolerance-checked) rather than the fragile boolean equality — reran the
  full suite 3× clean after. Also added `.hypothesis/` and `.pytest_cache/`
  to `.gitignore` (missed in Phase 0).
- **Phase 3 complete.** Wrote `test_config.py` first (field-set-matches-spec
  check via `dataclasses.fields`, default-value table check for every field,
  a mutable-default-factory independence check on `shapes`, and a
  keyword-override sanity check) — confirmed `ModuleNotFoundError` before
  implementing. Transcribed `PushTPOConfig` from §10 verbatim into
  `config.py`. All 4 new tests passed on the first implementation attempt (34
  total passing). No deviations.
- **Deviation** (§0 update, logged below in §3): `contains`/`containment_error`
  take a structural `RectLike` `Protocol` (`.center`/`.angle`/`.half_extents`)
  defined locally in `geometry.py`, not `SPEC.md`'s literal `rect: GoalRect`
  type hint — `GoalRect` lives in `goals.py` (Phase 4), which comes *after*
  `geometry.py` (Phase 2) in the dependency order, so importing it here would
  invert the module graph. Behaviourally identical: Phase 4's `GoalRect` will
  satisfy the protocol structurally with no changes needed on either side.
- **Phase 4 complete.** Before touching `goals.py`, extended `geometry.py`
  with `rect_corners`/`rects_overlap` (SAT overlap test for two rotated
  rectangles, 4 edge-normal axes) — needed for §5's "rejecting placements
  that overlap other goal rectangles or walls" but never given as literal
  pseudocode, so written test-first into `test_containment.py` (8 new tests:
  known cases, symmetry, self-overlap) before implementing, all passing
  first try. Settled the arena coordinate convention (see §0 update above),
  needed before sampling logic could be written at all.

  Then wrote `goals.py`'s tests first across three files: `test_assignment.py`
  (Hungarian-vs-brute-force via `itertools.permutations` oracle for k≤5,
  valid-permutation check, `fixed`-mode identity check, and total-cost
  invariance to row-permutation — the reward-permutation-invariance property,
  tested at the cost-matrix level rather than needing a full env),
  `test_modes.py` (generalised acceptance test #1 — all `symmetry_order`
  orientations contained with equal error, for all 9 shapes at margin
  24/area 4000, plus the literal hexagon case), and `test_goals.py`
  (`make_goal_rect` size/angle/`accepts` correctness, non-overlap and
  in-arena-bounds properties for `sample_goal_rects`, a monkeypatch check
  that global `np.random` is never touched, and an infeasible-placement
  `RuntimeError`). Confirmed all three files fail on `ModuleNotFoundError`
  before implementing `goals.py` (`GoalRect`, `make_goal_rect`,
  `sample_goal_rects` with rejection sampling, `resolve_assignment`). All 14
  new tests passed on the first implementation attempt — including
  `test_modes.py`, which I was genuinely unsure about (see the reasoning
  below) — and the full suite (52 tests) stayed green across 3 repeated runs.

  **Design note worth recording**: `min_area_rect` (Phase 2) returns only
  `(half_extents, angle)`, not a center — matching SPEC.md's literal
  construction steps, where the goal rect's center is simply the sampled
  `(gx, gy)`, independent of anything about the shape's own bounding-rect
  geometry. This is fine for production code, but for shapes *without*
  central symmetry (odd `symmetry_order` — triangle, pentagon; or
  `symmetry_order=1` — T/L tetromino), the true minimal-bounding-rectangle
  center can be offset from the shape's centroid (e.g. an equilateral
  triangle's centroid sits at 1/3 of its altitude, not at the bounding
  rectangle's half-height). Considered extending `min_area_rect` to return a
  center and threading it through so `test_modes.py` could place objects at
  a provably-exact fit — decided against it: SPEC.md's literal construction
  doesn't need it, and worked the numbers by hand for the worst case
  (equilateral triangle at `shape_area=4000`, offset ≈13.9 units against a
  margin of 24 and half-extents of ~42–48) to confirm the naive
  centroid-at-rect-center placement, at the spec's own margin=24, has
  comfortable slack for every registered shape. Verified empirically by
  running the test rather than trusting the estimate alone. If a future
  phase (e.g. demo collection's "ideal fit" tooling) needs the true center,
  revisit this — flagged in §4.
- **Test layout corrected**: moved `src/push_me/tests/` → top-level `tests/`
  per user feedback (see §3 deviation) — tests shouldn't ship inside the
  built package. Deleted `tests/__init__.py` (unneeded at top level), added
  `testpaths = ["tests"]` to `pyproject.toml`, cleared stale `__pycache__`.
  Full suite (52 tests) reran clean from the new location, both as
  `uv run pytest tests/` and bare `uv run pytest`.
- **Phase 5 complete.** Reassessed the Phase-0-era "implement-first" call for
  `lidar.py` (§2) — it assumed ray casting needed `env.py`'s real arena, but
  `cast_rays` only needs *a* pymunk `Space`, which a test can build directly.
  Switched to test-first. Checked pymunk 7.3.0's actual
  `Space.segment_query_first`/`ShapeFilter`/`SegmentQueryInfo` API by hand in
  a scratch script first (positional `(start, end, radius, shape_filter)`,
  `.alpha` as the hit fraction along the segment, thick shapes offset the hit
  point by their radius — used `radius=0.0` on test fixtures to keep expected
  distances exact) before writing anything, since SPEC.md doesn't specify
  pymunk's query API and guessing wrong would've produced confidently-wrong
  tests. Defined the hit-class contract as a `HitClass` `IntEnum`
  (`NONE`/`WALL`/`OBJECT`) read off `shape.collision_type` — `env.py` (Phase
  6) must tag every wall/object shape it creates with this. Wrote
  `test_lidar.py` first (7 tests: output shape, known-distance wall hit,
  genuine no-hit-in-path case, nearer-object-occludes-farther-wall — "first
  hit only" is inherent to `segment_query_first`, not something `cast_rays`
  implements itself — ray-index-to-world-angle correspondence, an
  empty-space property test across random `n_rays`/`lidar_range`, and a
  `shape_filter` exclusion case simulating "don't let the pusher's ray sensor
  hit its own body") — confirmed `ModuleNotFoundError` before implementing.
  `cast_rays` in `lidar.py` passed all 7 on the first attempt; full suite (59
  tests) stable across 3 repeated runs. No deviations from SPEC.md's
  observation contract; the `shape_filter` parameter and `HitClass` enum are
  necessary implementation details SPEC.md doesn't specify, not conflicts.
- **Phase 6 complete — the big one (`env.py`).** Built as planned per the
  **mixed** strategy: internal pymunk wiring implement-first (verified via
  scratch scripts, not formal tests, while the shape of the code was still
  moving), then a full `test_env.py` against the outward contract (§6/§11)
  once `reset`/`step` produced a coherent observation+reward+info. 18 new
  tests, all passing on the first attempt against the implementation as
  written; full suite (78 tests) stable across 3 repeated runs, plus
  `gymnasium.utils.env_checker.check_env` passes for both obs modes (two
  harmless warnings about unbounded `-inf`/`+inf` observation bounds, which
  is an intentional, documented choice — see §1's testing catalog didn't
  cover this, but it's the standard gymnasium idiom for envs with unbounded
  velocity/size components).

  **Before writing env.py**, extended `lidar.cast_rays` (Phase 5, already
  complete) to return a second array, `hit_object_index` — per-ray, which
  object index was hit (`-1` for wall/none hits) — needed for the `info`
  dict's `steps_since_observed` diagnostic (§11), which requires knowing
  *which* object a ray saw, not just that some object was seen. Done
  test-first: updated `test_lidar.py`'s existing assertions to unpack the new
  2-tuple, added `object_index`/`user_data` to the test fixtures and three
  new assertions, confirmed the updated tests fail against the old
  single-return implementation, then extended `cast_rays` — all 8 lidar
  tests (up from 7) passed after the change.

  **Design decisions SPEC.md leaves open, made and documented here** (none
  contradict SPEC.md — it's simply silent on these specifics):
  - *Object friction/elasticity*: SPEC.md gives pusher friction (0.6) and
    wall friction (0.6) + elasticity (0.0) explicitly, but never states
    object friction/elasticity. Used the same 0.6/0.0 for objects, for
    consistency with the rest of the "quasi-static pushing" setup (§3).
  - *Density and pusher mass*: SPEC.md's "mass scaled so all shapes have
    equal density" (§3) only requires *consistency*, not a specific value —
    used `density = 1.0` throughout (objects and pusher), since it's
    dimensionless without further constraints from SPEC.md.
  - *Pusher self-occlusion*: the ray sensor is mounted at the pusher's own
    position, so an unfiltered ray query would immediately self-hit at
    ~zero range. Solved with `pymunk.ShapeFilter` categories (`WALL`/
    `OBJECT`/`PUSHER` bits on each shape's `.filter.categories`, left at the
    default `mask` so physics collisions between all pairs are untouched)
    and an explicit query-time `shape_filter` excluding the pusher category
    when `_refresh_perception` calls `cast_rays` — confirmed via a scratch
    check that this doesn't also suppress physics collisions (pymunk's
    category/mask matching is symmetric per-pair; touching only
    `categories`, never `mask`, on the persistent shapes leaves collision
    resolution unaffected).
  - *Initial object/pusher placement*: SPEC.md's §5 "Construction" only
    describes sampling *goal* poses (rejection sampling against other goal
    rects and walls) — it's silent on where objects and the pusher start.
    Extended the same rejection-sampling approach to initial placement: at
    reset, the pusher is placed first (avoiding goal rects), then each
    object (avoiding goal rects *and* the pusher), via a local
    `_sample_free_pose` in `env.py`. This duplicates `goals.py`'s
    `sample_goal_rects` loop structure (~15 lines) rather than extracting a
    shared helper — considered refactoring `goals.py` to expose a generic
    rectangle-rejection-sampler both modules could call, declined for now to
    avoid touching a completed, tested phase for a modest amount of
    duplication; revisit if a third call site appears.
  - *`achieved_mode`'s "relative to rect" angle*: SPEC.md's formula
    (`round(θ_object_relative_to_rect / (2π/symmetry_order))`, §11) doesn't
    define what "relative to rect" means operationally. Derived it from the
    same relationship established in Phase 4/`test_modes.py`: an object
    "perfectly fits" its rect when its world angle equals `rect.angle - phi`
    (`phi` from `min_area_rect`), so `θ_relative = θ_object - (rect.angle -
    phi)`, wrapped mod `2π` before dividing and rounding, then `% n` to wrap
    the mode index. Validated directly: a property test places each of the 9
    shapes at all `k` of their `symmetry_order` orientations and asserts
    `achieved_mode == k` for every one — this only works if the formula is
    right, so it's a real check, not just a shape check.
  - *Reward's dense term isn't clipped*: implemented literally as
    `dense_weight * sum(containment_errors)` per §6's formula, with no
    `max(error, 0)` floor. Since `containment_error` is negative when
    well-contained, this means a *more deeply* contained object increases
    reward further (bounded, since error can't go below roughly
    `-half_extents`, but still a real, intentional-per-spec shaping effect
    worth flagging so it doesn't look like an oversight later). Verified by
    hand: placing a square exactly centered in its (margin=8) goal rect gave
    `reward = 1.08 = 1.0 - 0.01×(-8)`, matching the formula exactly.
  - *Trap geometry*: SPEC.md (§9) only says "concave wedge geometries";
    built each trap as 5 static segments (two funnel guides narrowing to a
    throat width of `1.2 × pusher_radius` — under the pusher's `2 ×
    pusher_radius` diameter, so it can't follow an object through — then two
    segments flaring back out to a `3 × pusher_radius`-wide pocket, closed
    by a back wall). `n_objects_trapped` is a geometric point-in-pocket test
    via `geometry.contains`, not a physics-derived "actually got stuck"
    signal — verified the geometry and the counting logic directly (an
    object placed at a trap's pocket center reports `n_objects_trapped=1`),
    but did *not* verify inescapability via a simulated escape attempt (out
    of scope for the effort this warranted — flagged in §4 if it matters
    later).
  - *Occluder walls*: placed as random segments (`0.15 × arena_size` long)
    at uniform random pose, with no rejection sampling against goals/objects
    — SPEC.md doesn't ask for placement guarantees here, and occluders are
    meant to be obstacles, so incidental overlap is acceptable.
  - *`shape_sampling="fixed"` with `len(config.shapes) != n_objects`*:
    SPEC.md's config table doesn't define this case. Cycles the given list
    (`names[i % len(names)]`) to fill `n_objects` slots — lets a preset like
    `shapes=["square"]` with `n_objects=3` work without requiring the config
    to spell out `["square", "square", "square"]`.
- **Phase 7 complete.** Small prerequisite fix in `env.py` first:
  `PushTPOEnv.__init__` now does `self.config = copy.deepcopy(config)`
  instead of holding the passed-in reference directly. Not needed for
  anything up to Phase 6, but registering presets means many `gym.make()`
  calls will share one `PushTPOConfig` instance from `_PRESETS` — without the
  copy, `env.config` mutations (or Phase 9's vectorised envs spinning up
  several instances from the same preset) would silently alias each other.
  One line, cheap, closes the hole before it's ever hit rather than after.

  Wrote `test_presets.py` first: registered-and-makeable + config-matches-
  table checks for all 6 presets (independently re-deriving the expected
  config values from §10's table rather than importing `push_me`'s own
  `_PRESETS` dict, consistent with not testing implementation against
  itself), a check that the trap preset actually sets `n_traps > 0` (§10
  says "as Multi3 plus traps" without a number — chose `n_traps=2`, see
  below), a few-steps smoke run through every preset, an independent-config
  check across two `gym.make()` calls of the same preset (this is what the
  `deepcopy` fix exists for), and a check that the four `Single`/`Multimodal`
  presets really do form the 2×2 factorial design §10 calls out as required
  ("Run it, or claims cannot be attributed"). Confirmed all fail against the
  unregistered IDs before writing `__init__.py`'s `_PRESETS` dict + `register()`
  loop. All 26 new tests passed on the first attempt — including the
  `push_me/__init__.py` → `push_me.env` → `push_me.config` import chain,
  which looked circular on paper (`__init__.py` importing a sibling module
  that imports another sibling of the same package before `__init__.py` has
  finished) but isn't, since none of the submodules import from the `push_me`
  package object itself, only from specific `push_me.X` module paths — this
  resolves fine in Python regardless of whether the parent package's
  `__init__.py` has finished executing. Full suite (104 tests) stable across
  3 repeated runs.

  Two presets are fully spec-literal restatements of `PushTPOConfig`
  defaults: `PushTPO-Lidar-Single-v0` is `PushTPOConfig()` unchanged, and
  `PushTPO-Full-Single-v0` differs only in `obs_mode`. No deviations from
  SPEC.md's table; `n_traps=2` for the trap preset is the one filled-in gap
  (documented, not specified).
- **Phase 8 complete.** Audited the source first: `grep` across `src/push_me`
  for `np.random.` usage found only the two legitimate `default_rng(...)`
  constructions in `env.py` plus a type hint in `goals.py` — no stray global
  randomness, no `import random` anywhere. Wrote `test_determinism.py`:
  a *meta-test* that greps the source tree itself for stray global-randomness
  usage (so a future regression gets caught automatically instead of relying
  on remembering to re-audit by hand), two preconditions that make the
  bit-identical claim meaningful rather than vacuous (different seeds give
  different initial observations; an unseeded `reset()` continues the
  existing stream while an explicit `reset(seed=X)` resets it — so "same
  seed ⇒ same trajectory" is actually testing something, not passing because
  seeding is a no-op), and four hypothesis-driven bit-identical-replay
  properties (full obs, lidar obs, traps + multiple objects, delta action
  mode) comparing two independent rollouts of the same `(config, seed,
  actions)` via `np.array_equal` on every `obs` and every `info` array, plus
  exact equality on `reward`/`terminated`/`truncated`. All 8 tests passed
  against the existing Phase 6/7 implementation with no code changes needed
  — the "no stray randomness" discipline held up from the start rather than
  needing a fix here. One detour worth recording: hypothesis's own
  statistics reported a handful of "invalid" examples per property (0
  failures) — traced this to the `_actions` strategy shape itself (lists of
  float-tuples) by reproducing the same "invalid" count on a bare `pass`-body
  test using an identical strategy with zero env involvement, confirming
  it's a benign artifact of hypothesis's internal data generation for this
  strategy shape, not a signal about `push_me`'s code. Full suite (112
  tests) stable across repeated runs. No deviations from SPEC.md.
- **Phase 9 complete — `vec.py`, `scripts/benchmark_speed.py`, plus a real
  optimization and one genuine spec-ambiguity resolved with the user.**

  `make_vec_env` is a thin `functools.partial(PushTPOEnv, config)` ×
  `n_envs` over `gymnasium.vector.AsyncVectorEnv`, per §14. Used `partial`
  specifically because `AsyncVectorEnv`'s workers need picklable zero-arg
  env factories — a nested closure (`def _make(): return PushTPOEnv(config)`)
  would fail to pickle; `functools.partial` over the module-level
  `PushTPOEnv` class doesn't. `test_vec.py` (4 tests, written alongside the
  implementation rather than strictly test-first — this module is a thin,
  low-risk wrapper around a standard gymnasium pattern) covers `num_envs`,
  reset/step batch shapes, that parallel sub-envs aren't lockstep-identical,
  and that `make_vec_env` doesn't mutate the caller's config (exercising the
  Phase 7 `deepcopy` fix under its actual intended use case).

  First run of `benchmark_speed.py` against the literal target
  (`AsyncVectorEnv`, `n_rays=64`, `n_objects=1`) **failed**: 1566 steps/sec/core
  at `n_envs=2`, and it got *worse* with more workers (520/sec/core at
  `n_envs=16`) — the opposite of what you'd expect from more parallelism.
  Profiled with `cProfile` before touching anything: found `_achieved_mode`
  was calling `min_area_rect(shape.outline)` — including a full convex-hull
  computation — from scratch on *every single step*, for a shape outline
  that never changes after `reset()`. This alone was ~20% of per-step cost.
  Fixed by caching `(half_extents, phi)` per object at `reset()`
  (`self._object_min_rects`), read by both `_build_objects` and
  `_achieved_mode` instead of recomputing. Raw single-process throughput
  went from 2837 → 4107 steps/sec immediately after.

  That fix alone wasn't enough to explain the `AsyncVectorEnv` numbers,
  though, so before concluding anything was still broken, isolated the
  IPC layer itself: built a genuine no-op env (empty `step()`, empty `info`)
  and ran it through `AsyncVectorEnv` at `n_envs=1` — 7657 steps/sec/core,
  comfortably above target even for doing *zero* work, ruling out "pymunk
  IPC/pickling is just slow" as the story. Then swept `n_envs` (1/2/4/8/16)
  for `push_me` itself: throughput *degraded* monotonically as workers
  increased even after the caching fix (`n_envs=1`: 2119/sec/core, passing;
  `n_envs=4`: 1535/sec/core, failing) — aggregate throughput across all
  workers *did* keep growing (2119 → 3753 → 6140 total), just sub-linearly,
  which pointed at the main process's per-step pipe-servicing loop (one
  Python-level round-trip per worker, every step) as the actual bottleneck,
  not per-worker compute — a known characteristic of subprocess-based
  vectorization when per-step compute is cheaper than IPC round-trip
  latency, not a `push_me`-specific bug.

  This left a genuine interpretive question for SPEC.md's acceptance test
  #5 ("`benchmark_speed.py` reports ≥ 2000 steps/sec/core"): does "per core"
  mean a single environment's own computational speed (which now clears the
  bar by 2×), or the realized aggregate throughput through
  `AsyncVectorEnv` specifically, divided by worker count (which degrades
  below the bar as `n_envs` grows, for reasons orthogonal to anything
  SPEC.md's own suggested remedy — "reduce n_rays" — could fix, since it's
  IPC-bound, not compute-bound)? Asked the user rather than guess on a
  stated acceptance criterion; they confirmed the single-process rate should
  gate pass/fail, with the `AsyncVectorEnv` number reported alongside for
  transparency rather than gated on. `benchmark_speed.py` now runs and
  prints both explicitly labelled this way.

  **Final measured numbers**: single-process 4204 steps/sec/core (target
  2000 — **PASS**, 2.1×), `AsyncVectorEnv` at `n_envs=4` on this 20-core
  machine: 1519 steps/sec/core (reported only, not gated). Added
  `tests/test_benchmark.py`: a lenient (>500/sec) regression guard using
  `env.step()` directly — deliberately far below the real 2000/sec target,
  since that number is machine-dependent and would make CI flaky if
  hard-gated — plus a subprocess smoke test that runs
  `scripts/benchmark_speed.py --skip-vec-env` end-to-end and checks it
  exits 0 with "PASS" in its output, catching script-level breakage (bad
  argparse setup, import errors when run standalone) that an in-process
  unit test wouldn't. Full suite (118 tests) stable across repeated runs.

  **Deviation-adjacent note** (not listed in §3 since it doesn't contradict
  SPEC.md's text, just resolves an ambiguity in it): the acceptance-test
  #5 pass/fail semantics above are a user-confirmed interpretation, not a
  unilateral judgment call — flagging here for visibility since it's the
  kind of decision a future reader would reasonably want to know was made
  deliberately, not overlooked.
- **Phase 10 complete — `render.py`, the pygame visualiser, plus prerequisite
  `env.py` extensions.** Implement-first as planned (no golden to write
  tests against before a renderer exists), then five canonical snapshots
  captured, *visually reviewed* (not just "didn't crash" — actually looked
  at the PNGs), which caught two real issues before they became load-bearing
  goldens: the "lidar rays" scene didn't actually demonstrate ray/object
  occlusion because the randomly-placed object happened to land outside
  `lidar_range` (fixed by deterministically placing it within range — the
  scene's whole point was to show the orange object-hit colour, so a scene
  that silently failed to do that would have been a bad regression fence);
  and the containment-error bar in the panel visually overlapped its own
  number text for large values (fixed by moving the bar to its own row
  below the text instead of trying to share a line).

  **Prerequisite `env.py` changes** (before touching `render.py` itself):
  added `pygame-ce` as a dependency (first time pygame enters the project —
  deferred since Phase 0 for exactly this reason); lifted the Phase 6-era
  `NotImplementedError` guard on non-`None` `render_mode`, now validated
  against `metadata["render_modes"] = ["human", "rgb_array"]`; added
  `render()`/`close()`/`set_belief_overlay()`/`_ensure_renderer()`, with
  `push_me.render` imported *lazily inside* `_ensure_renderer()` — not at
  `env.py`'s module top — so `render_mode=None` still involves zero pygame
  imports (§12's explicit requirement, verified with a subprocess-isolated
  test rather than an in-process `sys.modules` check, since another test in
  the same session could otherwise contaminate the check); added
  `self._last_reward`/`self._last_info` (the panel needs the *previous*
  step's reward/info, which nothing before this phase needed to retain);
  and — a real gap, not a style choice — `_build_walls`/
  `_build_occluder_walls`/`_build_traps` were constructing pymunk segment
  shapes and adding them straight to the space without keeping any
  reference, so there was no way for a renderer to later ask "where are the
  walls/occluders/trap geometry" at all. Added `self._wall_segments`/
  `self._occluder_segments`/`self._trap_wall_segments` to close that gap.

  **`render.py`**: draw order exactly per §12 (background → walls/traps →
  goal rects → objects → lidar rays → pusher → belief overlay → panel).
  World-to-pixel transform is `[0, arena_size]² → [0, 512]²` scaled by
  `512/arena_size` (so non-default `arena_size` configs still render
  correctly) with y flipped (world "up" renders as screen "up", not
  pygame's native y-down). Per-rect/per-object colour coding uses an
  8-colour palette indexed by rect number, with objects coloured by
  `rect_colors[assignment[object_i]]` — recolouring live as `assignment`
  changes each step is the "genuinely informative" behaviour §12 calls out,
  confirmed visually in the multi-object golden (traced the colours by hand
  against `info["assignment"]` for that frame: obj0→rect0(red),
  obj1→rect2(green), obj2→rect1(blue), all consistent in the image). Belief
  markers are dashed polygon outlines (hand-rolled dashed-line helper —
  pygame has no built-in dashed primitive) at alpha driven by `confidence`.
  `rgb_array` mode needs no display driver at all (only `"human"` calls
  `pygame.display.set_mode`), confirmed empirically before relying on it.

  **A real bug found and fixed via the keyboard tests**: `Esc`/window-close
  originally called `self.close()` (tearing down pygame's display) *directly
  from inside* `Renderer._handle_events`, which is itself called from partway
  through `Renderer.render()` — meaning execution would return from event
  handling and continue trying to draw to a surface whose display had just
  been torn down. Restructured to a `close_requested` flag: the renderer
  only flags intent during event handling and returns immediately without
  drawing; `PushTPOEnv.render()` (which owns the `self._renderer` reference)
  checks the flag *after* `Renderer.render()` returns cleanly and only then
  calls the actual teardown. Caught by writing `test_escape_closes_the_renderer`
  and tracing through why the naive version would have left the renderer in
  a half-torn-down state — not something the snapshot tests would have
  caught, since they never press Escape.

  **Keyboard**: all seven keys from §12's table are wired
  (`Space`/`R`/`L`/`B`/`G`/`Tab`/`Esc`), tested by posting synthetic
  `pygame.KEYDOWN` events (`sdl_dummy_driver` fixture) rather than needing a
  real display or physical input. `Tab` ("cycle presets") is the one key the
  renderer can't act on by itself — cycling presets means constructing a
  *new* env with a different config, which the renderer has no authority to
  do (it's attached to one specific env instance). It sets
  `preset_cycle_requested`, exposed via `consume_preset_cycle_request()`,
  for `play.py` (Phase 11) to poll and act on.

  Snapshot infrastructure landed in `tests/conftest.py` per §1.3's plan:
  `--snapshot-update` CLI flag, `sdl_dummy_driver` fixture, and
  `assert_matches_snapshot` — hand-rolled against `pygame.image.save`/`load`
  (no new dependency), tolerance-based (mean pixel diff + fraction of
  outlier pixels, not bit-exact) for the reasons already documented in §1.3.

  33 new tests across `test_render.py` (21) and the snapshot goldens
  themselves; full suite (139 tests) stable across 3 repeated runs.
  Re-verified the Phase 9 throughput target still holds after adding the
  pygame dependency and the `_last_reward`/`_last_info`/segment-list state:
  4090 steps/sec/core single-process, unchanged within noise — confirms the
  zero-pygame-import path for `render_mode=None` is genuinely zero-cost, not
  just zero-import.

  **Simplification, not a deviation**: §12 describes traps as "dark
  hatched." Implemented as a semi-transparent dark-red polygon fill over the
  pocket region plus the actual wedge wall lines in a distinct colour,
  rather than a literal cross-hatch texture — conveys the same information
  (hazard, visually distinct from goal rects) without the clipping
  complexity of rendering diagonal hatch lines confined to an arbitrary
  rotated polygon. Noted here rather than in §3 since SPEC.md's own wording
  is descriptive, not a literal rendering spec.
- **Phase 11 complete — `demo_storage.py`, `demo_collection.py`,
  `teleop.py`, `scripts/play.py`, `scripts/collect_demos.py`, plus one
  `env.py` extension and one real bug found and fixed in `render.py`.**

  **Storage backend decided**: zarr, following §13's Diffusion Policy
  layout literally (`data/<field>` arrays concatenated across episodes,
  `meta/episode_ends` cumulative index, matching the actual
  `diffusion_policy.common.replay_buffer.ReplayBuffer` convention this repo
  doesn't have installed to test against directly). Checked the installed
  `zarr` version (3.3.0 — a materially different Python API from the 2.x
  series Diffusion Policy's own codebase was written against) by hand before
  writing anything: `zarr.open_group(path, mode="a", zarr_format=2)`
  produces the classic `.zgroup`/`.zarray`/`.zattrs` on-disk layout
  (confirmed by inspecting the written files directly), so the *file format*
  Diffusion Policy's loader expects is reproduced even though the *writing*
  API is zarr 3.x's. `ReplayBufferWriter` appends incrementally per episode
  (resizing arrays, not buffering a whole session in RAM) so a crash
  mid-session doesn't lose already-kept episodes. Config is stored **once**
  at the store's top level (`root.attrs["config"]`, JSON via
  `dataclasses.asdict`), not per-episode — a collection session targets one
  fixed preset/config, so per-episode config storage would just be N
  identical copies. 8 tests in `test_demo_storage.py`, written against the
  schema before `demo_storage.py` existed, all passed on the first
  implementation attempt.

  **`env.py` extension**: added `compute_observation(obs_mode)`, computing
  either observation type from the *current* physics state regardless of
  `self.config.obs_mode` — needed because §13 requires recording both
  `obs_full` and `obs_lidar` every step even though the collection session
  itself runs in `obs_mode="full"`. Extracted `_cast_lidar()` out of
  `_refresh_perception()` and gave `_lidar_observation()` an optional
  override parameter so the "give me the other mode, just for recording"
  path is a genuinely separate call from the live perception pipeline — it
  must *not* touch `self._steps_since_observed`, since that bookkeeping is
  specifically about what an agent under the env's *actual* configured
  `obs_mode` would experience, not a side-channel probe for the dataset.
  Tested directly: a dedicated test checks the internal counter is
  unchanged immediately after two `compute_observation("lidar")` probe
  calls — deliberately *not* going through `info["steps_since_observed"]`
  after a real `step()`, since `obs_mode="full"` unconditionally zeroes that
  counter every step regardless, which would have made the test pass
  vacuously even with a real regression.

  **A real bug, found via the keyboard/discard tests, not guessed at.**
  Original design: a `teleop.poll_events(env)` helper would drain pygame's
  event queue itself, forward recognised keys (Space/R/L/B/G/Tab/Esc) to the
  renderer's `_handle_keydown`, and return anything else (like `collect_demos.py`'s
  `'N'`) for the caller to handle. Wrote `collect_one_episode` against this,
  and its discard-on-`N` and quit-on-`Esc` tests both failed — not with an
  error, just silently never triggering, running the full episode to
  completion regardless of the posted key. Debugged with actual
  instrumentation rather than guessing: printed `id(env._renderer)` around
  each step and found it **changing** — a brand-new `Renderer` was being
  created mid-episode. Root cause: `PushTPOEnv.step()` (and `reset()`)
  already auto-render in `render_mode="human"` (the standard gymnasium
  convention, built in Phase 10) — which means *that* internal render call
  is what actually drains pygame's event queue every step, before
  `collect_one_episode`'s own separate `poll_events()` call at the top of
  the *next* iteration ever got a chance to see anything posted during the
  current one. For `Esc` specifically it was worse than just "too late":
  the internal auto-render's `_handle_keydown` correctly set
  `close_requested` and closed the renderer *within* `env.step()` — but
  `collect_one_episode` then called `env.render()` *again* explicitly, and
  `_ensure_renderer()` saw `self._renderer is None` and silently created a
  **fresh** renderer, undoing the close. For `N` (a key the renderer doesn't
  recognise at all) the event was simply drained and discarded by the
  internal auto-render's `pygame.event.get()` call with nowhere to go.

  Fixed by removing the redundant explicit `env.render()` call from
  `collect_one_episode` entirely (relying on `step()`'s own auto-render, per
  gymnasium convention — an external caller in human mode never needs to
  call `render()` itself) and, more fundamentally, centralising *all* event
  draining in the `Renderer` — the one object that actually calls
  `pygame.event.get()` every step regardless of who asked. Added
  `Renderer._unhandled_keys` + `consume_unhandled_keys()` (same
  pop-and-clear pattern as the existing `consume_preset_cycle_request()`),
  populated by `_handle_keydown`'s `else` branch for any key it doesn't
  already own. Deleted `teleop.poll_events` and `_RENDERER_KEYS` entirely —
  no second poller is needed once the renderer is the sole source of truth
  for "what keys were pressed." `teleop.py` now holds only
  `mouse_to_action` (5 tests). Added `test_render.py` coverage for the new
  `consume_unhandled_keys()` mechanism directly, alongside the existing
  keyboard tests, so this exact failure mode has a regression fence.

  **`demo_collection.py`**: `collect_one_episode(env, seed, get_action) ->
  (EpisodeOutcome, dict | None)`, `EpisodeOutcome` = `KEPT`/`DISCARDED`/`QUIT`
  — the three-way distinction matters to the caller (`collect_demos.py`):
  `DISCARDED` means "try this episode again with a new seed," `QUIT` means
  "stop the whole session," which a bare `None` return couldn't
  distinguish. `get_action` is injected (defaults to real mouse polling in
  the scripts, a fixed/scripted callable in tests) — this is what makes the
  interactive loop's *logic* testable without a real mouse, even though the
  literal "move your mouse" interaction can't be. 5 tests, all against
  synthetic `pygame.KEYDOWN`/mouse-free scripted action sources under
  `sdl_dummy_driver`.

  **`scripts/play.py` / `scripts/collect_demos.py`**: thin CLI wrappers
  (argparse + the tested library calls). Automated coverage is `--help`
  smoke tests via subprocess (3 tests, `test_scripts.py`) — a full
  interactive run can't be meaningfully scripted without a real mouse, and
  `render_mode="human"`'s `clock.tick(30)` throttle makes even a
  fully-automated run of `collect_demos.py` (headless mouse position pinned
  at whatever `pygame.mouse.get_pos()` returns under the SDL dummy driver,
  running to `max_steps` via truncation) take a mandatory ~10s for a
  300-step episode — correct behavior for a real session, too slow to bake
  into the routine suite. Instead **verified manually, once, during this
  session**: ran `collect_demos.py --preset PushTPO-Lidar-Single-v0
  --n-episodes 1` genuinely end-to-end under `SDL_VIDEODRIVER=dummy`, then
  read the resulting store back directly — `obs_full` shape `(300, 26)`,
  `obs_lidar` shape `(300, 275)` (both matching the dimension formulas from
  Phase 6), config round-tripped correctly, and the mode-distribution report
  printed correctly (`mode 0: 1/1 (100%)`). This is the DESIGN_LOG's own
  stated Phase 11 plan working as intended: implement-first for the
  interactive shell, correctness judged by using it — the library layer
  underneath carries the automated test burden.

  **Not yet verifiable by me**: SPEC.md's acceptance test #3 ("A human can
  reliably solve `PushTPO-Lidar-Multi3-v0` via `play.py`") requires an
  actual human at a mouse — flagged for Phase 12 / the user, not something
  a coding session can self-certify.

  Full suite (166 tests) stable across 3 repeated runs, ~7s total.
- **Phase 12 complete — acceptance pass against §16, and the most
  significant bug found this entire build.**

  Mapped SPEC.md §16's 6 acceptance tests to what's already verified:
  1. `test_modes.py` (hexagon, margin 24, all 6 orientations equal-error and
     contained) — Phase 4. ✅
  2. `test_assignment.py` (Hungarian vs. brute force, k≤5) — Phase 4. ✅
  4. Bit-identical seed replay — Phase 8. ✅
  5. `benchmark_speed.py` ≥ 2000 steps/sec/core headless — Phase 9, currently
     4384 steps/sec/core (2.2×). ✅
  6. Belief overlay dashed outlines from ground-truth+noise — Phase 10. ✅
  3. "A human can reliably solve `PushTPO-Lidar-Multi3-v0` via `play.py`" —
     **cannot be self-certified**; needs an actual human at a mouse.

  Before asking the user to run #3, wanted independent evidence the task is
  at least *possible*, not just "doesn't crash" — so wrote a scripted greedy
  oracle policy (push whichever object has the worst containment error
  toward its assigned goal) and ran it against single-object and Multi3
  configs. Result: **0/20 solved on both**, and worse, single-object
  containment error barely moved at all over a full 300-step episode.

  Debugged from first principles rather than assuming "the task is just
  hard": traced a single object's position and the pusher's distance to its
  own commanded target over 60 steps — pusher was still ~192 units from a
  target it had been driving toward the whole time. Isolated the physics
  directly (no env, just a pymunk body under `space.damping=0.05` and a
  constant `force=(500, 0)`, both SPEC.md §3 literal values): terminal
  velocity was **0.24 units/s** — crossing the 512-unit arena would take
  ~21,000 environment steps against a `max_steps=300` budget. Root cause:
  `_OBJECT_DENSITY = 1.0` (Phase 6's arbitrary, undocumented-as-load-bearing
  choice for "mass scaled so all shapes have equal density," §3 — the
  density *value* itself was never specified by SPEC.md) gave the pusher a
  mass of ~707 (`density × π × pusher_radius²`), which the given
  `kp`/`kd`/`max_push_force`/`damping` constants — genuinely fixed by
  SPEC.md, not tunable — simply cannot move at a usable pace. This was
  invisible to every prior test because none of them required the pusher to
  travel any real distance: reward/termination tests manually set object
  poses directly, and the Phase 9 throughput benchmark measures steps/sec,
  not distance covered per step.

  Recalibrated empirically rather than guessing a round number: swept
  density from 1.0 down to 0.001, computing terminal velocity analytically
  and confirming with direct pymunk simulation at each step, and picked
  **0.002** — pusher crosses the arena in ~40 steps (leaving ~260 of the
  300-step budget for maneuvering), and a direct push test showed a
  stationary object moving 55 units in 60 steps of sustained contact.
  Reran the oracle: single-object jumped to **15/20 (75%) solved**. Multi3
  stayed at 0/20, but a stickier version of the oracle (commit to one
  object until it's actually contained, rather than always re-targeting
  whichever is currently worst) got 1–2 of 3 objects reliably into their
  goals (containment error going *negative*) across 10 seeds — real,
  substantial progress, just not enough within 300 steps for a naive
  heuristic to finish all three. This is consistent with SPEC.md's own
  framing of `PushTPO-Lidar-Multi3-v0` as "the dishwasher" — a deliberately
  hard multi-object task — rather than evidence of a remaining bug. Doesn't
  prove human-solvability (that's still §16 test #3, for an actual human),
  but it does rule out "the physics makes this task literally impossible,"
  which was the real risk after finding the density bug.

  Added two permanent regression tests (`test_pusher_can_cross_most_of_the_arena_within_a_third_of_max_steps`,
  `test_pusher_can_push_an_object_a_meaningful_distance`) that assert on
  actual distance covered per step budget, not just formula correctness —
  exactly the class of test that would have caught this in Phase 6 had it
  existed then. Regenerated the two render snapshots whose captured
  trajectories depended on movement speed (`02_lidar_rays`,
  `04_multi_object_assignment`); visually re-reviewed both, unchanged in
  substance. Re-ran `benchmark_speed.py` (4384 steps/sec/core, still 2.2×
  target — the density change doesn't affect step throughput) and the full
  suite (168 tests, 3 repeated runs, ~7s each).

  This is logged under Phase 12 rather than retroactively amending Phase
  6's entry, per this log's own append-only convention — Phase 6's entry
  above accurately describes what was built and verified at the time; this
  is where the gap was actually found.
- **Playtest-driven tuning pass.** After manually driving `PushTPO-Lidar-*`
  presets via `play.py`, user feedback: the lidar-only task is very hard to
  solve as a human, and worth making more approachable rather than treated
  as a fixed difficulty knob. Four changes, all defaults/config, no new
  mechanics:
  - `n_rays` default 64 → 128 (§10, §7): finer angular resolution, cheap
    (ray casting isn't the throughput bottleneck at this scale per Phase 9's
    benchmark).
  - `render.py`'s lidar visualisation switched from a full line per ray
    (pusher → hit point) to a single dot at the hit point (`_draw_lidar_rays`
    renamed `_draw_lidar_hits`). At 64 rays the line fan was already busy;
    at 128 it would be unreadable. Dots read as a point-cloud silhouette of
    what's actually in range instead of visual noise radiating from the
    pusher — confirmed by eye on `02_lidar_rays`/`04_multi_object_assignment`
    (regenerated, `01`/`03`/`05` are pixel-identical since they're
    `obs_mode="full"` scenes with no lidar drawn) and a throwaway render at
    the new `n_rays=128` default.
  - `Renderer.show_ground_truth` and `show_belief` default `True` → `False`
    (§12's `G`/`B` keys). Both draw information the lidar policy cannot
    itself see; leaving them on by default during teleop demo collection
    risked exactly the kind of operator-behavior confound §13 already warns
    about for mode preference, just visual instead of decision-level. `L`
    (lidar hits) stays on by default — that *is* the sensed observation, not
    an aid. The five canonical snapshot tests explicitly re-enable both
    before capturing (`_enable_ground_truth_and_belief` helper in
    `test_render.py`), since those scenes are specifically about showing the
    ground-truth/belief draw paths, independent of the interactive default.
  - `max_steps` default 300 → 600, doubling the per-episode budget given the
    task is materially harder than anticipated at design time.

  Updated `SPEC.md` defaults table and prose in lockstep (§7, §10, §12) per
  `test_config.py`'s exact-match contract with the config dataclass, and
  `tests/test_config.py`'s `EXPECTED_DEFAULTS`. Fixed the two render tests
  whose assertions encoded the old default (`test_b_toggles_belief_overlay`,
  `test_g_toggles_ground_truth` now assert off→on instead of on→off) and
  renamed/reordered `test_toggling_ground_truth_off_changes_the_frame` to
  `..._on_changes_the_frame` (captures the off-by-default frame first, then
  the explicitly-enabled one). Full suite green (168 passed) after
  regenerating the two affected snapshots.
- **Second playtest note: `play.py` starts each rollout mid-lurch.** Root
  cause: `play.py`'s loop feeds `mouse_to_action(pygame.mouse.get_pos(), ...)`
  into `env.step()` every iteration unconditionally, including the very first
  iteration after an auto-reset on episode end — the pusher's PD controller
  immediately targets wherever the mouse happens to be left over from the
  previous rollout, producing a hard yank before the operator can react.
  `Renderer.paused` already existed (Space key) but nothing gated stepping on
  it — it only skipped the *draw* call, since `env.step()` doesn't consult
  render state and `Renderer.render()`'s pause branch runs after event
  handling but returns before drawing.
  Fix, script-side only (didn't touch `PushTPOEnv.reset()` or
  `demo_collection.py` — see below): `Renderer._handle_keydown`'s `K_r` case
  now sets `self.paused = True` right after `self.env.reset()`, and
  `play.py`'s loop checks `renderer.paused` before building an action —
  while paused it just calls `env.render()` (still pumps events, so Space/
  Tab/Esc/etc. keep working, clock-limited to FPS) and skips `env.step()`
  entirely. `play.py`'s auto-reset-on-episode-end branch and `_make_env`
  (used for both the initial preset and `Tab` cycling) now pause the same
  way, so every rollout — including the first — starts paused until Space,
  giving the operator a uniform "reposition mouse, then go" beat. Updated
  `test_r_resets_the_episode` to assert the post-R paused state;
  `test_space_toggles_pause` needed no change since `env.reset()` itself was
  deliberately left alone.
  `scripts/collect_demos.py` has the identical bug (`collect_one_episode`
  steps on the mouse position immediately after its internal `env.reset()`)
  but wasn't touched here — fixing it means gating inside `collect_one_episode`
  itself (the reset-and-step-loop is one function, not split across a
  script-level loop like `play.py`'s), which would require simulating a Space
  press in the five `test_demo_collection.py` tests that currently expect
  stepping to start immediately. Left for a follow-up since it's a more
  invasive change than this playtest note asked for.
- **Follow-up: fixed `collect_demos.py` too**, per explicit request — a
  script-level "holding page" rather than touching `collect_one_episode` /
  `demo_collection.py`, so the five existing `test_demo_collection.py` tests
  (which expect stepping to start immediately after reset, no Space
  simulated) stay untouched. Added `_await_next_rollout(env)` in
  `collect_demos.py`: pauses the renderer and spins on `env.render()` (still
  pumps events, so Space/Esc/N/etc. work) until either `paused` clears
  (Space — proceed) or `env._renderer` goes `None` (Esc/window close —
  quit), called at the top of the episode loop so it gates the first
  episode, post-KEPT, and post-DISCARDED transitions uniformly, unlike the
  originally-suggested single insertion point (which only covered the
  post-KEPT path and would've left the discard path re-stepping on stale
  mouse position).
  Hit a real bug writing this: pausing *before* the very first `reset()`
  isn't safe the way it is in `play.py` (which only ever pauses after a
  successful reset) — `Renderer.render()` checks `self.paused` *after*
  processing events, so a Space keydown already queued when the hold starts
  flips `paused` back to `False` and falls through to the draw calls in that
  same `render()` invocation, crashing on `_pusher_body.position` being
  `None` pre-reset. Fixed by giving `main()` an initial `env.reset(seed=
  args.seed_start)` before entering the loop, so physics state always exists
  by the time anything can unpause — and since `collect_one_episode`'s first
  call reuses the same seed, this "throwaway" reset isn't actually wasted:
  by this repo's own determinism guarantee (§14, `test_determinism.py`) it
  produces a bit-identical layout, so the frame the operator previews during
  the very first hold *is* the upcoming episode, not a decoy.
  Verified by driving the real module functions directly (not a subprocess —
  `pygame.event.post` from a second thread hit `pygame.error: video system
  not initialized` under `SDL_VIDEODRIVER=dummy`, so drove `_await_next_rollout`
  / `collect_one_episode` in-process instead): quit-during-hold, Space-during-
  hold with a same-position pusher assertion (hold genuinely doesn't step
  physics), R-then-Space during hold, and a full two-episode run through
  `ReplayBufferWriter` confirming `n_episodes=2`/`n_steps` match. Full suite
  green (168 passed), pyrefly clean.
- **Third playtest note: can't see object colours.** Objects were recoloured
  live to match their currently-assigned goal rect (`_draw_objects` picked
  `rect_colors[assignment[i]]`), per §12's original item 3 — meant to make
  the agent's implicit mode/box commitment visible. Useless to an operator
  who can't distinguish the palette, and asked whether to go further: since
  `assignment_mode="free"` (the default, and the only mode any preset uses)
  already resolves objects to boxes via optimal bipartite matching — i.e.
  success was *already* order-agnostic, no policy or human ever had to send
  a particular object to a particular box — the actual ask reduced to
  dropping the now-misleading colour cue, not changing reward/success logic
  at all. Confirmed the alternative (drop the one-to-one matching entirely,
  let objects double up in one box) explicitly before touching anything,
  since that *would* have been a reward-semantics change; user chose to keep
  matching as-is and just fix the display.
  `_draw_objects` now always draws every object in one neutral colour
  (renamed `_UNASSIGNED_OBJECT_COLOR` → `_OBJECT_COLOR`, since "unassigned"
  no longer means anything special); `_draw_goal_rects` no longer needs to
  return `(rect_colors, assignment)` since nothing downstream consumes them
  anymore (goal-rect outlines keep their per-index palette colour and the
  green satisfied-fill, both unaffected — only the *object* fill colour
  changed). Updated §12 item 3's prose to match. Renamed the now-inaccurate
  `test_snapshot_multi_object_recoloring_under_assignment` →
  `test_snapshot_multi_object_scene` and its golden
  `04_multi_object_assignment.png` → `04_multi_object.png`. Regenerated all
  five canonical snapshots (object colour is global, so every scene with
  `show_ground_truth` on changed) and reviewed each by eye: goal-rect
  palette and the green "satisfied" fill both still distinct from the now-
  uniform grey objects. Full suite green (168 passed), pyrefly clean.
- **Fourth playtest note: does the observation include per-rect occupancy?**
  User asked directly rather than assuming; answer is no in either
  `obs_mode` — `_goal_rect_full_features`/`_goal_rect_lidar_features` only
  ever encode static task spec (`center`, `cos/sin angle`, `half_extents`,
  and `accepts` for lidar mode); `is_success`/`containment_errors`/
  `assignment` are computed solely into `info`, never `obs`. Following up on
  their own answer, user then caught that `Renderer._draw_goal_rects`'s
  green "currently satisfied" fill was being computed unconditionally
  (`if info is not None`, not gated on `show_ground_truth`) — so even with
  helpers off by default (this same log, playtest note 1) it was still
  leaking exactly this un-observed occupancy/success signal to a teleop
  operator on every frame. Real bug, not a hypothetical: it's the same
  category as the ground-truth/belief default-off fix, just missed because
  `_draw_goal_rects` (rect outlines, always fair — rects are given directly
  in the observation) and its satisfied-highlight computation live in the
  same method, and only the outline-drawing half was ever gated.
  Fix: gate `satisfied_rects` computation on `self.show_ground_truth`
  (one-line condition change), leaving the per-index outline/base-fill
  drawing unconditional as before. No snapshot regeneration needed —
  `test_snapshot_satisfied_goal_turns_green` already forces
  `show_ground_truth = True` via `_enable_ground_truth_and_belief` (playtest
  note 1's helper), so it still exercises and passes against the green fill
  unchanged. Full suite green (168 passed), pyrefly clean.
- Unrelated to the above: `config.py`'s `max_steps` was hand-edited to
  `1000` outside this session (was `600`, this log's playtest note 1),
  desyncing `test_config.py`'s `EXPECTED_DEFAULTS`/`SPEC.md` §6/§10 from the
  dataclass and red-lining `test_default_values_match_spec_table`. Not
  reverted (per this repo's own convention, SPEC.md follows the config, not
  the other way round, same as every other default change in this log) —
  updated the two doc/test references to `1000` to restore the contract.
