# METR-LA Dissertation — Complete Notebook Series (01–13)

Two folders, one continuous pipeline, all `.ipynb`, all built against your
`witgaw/METR-LA` Hugging Face dataset.

```
01_06_original_notebooks/          <- your existing EDA, reconstructed from
                                       the HTML exports you uploaded (real
                                       code preserved, outputs not included —
                                       re-run to regenerate them)
07_13_methodology_validation/      <- continuation, methodology checks
LITERATURE_REVIEW.md               <- paper-by-paper table: dataset,
                                       limitation, models used, proposed
                                       system (27 papers, same review
                                       depth as everything else here)
```

## `01_06_original_notebooks/`

Reconstructed directly from your uploaded HTML exports (using the most
recent version of each — `02_new.html` and `05_new.html` where those existed,
not the earlier `02.html`/`05.html`). Every code cell was checked to parse as
valid Python before being included. Outputs (printed results, plots) are
NOT included — HTML exports don't cleanly round-trip back into re-runnable
output cells, so these are code-only. Run them top to bottom to regenerate
everything.

| File | What it does |
|---|---|
| `01_schema_and_sanity_check.ipynb` | Confirms dataset schema, `d0`/`d1` meaning, node_id↔graph alignment |
| `02_data_quality_and_reliability.ipynb` | Zero-rate detection + de-seasonalized CUSUM/EWMA drift (the fixed version) |
| `03_descriptive_stats.ipynb` | Distributions, stuck-at-zero detection, per-sensor disparity check |
| `04_temporal_eda.ipynb` | Weekend/timestamp/seasonal sanity checks |
| `05_graph_topology_eda.ipynb` | Graph sparsity, density, haversine unit verification (the fixed version) |
| `06_disparity_and_collinearity_check.ipynb` | Correlation, VIF, honest non-causal framing |

## `07_13_methodology_validation/`

Continues directly from 06. Resolves the reviewer critique's "what must be
fixed before implementation" list.

| File | Language (in-notebook) | Tested? | What it resolves |
|---|---|---|---|
| `07_setup_and_fairtp_verified.ipynb` | Python | ✅ Yes | Confirms FairTP's code is public, extracts and verifies its real RSF/SDF formulas, sets up R via `rpy2` for everything after this |
| `08_dag_identifiability.ipynb` | R (`dagitty`, via `%%R`) | ⚠️ Not yet run | Checks the causal DAG is actually identifiable |
| `09_ctf_estimation_faircause.ipynb` | R (`faircause`, via `%%R`) | ⚠️ Not yet run | The core Ctf-DE/Ctf-IE/Ctf-SE estimation |
| `10_power_simulation.ipynb` | R (`faircause`, via `%%R`) | ⚠️ Not yet run | Is n=207 enough for stable estimates? |
| `11_reliability_weight_sensitivity.ipynb` | Python + R, one notebook | Python half ✅ / R half ⚠️ | Does the causal conclusion depend on the heuristic 0.6/0.2/0.2 weights? |
| `12_disparity_reconciliation.ipynb` | Python | ✅ Yes | Does the EDA's persistence-error finding agree with DCRNN-residual disparity? |
| `13_temporal_alignment.ipynb` | R (`sandwich`, `lme4`, via `%%R`) | ⚠️ Not yet run | Handles static (density/topology) vs. time-varying (reliability) mediators correctly |

**Run order:** `07` first (sets up R for every notebook after it, in that
notebook's own session — Colab doesn't share R installs across notebook
tabs, so each of 08/09/10/11/13 also includes its own R setup cell so they
work independently too). Then `08` through `13` in numerical order.

## What "tested" means here

Every cell marked ✅ was actually executed successfully before being handed
to you — the printed outputs shown in those notebooks are real, not
illustrative. Cells marked ⚠️ were written against real, verified package
documentation (confirmed from `faircause`'s own authors' vignettes, not
assumed) but could not be executed in the environment that built this
package (no R/CRAN network access there). Run them in Colab and check the
first output carefully — a couple of cells have explicit notes flagging
where a field name might need adjusting once you see `faircause`'s real
output structure.

## Key finding baked into notebooks 09, 10, 11

`faircause` expects a binary/categorical treatment (`x0`/`x1` reference
levels — their own examples use `"male"`/`"female"`). Sensor density is
continuous here, so every R notebook discretizes it via median split
(`density_bin`). This is flagged explicitly in-notebook, not hidden in a
default you'd only notice later.

## Dataset used throughout

`witgaw/METR-LA` (Hugging Face) — same dataset in all 13 notebooks. Load via
`datasets.load_dataset("witgaw/METR-LA")`, not `hf_hub_download` (the auto-
converted parquet lives on a separate branch — this is already handled
correctly in notebook 01). Graph companion files (`adj_mx.npy`,
`adj_mx_mapping.json`, `distances.csv`, `sensor_locations.csv`) come from the
same repo's `METR-LA/sensor_graph/` folder.
