# NASA C-MAPSS Turbofan Engine Degradation Dataset

Facts below are tagged by source:

- **[readme]**: `data/raw/cmapss/readme.txt`, shipped with the data
- **[paper]**: `data/raw/cmapss/Damage Propagation Modeling.pdf` (Saxena et al., 2008)
- **[NASA]**: the [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) page, entry 6
- **[files]**: computed by us from the data files

Statements marked *inference* are our interpretation and are not stated by any source.

## 1. Source and citation

The dataset is published by the NASA Ames Prognostics Center of Excellence (PCoE) as
"6. Turbofan Engine Degradation Simulation" [NASA].

Cite the paper describing the simulation [readme]:

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine
> Run-to-Failure Simulation", in *Proceedings of the 1st International Conference on Prognostics
> and Health Management (PHM08)*, Denver, CO, Oct 2008.

NASA also gives a citation for the dataset itself [NASA]:

> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA
> Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

**How we download it.** Run [`src/predmaint/data/download.py`](../src/predmaint/data/download.py):

```bash
uv run python -m predmaint.data.download
```

The module does the following:

- downloads the official archive from NASA's PCoE repository (the URL is in the module);
- checks it against a pinned SHA-256 checksum
  (`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`, 12,429,152 bytes);
- unpacks the nested `CMAPSSData.zip` into `data/raw/cmapss/`.

Re-running it skips work that has already been done, and `data/` is not committed to git.

## 2. Licence and terms of use

**No licence is stated.** The readme and the paper contain no licence or terms of use. The NASA
repository page states only that:

- publications using the data "are requested to acknowledge both the assistance received by using
  this repository and the donators of the data";
- "Users employ the data at their own risk. Neither NASA nor the donators of the data sets assume
  any liability for the use of the data, or any system developed using the data."

We cite both references above and make no further assumptions about licensing.

## 3. What it simulates

- **Simulator.** C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), a
  MATLAB/Simulink model of a large commercial turbofan engine in the 90,000 lb thrust
  class [paper].
- **A fleet run to failure.** Each time series is a different engine of the same type, so the
  data can be treated as a fleet [readme]. Each engine starts with an unknown amount of initial
  wear and manufacturing variation, which is considered normal [readme]. The paper bounds this
  at 1% degradation, so each health index starts between 0.99 and 1 [paper].
- **How a fault develops.** Each engine operates normally at first and develops a fault at some
  point. In the training data the fault grows until system failure [readme]. Degradation is
  simulated as an exponentially growing loss of flow and efficiency in the affected
  module [paper].
- **What counts as failure.** Failure is reached when a health index hits zero. The health index
  is the minimum of normalised operability margins (fan, HPC and LPC stall margins, and EGT).
  The health index itself is *not* included in the data [paper].
- **Fault modes.** HPC (high-pressure compressor) degradation, and in FD003/FD004 also fan
  degradation [readme]. The paper only describes HPC degradation, because it documents the PHM08
  challenge data, which NASA says is "similar" to this dataset [NASA]. The fan fault mode is
  documented only in the readme.
- **Operating conditions.** Three operational settings have a substantial effect on engine
  performance and are included in the data [readme]. The paper describes six flight conditions
  combining altitude (0–42K ft), Mach number (0–0.84) and throttle resolver angle (TRA,
  20–100) [paper].
- **Noise.** The data is contaminated with sensor noise [readme]. The paper models process noise
  as a mixture of two normal distributions, plus measurement noise on every output. Maintenance
  between flights is not modelled explicitly; it is folded into the process noise [paper].
- **Sampling.** Each row is one snapshot per operational cycle [readme], taken during or right
  after a flight [paper].

## 4. The four subsets

| Subset | Operating conditions [readme] | Fault modes [readme] | Train engines [files] | Test engines [files] | Train rows [files] | Test rows [files] |
|---|---|---|---|---|---|---|
| FD001 | 1 (sea level) | HPC | 100 | 100 | 20,631 | 13,096 |
| FD002 | 6 | HPC | 260 | 259 | 53,759 | 33,991 |
| FD003 | 1 (sea level) | HPC, fan | 100 | 100 | 24,720 | 16,596 |
| FD004 | 6 | HPC, fan | 249 | 248 | 61,249 | 41,214 |

Trajectory lengths in cycles, as min–max [files]:

| Subset | Train (full run to failure) | Test (truncated) | True test RUL (`RUL_FD00x.txt`) |
|---|---|---|---|
| FD001 | 128–362 | 31–303 | 7–145 |
| FD002 | 128–378 | 21–367 | 6–194 |
| FD003 | 145–525 | 38–475 | 6–145 |
| FD004 | 128–543 | 19–486 | 6–195 |

Notes:

- **The readme swaps the FD004 counts.** It lists 248 train and 249 test trajectories, but the
  files contain 249 train and 248 test engines. `RUL_FD004.txt` has 248 values, which agrees
  with the files. All other counts match the readme.
- **The conditions are visible in the data** [files]:
  - FD001 and FD003 have one condition. The settings are approximately (0, 0, 100).
  - FD002 and FD004 have the same six distinct combinations, in both train and test. Rounded,
    they are (0, 0.00, 100), (10, 0.25, 100), (20, 0.70, 100), (25, 0.62, 60), (35, 0.84, 100)
    and (42, 0.84, 100).
- *Inference:* these values fit the paper's ranges if settings 1, 2 and 3 are altitude (thousand
  ft), Mach number and TRA. The sources do not state this mapping.

## 5. File format and columns

The archive contains `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` for each subset,
plus `readme.txt` and the paper PDF.

**Format** [files]:

- plain text, numbers only, **no header row**;
- values separated by **single spaces**;
- every line ends with **two trailing spaces** and a Unix newline. A parser that splits on one
  literal space produces two extra empty columns, so split on whitespace instead
  (for example `sep=r"\s+"` in pandas).

**Columns:** every data row has exactly 26 values [files].

| Column | Content [readme] |
|---|---|
| 1 | unit (engine) number |
| 2 | time, in cycles |
| 3–5 | operational settings 1–3 |
| 6–26 | sensor measurements 1–21 |

- **The readme miscounts the sensors.** It lists sensor measurements up to "26", which conflicts
  with its own statement of 26 columns. The files confirm 21 sensors (26 − 5).
- **IDs and cycles** [files]. Unit numbers run from 1 to N in each file. Each engine's cycles run
  1, 2, 3, … with no gaps.
- **Constant sensors** [files]. Some sensors never change within a subset. In FD001 these are
  sensors 1, 5, 10, 16, 18 and 19; in FD003, sensors 1, 5, 16, 18 and 19. In FD002 and FD004 no
  sensor is constant, because readings vary with the operating condition.
- **`RUL_FD00x.txt` format** [files]. One integer per line (followed by a trailing space), one
  line per test engine, in unit order.

## 6. How the test set works

- **Training data.** Trajectories run until system failure [readme]. *Derivation:* the RUL label
  at cycle *t* of a training engine is its last cycle minus *t*.
- **Test data.** Trajectories end some time before failure [readme].
- **The task.** Predict the number of operational cycles the engine will keep running after the
  last observed cycle [readme].
- **`RUL_FD00x.txt`.** Gives this true RUL for each test engine. Line *i* belongs to unit *i*.
  That means only **one label per test engine**, at its last cycle.
- **The PHM08 score** [paper]. The PHM08 challenge scored predictions with an asymmetric
  exponential function of the error *d* = predicted RUL − true RUL. Late predictions are
  penalised more than early ones:
  - s = Σ (e^(−d/13) − 1) for *d* < 0;
  - s = Σ (e^(d/10) − 1) for *d* ≥ 0.

  The paper's equation (11) gives a1 = 10 and a2 = 13, but it pairs a1 with *d* < 0, so its
  formula as printed would penalise early predictions more. That contradicts its own text and
  Figure 9. The formula above follows the text and figure.

## 7. Known limitations

- **Simulated, not real.** All data comes from a simulation model. The noise models were *assumed*
  because no real data was available to characterise true noise levels [paper]. Results may not
  transfer to real engines.
- **Small test sets.** There are 100–259 test engines per subset, and each has only one true-RUL
  label [files]. Test metrics therefore rest on few points and can vary noticeably between models.
- **No sensor names or units.** Columns are only numbered [readme, files]. The paper's Table 2
  lists 21 sensor variables (e.g. T24, T30, Nf, Nc, Ps30) with units [paper]. *Inference:* matching
  sensor *k* to row *k* of Table 2 is common practice, but no source confirms that order.
- **Operational settings are unlabelled.** The meaning of settings 1–3 is inferred, not stated
  (see section 4).
- **Fan fault mode documented only in the readme.** The paper does not document the fan
  degradation in FD003/FD004 [paper, readme].
- **Errors in the documentation.** The readme swaps the FD004 counts, lists sensors up to "26",
  and contains typos ("trjectories"). Prefer counts computed from the files.
- **Simplified dynamics.** There is one snapshot per cycle, and maintenance between flights is not
  modelled explicitly [paper].
