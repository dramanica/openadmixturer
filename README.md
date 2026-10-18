# openadmixturer

`openadmixturer` is an R/Rcpp reimplementation of
[OpenADMIXTURE.jl](https://github.com/OpenMendel/OpenADMIXTURE.jl) (Ko et al.
2023), a fast algorithm for estimating individual ancestry proportions (Q)
and ancestral allele frequencies (P) under the ADMIXTURE model. It runs on
the CPU, multithreaded with OpenMP.

Genotypes are read directly from the memory-mapped backing file of a
[`tidypopgen`](https://github.com/EvolEcolGroup/tidypopgen) `gen_tibble`,
respecting any filtering of individuals and selection of loci, so no
intermediate PLINK files are written. Results are returned as `gt_admix`
objects, so all of `tidypopgen`'s methods (`autoplot()`, `get_q_matrix()`,
`tidy()`, `augment()`, `c()`, `summary()`) work out of the box.

## Installation

```r
# install.packages("remotes")
remotes::install_github("dramanica/openadmixturer")
```

## Usage

```r
library(tidypopgen)
library(openadmixturer)

bed_file <- system.file("extdata", "lobster", "lobster.bed",
                        package = "tidypopgen")
lobsters <- gen_tibble(bed_file, backingfile = tempfile("lobsters"),
                       quiet = TRUE)
lobsters <- lobsters %>% group_by(population)

adm <- gt_openadmixture(lobsters, k = 2:4, n_runs = 2, seed = c(1, 2),
                        n_cores = 4)
autoplot(adm, type = "barplot", k = 3, run = 1)
```

## Algorithm

As in OpenADMIXTURE.jl:

1. random initialisation followed by a few EM (FRAPPE) iterations;
2. block-wise Newton updates of Q (per individual, on the simplex) and P
   (per locus, in the unit box), each solved exactly as a small quadratic
   program with K. Lange's sweep-operator active-set method;
3. quasi-Newton acceleration (Zhou, Alexander and Lange 2011) using the last
   three secant pairs, accepted only when it improves the log-likelihood.

Differences from the Julia implementation:

- there is no GPU (CUDA) support;
- pseudohaploid individuals are supported (they contribute one allele per
  locus);
- with missing data, the EM update of Q is normalised by the number of
  observed loci of each individual, which keeps Q on the simplex;
- sparse K-means AIM selection (SKFR) is not (yet) implemented.

P gives the frequency of the **alternate** allele of each locus.

## Validation

On the `EUR_subset` data used in the OpenADMIXTURE.jl tests (379 individuals,
54,051 SNPs) with K = 4, `gt_openadmixture()` converges in 23 iterations to a
log-likelihood of -15,552,194.2, compared with -15,552,193.9 reported by
OpenADMIXTURE.jl. The scripts used for this validation are in `data-raw/`:
run `data-raw/download_eur_subset.R` first to download the data.

The vignette (`vignette("openadmixturer")`) compares `gt_openadmixture()`
with the original ADMIXTURE program, installed via
[`tidygenclust`](https://github.com/EvolEcolGroup/tidygenclust), on the
*Anolis punctatus* dataset distributed with `tidypopgen`.
