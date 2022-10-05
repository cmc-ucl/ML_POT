## ML_POT
GULP-GAP interface - version 1.1.0
* * *
```GAP_1_gulp.py```: Preparing IP structures (IP optimised structures, breathing configurations, IP optimised structures + (eigenvector*λ)

```GAP_2_fit.py```: Fitting the GAP potential on the prepared GULP training data

```GAP_3_vis.py```: Visualise the potential from QUIP output; potential energy VS interatomic potential
(it shows the ideal potentials that we want to reproduce, GAP potentials (cation-anion and anion-anion), RDF of the training data configurations, transparent-boxed interatomic distance region (green/red) that covered by the training data (cation-anion/anion-anion) 

```gap_3_vis_hist.py```: same as ```GAP_3_vis.py``` but visualise interatomic distance in histogram manner rather than RDF.
