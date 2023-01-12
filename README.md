## GULP-GAP interface 
version 1.4.2

✅ Aug 2022: v1.0.0 preparing trainig data using GULP<br>
✅ Aug 2022: v1.1.0 GAP training<br>
✅ Sep 2022: v1.2.0 dimer curve (Al-F pairwise interaction energy vs interatomic distance)<br>
✅ Oct 2022: v1.3.0 separate the prep, train, vis scripts<br>
✅ Oct 2022: v1.3.1 vis script: RDF<br>
✅ Nov 2022: v1.3.2 vis script: histogram instead of RDF<br>
✅ Dec 2022: v1.4.0 prep: degeracy filter<br>
✅ Dec 2022: v1.4.1 prep: symmetric (duplicate) configuration filter<br>
✅ Dec 2022: v1.4.2 prep: extremely short interatomic dist filter [degeneracyFilter branch]<br>

✅ Jan 2022: v2.0.0 Highly modulised scripts (version 2)

=======
* * *
[linke to the image](https://github.com/DGKang234/ML_POT/blob/degeneracyFilter/parameter_search/htmls)
![alt text](https://github.com/DGKang234/ML_POT/blob/degeneracyFilter/parameter_search/htmls/GAP_r-cutoff-energy.png)
![alt text](https://github.com/DGKang234/ML_POT/blob/degeneracyFilter/parameter_search/htmls/GAP_sparse-cutoff-energy_F-F.png)
>>>>>>> degeneracyFilter
* * *
How to use the code: <br>
```python GULP_GAP.py '{eigvec 1} {eigvec 2}...{eigvec n}' {step size for the lambda} {from what IP rank structure} {to what IP rank structure} {n or y (whether including the breathing config or not)} {GAP cutoff parameter} {GAP n_sparce parameter} {degeneracy/duplicate filter}```
<br>
**Example**: ```python GULP_GAP.py '7 9 12' 10 1 10 n 3.0 100 y```

```GAP_1_gulp.py```: Preparing IP structures (IP optimised structures, breathing configurations, IP optimised structures + (eigenvector*λ) <br>
It will consider degenerate frequencies to select only one frequency which is to avoid the overfitting the potential energy landscape. <br>
(It will select first frequency among the degenerate frequencies for the training. For example, 7 and 8 are degenerate, then it will use 7th frequency(eigenvalue/eigenvector))
<br>
***N.B.*** At the moment 100 % of dataset will be distributed as a training dataset if you want to distribute some to validation/test set please change from **line 133**.

```GAP_2_fit.py```: Fitting the GAP potential on the prepared GULP training data

```GAP_3_vis.py```: Visualise the potential from QUIP output using ```plotly```; potential energy VS interatomic potential
(it shows the ideal potentials that we want to reproduce, GAP potentials (cation-anion and anion-anion), RDF of the training data configurations, transparent-boxed interatomic distance region (green/red) that covered by the training data (cation-anion/anion-anion) 

```gap_3_vis_hist.py```: same as ```GAP_3_vis.py``` but visualise interatomic distance in histogram manner rather than RDF.
<br>
<br>
*The script will be further modulised for numerous purpose. If you have any questions about the project please ask me*
