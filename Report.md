# Reproducing CopresheafSAGE on MUTAG

## A Reproduction of Table 3 from *Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework*

**Student:** Mete Yilmaz  
**Paper:** Mustafa Hajij, Lennart Bastian, Sarah Osentoski, Hardik Kabaria, John L. Davenport, Sheik Dawood, Balaji Cherukuri, Joseph G. Kocheemoolayil, Nastaran Shahmansouri, Adrian Lew, Theodore Papamarkou, and Tolga Birdal  
**Venue:** NeurIPS 2025  
**arXiv:** 2505.21251  
**Paper link:** https://arxiv.org/abs/2505.21251  
**OpenReview link:** https://openreview.net/forum?id=3G56xClPYg  

---

## 1. Introduction

In this project, I reproduced part of Table 3 from the paper *Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework*. I focused on the comparison between GraphSAGE and CopresheafSAGE on the MUTAG molecular graph classification dataset. I selected this experiment because it directly tests one of the main practical claims of the paper while remaining realistic to reproduce with the available time and computational resources.

The paper reports a test accuracy of **0.689 ± 0.022** for GraphSAGE and **0.732 ± 0.029** for CopresheafSAGE. I implemented both models in PyTorch Geometric and trained them using five paired random 80/20 train-test splits. I followed the reported settings as closely as possible: two graph-convolution layers, hidden dimension 32, global mean pooling, Adam with a learning rate of 0.01, batch size 16, negative log-likelihood loss, and 50 training epochs.

My implementation obtained **0.716 ± 0.060** for GraphSAGE and **0.726 ± 0.048** for CopresheafSAGE. Therefore, the CopresheafSAGE mean was approximately **1.05 percentage points** higher than the GraphSAGE mean. The direction of the result agrees with the paper, and the CopresheafSAGE mean is close to the value reported by the authors. However, the improvement was smaller than the paper's reported 4.3 percentage-point improvement and was not statistically conclusive with only five runs. I therefore consider this a successful architectural and directional reproduction, but not an exact numerical replication.

---

## 2. Paper and Reproduction Target

I reproduced the **GraphSAGE** and **CopresheafSAGE** rows of **Table 3**.

The reported results are:

| Model | Reported test accuracy |
|---|---:|
| GraphSAGE | 0.689 ± 0.022 |
| CopresheafSAGE | 0.732 ± 0.029 |

The reported mean improvement is:

\[
0.732 - 0.689 = 0.043.
\]

Therefore, the paper reports an improvement of **4.3 percentage points**.

I did not attempt to reproduce the entire table or the entire paper. Table 3 also contains GCN, CopresheafGCN, GIN, and CopresheafGIN. Reproducing all six models would increase the implementation and debugging workload, while the GraphSAGE comparison already provides a direct test of the central copresheaf transport idea.

---

## 3. Dataset

### 3.1 MUTAG

MUTAG is a molecular graph classification dataset. Each graph represents a chemical compound.

- Nodes represent atoms.
- Edges represent chemical bonds.
- Each node has a seven-dimensional feature vector encoding atom type.
- Each graph has one of two labels, representing whether the compound is mutagenic or non-mutagenic.

The dataset contains:

- **188 graphs**
- **7 node features**
- **2 classes**

The full dataset contains 63 graphs from class 0 and 125 graphs from class 1. Therefore, the dataset is not perfectly balanced.

### 3.2 Data splitting

The paper states that it uses an 80/20 train-test split. I followed this protocol.

For each seed:

- 150 graphs were used for training.
- 38 graphs were used for testing.

The split was randomly generated using the corresponding seed. Most importantly, GraphSAGE and CopresheafSAGE used the **same split for the same seed**. This creates a paired comparison and avoids giving one model an easier test set.

I did not use stratified splitting because the paper only states a random 80/20 split and does not state that the split was stratified. As a result, the class distribution differs slightly between seeds.

For example:

- Seed 0 test split: 13 class-0 graphs and 25 class-1 graphs.
- Seed 1 test split: 17 class-0 graphs and 21 class-1 graphs.
- Seed 3 test split: 15 class-0 graphs and 23 class-1 graphs.

This variation is one reason that accuracy can change across seeds.

---

## 4. Implementation, Assumptions, Limitations and Corrections

### 4.1 Software and hardware

The implementation was written in Python using:

- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- SciPy
- Matplotlib

The final experiment environment was:

| Component | Version or value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| PyTorch Geometric | 2.8.0 |
| Operating system | Linux |
| Device | CUDA |
| GPU | Tesla T4 |

### 4.2 Implementation correction

During implementation, I noticed that the paper applies `tanh` to the learned transport update:

```text
Delta_ij = tanh(Linear([h_i ; h_j]))
```

This detail is important because `tanh` bounds every diagonal update between -1 and 1. Therefore, each diagonal entry of \(I+\Delta_{ij}\) is between 0 and 2. Without this operation, the transport could scale features without a bound and would not faithfully follow the paper.

I corrected the implementation before running experiments.

### 4.3 Epsilon assumption

The paper writes the self-feature contribution as:

\[
(1+\epsilon)h_i.
\]

However, the paper does not clearly state the value of \(\epsilon\) or whether it is trainable in this experiment. I fixed:

\[
\epsilon = 0.
\]

This makes the self contribution equal to \(h_i\). I recorded this explicitly by NOTE_TO_GRADER keyword as an assumption.

### 4.4 Parameter counts

The trainable parameter counts are:

| Model | Trainable parameters |
|---|---:|
| GraphSAGE | 2,626 |
| CopresheafSAGE | 4,811 |

The relative increase is:

\[
\frac{4811-2626}{2626}\times100
\approx 83.2\%.
\]

The extra parameters come from the edge-conditioned transport networks in the CopresheafSAGE layers. This is important when interpreting the result because the enhanced model has both a different inductive bias and greater capacity.

However, The 83.2% increase creates a capacity confound.
A fair extension would increase the GraphSAGE hidden dimension until its parameter count is close to the copresheaf model.


### 4.5 Exact seeds are unknown

The paper states the number of runs but does not publish the exact random seeds or split indices. I used seeds 0 through 4. Different splits can produce different accuracy values on MUTAG.

### 4.6 Not fully Specified Architecture Details

- the value of \(\epsilon\),
- whether \(\epsilon\) is learned,
- all bias settings,
- exact initialization,
- every projection placement,
- whether any hidden regularization was used.

### 4.7 Only five runs

Five runs match the number stated in the paper for GraphSAGE models, but five observations provide limited statistical power.

A 20- or 50-seed experiment would give a more stable estimate of the mean and confidence interval.

---

## 5. Results

### 5.1 Main comparison

| Model | Paper result | Reproduction result |
|---|---:|---:|
| GraphSAGE | 0.689 ± 0.022 | **0.716 ± 0.060** |
| CopresheafSAGE | 0.732 ± 0.029 | **0.726 ± 0.048** |

The CopresheafSAGE mean is close to the reported value:

\[
|0.726 - 0.732| = 0.006.
\]

The GraphSAGE baseline is higher than the reported paper value:

\[
0.716 - 0.689 = 0.027.
\]

Because the reproduced baseline is stronger, there is less room for the enhanced model to show the same 4.3 percentage-point gain.

The reproduced mean difference is:

\[
0.7263158 - 0.7157895
=
0.0105263.
\]

Therefore, CopresheafSAGE improves the reproduced mean by approximately **1.05 percentage points**.

The paper's reported improvement is 4.3 percentage points, so the reproduction preserves the direction of the comparison but not the full reported effect size.

### 5.2 Seed-level results

| Seed | GraphSAGE | CopresheafSAGE | Copresheaf − GraphSAGE |
|---:|---:|---:|---:|
| 0 | 0.8158 | 0.7895 | -0.0263 |
| 1 | 0.6842 | 0.7105 | +0.0263 |
| 2 | 0.7105 | 0.7632 | +0.0526 |
| 3 | 0.6579 | 0.6842 | +0.0263 |
| 4 | 0.7105 | 0.6842 | -0.0263 |

CopresheafSAGE:

- performed better in 3 runs,
- performed worse in 2 runs,
- tied in 0 runs.

The largest positive improvement was at seed 2:

\[
0.7632 - 0.7105 = 0.0526,
\]

which corresponds to two more correctly classified graphs out of 38.

The negative differences at seeds 0 and 4 correspond to one fewer correct prediction.

### 5.3 Final training loss

| Seed | GraphSAGE final train NLL | CopresheafSAGE final train NLL |
|---:|---:|---:|
| 0 | 0.4892 | 0.4565 |
| 1 | 0.4549 | 0.3836 |
| 2 | 0.4441 | 0.4337 |
| 3 | 0.4040 | 0.3555 |
| 4 | 0.4365 | 0.4214 |

CopresheafSAGE obtained a lower final training NLL in every run. This suggests that the larger and more flexible model fitted the training data more easily. However, a lower training loss does not automatically imply better generalization. In seeds 0 and 4, CopresheafSAGE had a lower training loss but lower test accuracy.

This observation supports the need to distinguish training fit from test performance.

## 6. Analysis

The paired differences are:

\[
[-0.0263,\ 0.0263,\ 0.0526,\ 0.0263,\ -0.0263].
\]

Their mean is:

\[
\bar{d} = 0.0105.
\]

The sample standard deviation of the paired differences is approximately:

\[
s_d = 0.0353.
\]

The 95% t-based confidence interval for the mean difference is:

\[
[-0.0333,\ 0.0544].
\]

In percentage-point form, this is:

\[
[-3.33,\ +5.44].
\]

The interval includes zero. Therefore, these five runs do not provide enough evidence to conclude that CopresheafSAGE reliably outperforms GraphSAGE.

The paired t-test produced:

\[
p = 0.5415.
\]

This p-value is much larger than 0.05. I do not interpret it as evidence that the models are equal. Instead, it means that the current experiment is too small and variable to establish a statistically reliable difference.

With only five paired observations, the statistical power is low. The confidence interval is therefore more useful than presenting the p-value alone because it shows the range of improvement values compatible with the experiment.

---

## 7. Reproduction Results

### 7.1 Directional reproduction

This part was successful at the mean level. The paper reports:

\[
\text{CopresheafSAGE} > \text{GraphSAGE}.
\]

My mean results also satisfy:

\[
0.726 > 0.716.
\]

Therefore, the mean direction agrees with the paper.

### 7.2 Exact numerical replication

This part was not fully achieved, but close results are obtained 

The paper reports a 4.3 percentage-point gain, while I observed a 1.05 percentage-point gain. My standard deviations were also larger:

| Model | Paper SD | Reproduction SD |
|---|---:|---:|
| GraphSAGE | 0.022 | 0.060 |
| CopresheafSAGE | 0.029 | 0.048 |

The exact difference is not surprising because the paper does not provide all random seeds or full source code for this experiment. On a dataset as small as MUTAG, changing only one or two predictions can noticeably change the reported mean.

My conclusion is that the reproduction supports the plausibility of the reported direction, but it does not independently confirm the size or consistency of the paper's improvement.

## 8. Conclusion

In this project, I reproduced the GraphSAGE and CopresheafSAGE comparison from Table 3 of *Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework*.

The paper reports:

- GraphSAGE: **0.689 ± 0.022**
- CopresheafSAGE: **0.732 ± 0.029**

My reproduction obtained:

- GraphSAGE: **0.716 ± 0.060**
- CopresheafSAGE: **0.726 ± 0.048**

The CopresheafSAGE mean was **1.05 percentage points** higher than GraphSAGE. It performed better on three of five paired splits and worse on two. The result therefore agrees with the direction of the paper's mean comparison, and the reproduced CopresheafSAGE accuracy is close to the reported number.

---

## Appendix A: Complete Run Information

| Seed | Model | Test accuracy | Final train NLL | Parameters | Train class counts | Test class counts |
|---:|---|---:|---:|---:|---|---|
| 0 | GraphSAGE | 0.8158 | 0.4892 | 2,626 | {0: 50, 1: 100} | {0: 13, 1: 25} |
| 0 | CopresheafSAGE | 0.7895 | 0.4565 | 4,811 | {0: 50, 1: 100} | {0: 13, 1: 25} |
| 1 | GraphSAGE | 0.6842 | 0.4549 | 2,626 | {0: 46, 1: 104} | {0: 17, 1: 21} |
| 1 | CopresheafSAGE | 0.7105 | 0.3836 | 4,811 | {0: 46, 1: 104} | {0: 17, 1: 21} |
| 2 | GraphSAGE | 0.7105 | 0.4441 | 2,626 | {0: 50, 1: 100} | {0: 13, 1: 25} |
| 2 | CopresheafSAGE | 0.7632 | 0.4337 | 4,811 | {0: 50, 1: 100} | {0: 13, 1: 25} |
| 3 | GraphSAGE | 0.6579 | 0.4040 | 2,626 | {0: 48, 1: 102} | {0: 15, 1: 23} |
| 3 | CopresheafSAGE | 0.6842 | 0.3555 | 4,811 | {0: 48, 1: 102} | {0: 15, 1: 23} |
| 4 | GraphSAGE | 0.7105 | 0.4365 | 2,626 | {0: 50, 1: 100} | {0: 13, 1: 25} |
| 4 | CopresheafSAGE | 0.6842 | 0.4214 | 4,811 | {0: 50, 1: 100} | {0: 13, 1: 25} |

---

## Appendix B: Main Assumptions

1. Seeds 0–4 were used because the paper does not provide exact seeds.
2. Epsilon was fixed to zero because its value and trainability are not specified.
3. Both models use separate root and neighbor projections.
4. The split is random rather than stratified.
5. Test accuracy is measured after 50 epochs without selecting the best test epoch.
6. Sample standard deviation is reported across runs.
7. The same split is used for both models for every seed.

---

## References

1. Mustafa Hajij et al. *Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework*. NeurIPS 2025. https://arxiv.org/abs/2505.21251

2. William L. Hamilton, Rex Ying, and Jure Leskovec. *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.

3. Christopher Morris et al. *TUDataset: A Collection of Benchmark Datasets for Learning with Graphs*. ICML Workshop on Graph Representation Learning and Beyond, 2020.

4. Matthias Fey and Jan E. Lenssen. *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
