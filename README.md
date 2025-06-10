# Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights

**Official repository of the paper presented at CVPR 2025**

**Authors**: [Ondřej Týbl](https://fel.cvut.cz/en/faculty/people/33156-ondrej-tybl) [Lukáš Neumann](https://cmp.felk.cvut.cz/~neumann/projects.html)

**Paper**: [CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Tybl_Training-free_Neural_Architecture_Search_through_Variance_of_Knowledge_of_Deep_CVPR_2025_paper.pdf)

We are proud members of the [VRG at Czech Technical University in Prague](https://vrg.fel.cvut.cz). Join us if you are interested in cutting-edge research in neural networks and computer vision! [Click for more info.](https://cmp.felk.cvut.cz/~neumann/projects.html)

![Open positions](positions.png)

## Abstract

Deep learning has revolutionized computer vision, but it achieved its tremendous success using deep network architectures which are mostly hand-crafted and therefore
likely suboptimal. Neural Architecture Search (NAS) aims
to bridge this gap by following a well-defined optimization
paradigm which systematically looks for the best architecture, given objective criterion such as maximal classification accuracy. The main limitation of NAS is however its
astronomical computational cost, as it typically requires
training each candidate network architecture from scratch.
In this paper, we aim to alleviate this limitation by
proposing a novel training-free proxy for image classification accuracy based on Fisher Information. The proposed
proxy has a strong theoretical background in statistics and
it allows estimating expected image classification accuracy
of a given deep network without training the network, thus
significantly reducing computational cost of standard NAS
algorithms.
Our training-free proxy achieves state-of-the-art results
on three public datasets and in two search spaces, both
when evaluated using previously proposed metrics, as well
as using a new metric that we propose which we demonstrate is more informative for practical NAS applications.

## Method
- Training-free NAS proxy called \textit{Variance of Knowledge of Deep Network Weights (VKDNW)}.
- Strong theoretical support and achieved state-of-the-art results on three public datasets and in two search spaces.
- VKDNW provides information orthogonal to network size unlike previous methods (this helps to easily factor the network performance into \textit{size} and \textit{shape} proxies).
- Zero-cost network ranking where contributions of network size and architectural feasibility are separated.
- We demonstrated that previously used correlation metrics for proxy evaluation do not sufficiently assess the key ability to discriminate top networks.
- To address this, we proposed a new evaluation metric Normalized Discounted Cumulative Gain (nDCG).

## Results

\begin{table}
    \centering    
\begin{tabular}{lc|ccc|ccc|ccc}
\hline
 & & \multicolumn{3}{c|}{CIFAR-10} & \multicolumn{3}{c|}{CIFAR-100} & \multicolumn{3}{c}{ImageNet16-120} \\
 & Type & KT & SPR & $\text{nDCG}$ & KT & SPR & $\text{nDCG}$ & KT & SPR & $\text{nDCG}$ \\
 \hline
 \multicolumn{10}{c}{Simple rankings}\\
 \hline
FLOPs & S & 0.623 & 0.799 & 0.745 & 0.586 & 0.763 & 0.576 & 0.545 & 0.718 & 0.403 \\
GradNorm [1] & S  & 0.328 & 0.438 & 0.509 & 0.341 & 0.451 & 0.278 & 0.310 & 0.418 & 0.265 \\
GraSP [1, 42] & S  & 0.352 & 0.505 & 0.518 & 0.349 & 0.498 & 0.284 & 0.359 & 0.502 & 0.281 \\
SNIP [1, 23] & S  & 0.431 & 0.591 & 0.513 & 0.440 & 0.597 & 0.286 & 0.389 & 0.521 & 0.286 \\
SynFlow [1, 41] & S  & 0.561 & 0.758 & 0.709 & 0.553 & 0.750 & 0.594 & 0.531 & 0.719 & 0.511 \\
Jacov [1] & S  & 0.616 & 0.800 & 0.540 & 0.639 & 0.820 & 0.402 & 0.602 & 0.779 & 0.356 \\
NASWOT [31] & S  & 0.571 & 0.762 & 0.607 & 0.607 & 0.799 & 0.475 & 0.605 & 0.794 & 0.490 \\
ZenNAS [26] & S  & 0.102 & 0.103 & 0.120 & 0.079 & 0.072 & 0.115 & 0.091 & 0.109 & 0.073 \\
GradSign [49] & S  & $\cdot$ & 0.765 & $\cdot$ & $\cdot$ & 0.793 & $\cdot$ & $\cdot$ & 0.783 & $\cdot$ \\
ZiCo [25] & S  & 0.607 & 0.802 & 0.751 & 0.614 & 0.809 & 0.607 & 0.587 & 0.779 & 0.523 \\
TE-NAS [6] & A  & 0.536 & 0.722 & 0.602 & 0.537 & 0.723 & 0.327 & 0.523 & 0.709 & 0.330 \\
AZ-NAS [21] & A  & 0.712 & 0.892 & 0.749 & 0.696 & 0.880 & 0.549 & 0.673 & 0.859 & 0.534 \\
No. of trainable layers ($\aleph$) & S & 0.626 & 0.767 & 0.671 & 0.646 & 0.787 & 0.525 & 0.623 & 0.764 & 0.497 \\
$\text{VKDNW}_{\text{single}}$ (ours)& S  & 0.618 & 0.815 & 0.751 & 0.634 & 0.829 & 0.617 & 0.622 & 0.814 & 0.608 \\
$\text{VKDNW}_{\text{agg}}$ (ours)& A  & \textbf{0.750} & \textbf{0.919} & \textbf{0.785} & \textbf{0.753} & \textbf{0.919} & \textbf{0.636} & \textbf{0.743} & \textbf{0.906} & \textbf{0.664} \\
 \hline
 \multicolumn{11}{c}{Model-driven rankings}\\
 \hline
GRAF [14] & A & 0.820 & 0.953 & 0.935 & 0.809 & 0.948 & 0.858 & 0.796 & 0.941 & 0.828 \\
$\text{VKDNW}_{m}$ (ours) & A & 0.647 & 0.831 & 0.750 & 0.636 & 0.821 & 0.602 & 0.611 & 0.798 & 0.575 \\
$\text{(VKDNW+ZCS)}_{m}$ (ours) & A & 0.840 & 0.963 & 0.922 & 0.834 & 0.960 & 0.884 & 0.830 & 0.958 & 0.843 \\
$\text{(VKDNW+ZCS+GRAF)}_{m}$ (ours) & A & \textbf{0.859} & \textbf{0.971} & \textbf{0.946} & \textbf{0.847} & \textbf{0.966} & \textbf{0.895} & \textbf{0.842} & \textbf{0.963} & \textbf{0.867}
\end{tabular}
\caption{Training-free NAS methods in the NAS-Bench-201 [10] search space, evaluated on three public datasets. Kendall's $\tau$ (KT), Spearman's $\rho$ (SPR) and Normalized Discounted Cumulative Gain ($\text{nDCG}$) are reported, results are averages of 5 independent runs. The Type column differentiates single (S) and aggregated (A) rankings. NAS-Bench-201 dataset includes 15,625 networks with validation accuracies on CIFAR-10, CIFAR-100, and ImageNet16-120 after 200 training epochs; networks have unique cell structures given by graph operations.}
\end{table}

\begin{table}
  \centering
  \small
  \begin{tabular}{@{}lcccc@{}}
    \toprule
    Method & FLOPs & Top-1 acc. & Type & Search cost \\
    &&&& (GPU days) \\
    \midrule
    % \multicolumn{5}{c}{450M} \\  \hline
NASNet-B [50] & 488M & 72.8 & MS & 1800 \\
CARS-D [45] & 496M & 73.3 & MS & 0.4 \\
BN-NAS [5] & 470M & 75.7 & MS & 0.8 \\
OFA [4] & 406M & 77.7 & OS & 50 \\
RLNAS [48] & 473M & 75.6 & OS & - \\
DONNA [32] & 501M & 78.0 & OS & 405 \\
\# Params & 451M & 63.5 & ZS & 0.02 \\
ZiCo [25] & 448M & 78.1 & ZS & 0.4 \\
AZ-NAS [21] & 462M & 78.6 & ZS & 0.4 \\
$\text{VKDNW}_{\text{agg}}$ (ours) & 480M & \textbf{78.8} & ZS & 0.4 \\
    \bottomrule
  \end{tabular}
  \caption{Results on ImageNet-1K [9] in the MobileNetV2 search space, the size of the model is constrained to $\approx$450M FLOPS. We present performance of the network chosen by different proxies and trained afterwards on 480 epochs.
  }
\end{table}

## Install and run

TODO
