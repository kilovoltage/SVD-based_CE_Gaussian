# SVD-based Causal Emergence for Gaussian Iterative Systems

Causal emergence (CE) based on effective information (EI) shows that macro-states can exhibit stronger causal effects than micro-states in dynamics. However, the identification of CE and the maximization of EI both rely on coarse-graining strategies, which is a key challenge. A recently proposed CE framework based on approximate dynamical reversibility utilizing singular value decomposition (SVD) is independent of coarse-graining but is limited to transition probability matrices (TPM) in discrete states. To address this problem, this article proposes a pioneering CE quantification framework for Gaussian iterative systems (GIS), based on approximate dynamical reversibility derived from SVD of covariance matrices in forward and backward dynamics. The positive correlation between SVD-based and EI-based CE, along with the equivalence condition, are given analytically. After that, we can provide precise coarse-graining strategies directly from singular value spectrums and orthogonal matrices. This new framework can be applied to any dynamical system with continuous states and Gaussian noise, such as auto-regressive growth models, Markov Gaussian systems, and even SIR modeling by neural networks (NN). Numerical simulations on typical cases validate our theory and offer a new approach to studying the CE phenomenon, emphasizing noise and covariance over dynamical functions in both known models and machine learning.

This set of code can help us handle complex calculations, especially matrix related operations such as singular value decomposition, inverse matrix, determinant, and directly apply them to the computation of effective information and approximate reversibility quantification of causal emergence. At the same time, we also provide neural networks that can be used to handle unknown models, expanding the application scope of our framework.

![image](https://github.com/user-attachments/assets/34640c3d-f099-4cfd-a6f2-82dd66de50be)

## 1.Delta_Gamma.ipynb

Suppose $x_{t+1}=a_0+Ax_t+\varepsilon_t,\varepsilon_t\sim\mathcal{N}(0,\Sigma)$ as the transitional probability from $x_t\in\mathcal{R}^n$ to $x_{t+1}\in\mathcal{R}^n$ in GIS, its $\alpha$-ordered singular value spectrum is:
$\zeta^\alpha(\omega)=\{{\rm det}(\Sigma)^{\frac{1}{2}}{\rm pdet}(A^T\Sigma^{-1} A)^{\frac{1}{2}}\}^{-\frac{\alpha}{2}}\exp\{-\frac{\alpha}{2}\left(\omega^T(A^T\Sigma^{-1}A)^{\dagger}\omega\right)\}$

$\omega\in\mathcal{R}^n$, the $\alpha$-ordered approximate dynamical reversibility of $p(x_{t+1}|x_t)$ is defined as: 
$\Gamma_\alpha=\left(\frac{2\pi}{\alpha}\right)^\frac{n}{2}{\rm pdet}(A^T\Sigma^{-1}A)^{\frac{1}{2}-\frac{\alpha}{4}}{\rm det}(\Sigma^{-1})^\frac{\alpha}{4}$

We can use **gamma** to calculate the approximate reversibility index after removing the constant term, and average it in dimensions. **gamma0** can be used to calculate clear causal emergence. 

**clear_causal_emergence** and **vague_causal_emergence** can help us calculate SVD-based CE directly, we only need to know the gradient **A** and covariance **Sigma** of the system's dynamics.

![image](https://github.com/user-attachments/assets/101d3782-b64b-48eb-a473-e23caf767254)

The example contains 4 variables, in which the first two variables $x_1,x_2$ follow the Malthusian growth model \cite{Galor2000} with different growth rates of 0.2 and 0.05. To study CE, we define the other two variables $x_3,x_4$ as the copies of the first two variables shown as Fig.\ref{fig:Known_model}a, thus, they are redundant dimensions. If $x=(x_1,x_2,x_3,x_4)$, the evolution of $x$ is a GIS $x_{t+1}=a_0+Ax_t+\varepsilon_t$, $\varepsilon_t\sim\mathcal{N}(0,\sigma^2 I_4)$ as $x_t,x_{t+1}\in\mathcal{R}^{4}$, $\sigma^2=0.1$, $a_0=0$, and
$A = \begin{matrix}1.2 & 0 &0 &0 \\0 & 1.05 &0 &0 \\1.2 & 0 &0 &0 \\ 0 & 1.05 &0 &0  \end{matrix}$.

## 2.Gaussian_Markov.ipynb

![image](https://github.com/user-attachments/assets/6b8d5624-97af-42d3-859f-fecf2816a2c5)

In addition to directly copying, the system also has information redundancy if there are several dimensions with strong correlations. In this situation, CE will also be evident. A classic case here is the Markov Gaussian system , which can be seen as a Markov chain in a continuous state space. The example systems are characterized by the dynamics $x_{t+1} =Ax_t+\varepsilon_t$, where $x_t$ contains $n$ variables, $A={(a_{ij})}$ is the connectivity matrix and $\varepsilon_t\sim\mathcal{N}(0,I_n)$. $0\leq a_{ij}\leq 1$ reflects whether there is a connection between $i$ and $j$ and the strength of the connection. In this way, the system's state fluctuates within a stable region, and the Markovian Gaussian system can model EEG data to study consciousness-related issues. The relevant data can be transformed into a 0-1 time series and studied using TPM, but the information is lost.

We considered a system with $n=8$ and connectivity as shown in Figure. Connection strengths between the first seven nodes are $1/7$ and the eighth node is a self-ring with a connection strength of 1. To avoid $\gamma_\alpha=0$ caused by singular values all being 1, we can add some perturbations to $A$.

## 3.SIR_Example.ipynb

![image](https://github.com/user-attachments/assets/23b01463-28fa-4e1b-bda1-223b10a10396)

SIR trained by the original neural network. Most systems in reality are unable to obtain precise dynamic models to calculate analytical solutions for CE as demonstrated in the previous two examples. However, we can train a neural network to obtain approximate dynamics by observed time series data. Our third case is to show the phenomenon of CE obtained by a well-trained neural network (NN) on the training time series data generated by a Susceptible-Infected-Recovered (SIR) dynamical model

To generate the time series data of the micro-state, we adopt the same method in our previous work of NIS+\cite{Yang2024}. We generate data by converting $\mathrm{d}S/\mathrm{d}t,\mathrm{d}I/\mathrm{d}t$ into $\Delta S/\Delta t,\Delta I/\Delta t$ as $\Delta t = 0.01$ and $(S_{t+\Delta t},I_{t+\Delta t})\approx(S_{t},I_{t})+(\mathrm{d}S_t/\mathrm{d}t,\mathrm{d}I_t/\mathrm{d}t)\Delta t$. Then we duplicate the macro-state $(S_t,I_t)$ as shown in Fig.\ref{fig:SIR}c and added Gaussian random noise to form the micro-state $x_t$

## 4.SIR_Cov_example.ipynb

![image](https://github.com/user-attachments/assets/ea43a254-7128-414a-9b74-99539c585e2a)

By feeding the micro-state data into a forward neural network (NN) called \textbf{Covariance Learner Network}, we can use this model to approximate the micro-dynamics of the SIR model. The model we trained has the following structure:

$\bullet$ Input layer: The NN has $n=4$ input neurons, corresponding to the size of the input vector $x_t$.

$\bullet$ Hidden layers: The network contains two hidden layers. The first hidden layer has $hiddensize=64$ neurons, followed by a $Leaky ReLU$ activation function. The second hidden layer also has $hiddensize=64$ neurons, followed by another $Leaky ReLU$ activation. 

$\bullet$ Output layer: The output layer contains two parts. The first part, $f_{\mu}$, outputs a mean vector $\mu$ of size $n$. The second part, $f_L$, outputs the elements of the lower triangular part of the Cholesky decomposition of the covariance matrix, with size $n\times n$.
