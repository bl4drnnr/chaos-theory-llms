# Lab 2 — Complex Systems Informatics

## Goal  
To investigate whether the process of text generation (e.g. by a language model or a recursive algorithm) exhibits features of a dynamical system sensitive to initial conditions, and to empirically estimate its Lyapunov exponent or other instability measures.

## Theoretical Context  
During the lecture we discussed the notion of deterministic chaos and the Lyapunov exponent, classically defined as

$$
\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{ \| \delta x(t) \| }{ \| \delta x(0) \| }
$$

where $\langle \cdot \rangle$ denotes the average separation of trajectories over time.  
In this project, the text serves the role of a “trajectory” in the space of linguistic symbols. Different versions of the same motif (with minimal perturbation) correspond to nearby points in the phase space.

## Research Task  

1. Prepare two very similar text seeds (e.g. two sentences differing by one word, character or mood).

2. Generate two long texts (at least 500 words) in the same style using a language model (GPT, Llama, etc.).

3. Compute the divergence of the textual trajectories:  
   - split both texts into tokens (e.g. words),  
   - compute Levenshtein edit distance for growing prefixes,  
   - obtain the divergence series $d(k)$.

   i. Does the divergence grow exponentially, linearly, or does it saturate?  
   ii. How long does the system remain in a “stable phase”?  
   iii. How do seed length and amount of external prior knowledge influence divergence?  
   iv. Comment that a small perturbation is relatively smaller for a long seed than for a short one.

4. Estimate the effective Lyapunov exponent and a normalized version. A standard discrete empirical analogue is

$$
\lambda_{\mathrm{eff}}(k) = \frac{1}{k} \ln \frac{ d(k)}{ d(0)}
$$

and the normalized variant used in text-trajectory setting

$$
\Lambda(k) = \frac{1}{k}\,\ln\!\bigl(d(k)+1\bigr)
$$

Plot $\Lambda(k)$ or $\lambda_{\mathrm{eff}}(k)$ as a function of $k$. Interpret how $d(k)$ depends on the generated length.

5. Apply the metaphor of an iterated map (e.g. logistic map)

$$
x_{n+1} = r\,x_n(1-x_n)
$$

to the analysis of chaos in text generation (semantic trajectory) and relate the observed phenomena to the above experiments.

6. Evaluation depends on how rigorously the experiment is executed and on demonstrating analogies between language generation and chaotic numeric trajectories.

7. Original approaches to the objective will be graded particularly highly.

## Literature  

Rodríguez, E. Q. *Towards a Theory of Chaos in Large Language Models: The Prompt as Initial Condition and the Quest for a Semantic Attractor.*  
https://www.researchgate.net/profile/Elio-Quiroga/publication/396159200_Towards_a_Theory_of_Chaos_in_Large_Language_Models_The_Prompt_as_Initial_Condition_and_the_Quest_for_a_Semantic_Attractor/links/68dfb4d3ffdca73694b52eec/Towards-a-Theory-of-Chaos-in-Large-Language-Models-The-Prompt-as-Initial-Condition-and-the-Quest-for-a-Semantic-Attractor.pdf

Li, X., Leng, Y., Ding, R., Mo, H., & Yang, S. (2025). *Cognitive activation and chaotic dynamics in large language models: A quasi-lyapunov analysis of reasoning mechanisms.* arXiv:2503.13530  
https://arxiv.org/pdf/2503.13530
