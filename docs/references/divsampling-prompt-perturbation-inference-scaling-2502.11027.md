# Diversified Sampling Improves Scaling LLM Inference

**Tianchun Wang**[* 1] **Zichuan Liu**[* 2] **Yuanzhou Chen**[* 3] **Jonathan Light**[4] **Haifeng Chen**[5] **Xiang Zhang**[1] **Wei Cheng**[5] 

## Abstract

While increasing training compute has significantly improved the performance of large language models (LLMs), similar gains have not been observed when scaling inference compute. We hypothesize that the primary issue lies in the uniformity of LLM outputs, which leads to inefficient sampling as models repeatedly generate similar but inaccurate responses. Motivated by an intriguing relationship between solution accuracy (Pass@10) and response diversity, we propose DivSampling—a novel and versatile sampling technique designed to enhance the diversity of candidate solutions by introducing prompt perturbations. DivSampling incorporates two categories of perturbations: task-agnostic approaches, which are general and not tailored to any specific task, and task-specific approaches, which are customized based on task content. Our theoretical analysis demonstrates that, under mild assumptions, the error rates of responses generated from diverse prompts are significantly lower compared to those produced by stationary prompts. Comprehensive evaluations across various tasks — including reasoning, mathematics, and code generation — highlight the effectiveness of DivSampling in improving solution accuracy. This scalable and efficient approach offers a new perspective on optimizing test-time inference, addressing limitations in current sampling strategies. 

## 1. Introduction

The debate between investing resources in training stronger models versus developing effective inference-time methods reflects a fundamental trade-off in the field of machine learning. Improvements through training typically involve 

> *Equal contribution 1The Pennsylvania State University 

> 2Nanjing University 3University of California, Los Angeles 

> 4Rensselaer Polytechnic Institute 5NEC Laboratories America. Correspondence to: Jonathan Light _<_ lij@rpi.edu _>_ , Wei Cheng _<_ weicheng@nec-labs.com _>_ . 

scaling up model size, enhancing training datasets, or incorporating domain-specific fine-tuning. While these approaches can significantly boost performance, they focus on producing a single optimal solution and often come with substantial resource costs. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** DivS 3 DivSampling 1 Small DivSampling 2 Probability (a) Direct Sampling (b) Diversified Sampling

_Figure 1._ A brief sketch of (a) direct sampling without perturbing prompts and (b) diversified sampling. 

Conversely, test-time scaling (Zeng et al., 2024; Wu et al., 2024; Nori et al., 2024; Snell et al., 2024; Brown et al., 2024; Gandhi et al., 2024; Snell et al., 2025; Lee et al., 2025; Wang et al., 2025), such as best-of-N sampling (Cobbe et al., 2021; Lightman et al., 2023), aims to maximize the utility of pretrained models, enabling efficient exploration and improved accuracy without additional training. Repeated sampling generally selects the best solution from a diverse set of candidate responses. However, sampling solutions from a large language model (LLM) using the same prompt often leads to similar outputs, “trapped” into a local cluster (Figure 1(a)). The concentrated nature of the generated solutions might arise from the limited diversity inherent in the post-training objectives commonly employed to train large language models (LLMs) as instruction-following chatbots by optimizing zero-shot performance (Xiang et al., 2025). These objectives often prioritize optimizing the model to produce a single, correct answer (Xiang et al., 2025), which mismatches with the goal of repeated sampling. The commonly used distillation technique may also diminish model diversity (Cideron et al., 2024; DeepSeek-AI et al., 2025). Diverse candidate 

_Table 1._ **Effects of different injection strategies.** 10 solutions were generated using gpt-3.5-turbo for each strategy on the HumanEval benchmark. 

|**Strategies**|**Pass@10 tf-idf sim. BERT sim. lev. sim. seq. sim.**|
|---|---|
|None<br>Role<br>Instruction<br>Jabberwocky|0.2050<br>0.4687<br>0.9964<br>0.5842 0.5370<br>0.3150<br>0.4301<br>0.9961<br>0.5348 0.4944<br>0.3050<br>0.3545<br>0.9947<br>0.4821 0.4100<br>0.3100<br>0.4279<br>0.9967<br>0.5309 0.4815|

solutions should span multiple clusters, with responses distributed across a broader solution space, breaking out of local clusters (Figure 1(b)). A natural approach to achieving this is to inject diversity into the prompts. Table 1 shows the Best-of-N (BoN) using diversified prompts, with different metrics: (a) the **Pass@k** rate measures the performance of tasks for which the correct solution is found; (b) the diversity of solutions is measured by a series of similarity metrics, including **tf-idf** , **BERT** , **Levenshtein** , and **token sequence** (see Appendix B for more details on the metrics). The diversity strategies include Role, Instruction, and Jabberwocky perturbations, representing different styles of prompt injection to promote varied responses. We refer to these strategies as task-agnostic approaches (Section 3.2). Table 1 shows that the pass rate improves when injection strategies produce candidate solutions with reduced similarity. 

In this paper, we propose two categories of perturbations, dubbed DivSampling ( **Div** ersified **Sampling** ). These perturbations modify the prompt distribution, encouraging the generative model to produce a diverse set of candidate solutions, thereby improving the quality of the best selection in repeated sampling. In addition to the taskagnostic approaches of Role, Instruction, and Jabberwocky, we introduce two groups of task-specific methods: Random Idea Injection (RandIdeaInj), designed to generate high-level candidate ideas for a given task, and Random Query Rephraser (RandQReph), which restates the original question. Our methodology builds upon the following observation: 

_Maximizing the generation of diverse yet relevant answers from LLMs can significantly enhance the pass performance of scaling inference._ 

We also show theoretically that our prompt perturbation framework reduces the Pass@k or EM@k error rate by a substantial ratio, which is linear in the number of prompts _k_ (see Sec. 4 and A for details). 

Our empirical findings show that these approaches significantly increase solution accuracy for repeated sampling. RandIdeaInj achieves relative improvements of 13.5% in EM@10 on reasoning tasks, 15.5% in EM@10 on mathematics tasks, and 15.4% in Pass@10 on code generation tasks. When combined with task-specific perturbations, it demonstrates a 75.6% relative improvement in Pass@10 for 

code generation. Similarly, RandQReph delivers a 63.4% relative improvement when restating the question and 29.3% relative improvement through back-translation. 

## 2. Background

### 2.1. Problem Description

We consider sets of tasks defined by a tuple _⟨_ **p** _, Q, V ⟩_ of an instruction prompt **p** , a distribution _Q_ over the question set and a verifier _V_ . For a solver of the task, the **prompt p** and a **question q** sampled from the distribution _Q_ ( _·_ ) are given, from which the solver predicts an **answer s** . This answer is finally judged by the **verifier** _V_ ( **s** _|_ **p** _,_ **q** ), which assigns 1 to accepted answers and 0 to rejected answers. Specifically, we inspect the following scenarios under this framework. 

**Reason & MATH.** In reasoning tasks and math tasks, the prompt **p** asks the solver to choose answer **s** from an answer set **A** for some question **q** _∼Q_ , and the verifier _V_ simply checks if the answer exactly matches the hidden ground truth, which we denote by **H** . 

**Code Generation.** In a program synthesis task, the solver is given a prompt and object pair _⟨_ **p** _,_ **o** _⟩_ in natural language with **o** _∼Q_ , which asks the solver to write code for some object **o** . The goal is to complete the code implementation of **o** such that it passes all hidden tests designed to evaluate its correctness, denoted by **H** . These hidden tests **H** are not visible to the solver under any circumstances. The verifier _V_ considers a solution _s[′]_ to be correct if it passes all hidden tests **H** . 

### 2.2. Best-of-N sampling

Best-of-N, or repeated sampling, involves sampling i.i.d. responses [ **s** ] _N_ := [ **s** 1 _,_ **s** 2 _, ...,_ **s** _N_ ] _∼_ LLM( _·|_ **p** _,_ **q** ) given prompt **p** and question **q** from the LLM solver. Typically, to select a single best answer **s** _[∗]_ from _N_ submissions, one would use a reward model to assign scores to each individual answer. The reward model can be a trained heuristic (Zhang et al., 2024b), self-consistency (Wang et al., 2023c) or an LLM-as-a-judge (Zheng et al., 2023). Since our focus is on diversity injection, we use the ground truth reward model in our experiments where possible. For reasoning and math tasks, A task is considered to be solved if at least one submission exactly matches the ground truth (Wang et al., 2023a); in this case the proportion of tasks that are solved by the LLM solver with _k_ submissions is called the **EM@k rate** . For code generation tasks, a task is solved if at least one submission passes all hidden tests (this is equivalent to selecting the answer that passes the highest number of validation tests (Chen et al., 2024a)); in this case the proportion of tasks that are solved with _k_ submissions is called the **Pass@k rate** (Chen et al., 2021). More details on evaluation metrics can be found in Appendix B. 

## 3. Method

In this section, we present two categories of perturbations aimed at diversifying prompts: task-agnostic and taskspecific approaches. The goal is to modify the input prompt distribution, encouraging the LLM to generate more dissimilar solutions. Task-agnostic approaches are general modifications that are not tailored to any specific prompt, whereas task-specific approaches perturb the prompt based on the content of each task. 

### 3.1. Task-Agnostic Approaches

We introduce three styles of perturbations, aimed at increasing prompt diversity by injecting a randomly sampled sentence from a pool of predefined ones, including Jabberwocky, Role and Instruction injections, in the hope of shifting the model’s focus when generating responses. 

**Jabberwocky** injection randomly select a segment from the poetic “Jabberwocky” to enrich the linguistic diversity of the prompts. 

**Role** injection introduces predefined role-descriptive sentences into prompts to steer the language model’s generation process, guiding them to generate outputs that are tailored to specific roles. The predefined set of roles characterizes the generative model’s descriptive identities, such as “mentor”, “optimizer”, “innovator,” etc. These roles are encapsulated in the original prompt, highlighting the key attributes of each persona. 

**Instruction** injections are a series of steps or guidances that are critical for problem-solving within a domain. By injecting an instruction into the prompt, we aim to guide the model’s processing toward generating outputs that are logical and contextually aligned with the given instruction. Examples of instructions for the code generation task include: 

#### Example Instructions for Code Generation

**Instruction 1:** Write the code in a highly modular way, breaking down functionality into small, reusable components. Each function or class should have a single responsibility, and avoid large monolithic structures. 

**Instruction 2:** Focus on brevity and clarity, minimizing boilerplate code. Use shorthand syntax and built-in functions whenever possible to achieve a minimalist codebase without sacrificing readability. 

**Instruction 3:** Use an object-oriented approach where each concept is modeled as a class. Leverage inheritance, encapsulation, and polymorphism to create a flexible, scalable design. 

### 3.2. Task-Specific Approaches

To provide more meaningful diverse prompts relevant to any specific task, we propose two approaches: the Random Idea Injection (RandIdeaInj) and Random Query Rephraser (RandQReph). 

#### 3.2.1. RANDOM IDEA INJECTION

This group of approaches involves prompting an LLM that may differ from the original generative model, acting as a _thinker_ to generates ideas related to the question. These outputs, serving as perturbations, are then injected into the original prompt to create more targeted and diversified prompts. An example of prompting the LLM for ideas is like: 

#### Prompt to Seek Ideas on Solving the Math Problem

**Prompt:** You are a helpful math tutor with expertise in problem-solving. When given a math question, only provide a concise explanation of the approach or idea needed to solve it. Do not write out every detailed step or final numeric answer. Instead, focus on the method or strategy (like “First, consider the Pythagorean theorem...” or “We can solve by factoring out common terms...”). 

We outline three strategies for selecting the thinker model: 

**Single** -model strategy employs a single model configuration. In this approach, the prompt, along with the problem, is fed directly into the generative model. The generative model acts as the thinker, producing solutions or ideas specifically designed to address the given problem. 

**Dual** -model approach employs a separate, distinct model that serves as a thinker, working alongside the primary generative model. In this setup, the thinker is first prompted to generate ideas before the current iteration of repeated sampling. This dual-system allows for a broader range of generated solutions by incorporating insights from another model. We show examples of ideas generated in response to a question from GSM-Hard using a dual-model strategy: 

#### Example Dual-model Ideas for a Math Question

**Question:** Ruby is 8882014 times older than Sam. In 9 years, Ruby will be 3 times as old as Sam. How old is Sam now? 

**Idea 1:** Let Ruby’s current age be _R_ and Sam’s current age be _S_ . From the problem, we can create two equations based on the relationships given: 1. _R_ = 8882014 _S_ (Ruby is 8882014 times older than Sam) 2. In 9 years, _R_ + 9 = 3( _S_ + 9) (Ruby will be three times Sam’s age in 9 years) 

You can substitute the expression from the first equation into the second equation and solve for _S_ . This will give you Sam’s current age. **Idea 2:** Start by letting Sam’s current age be _x_ . Then, Ruby’s current age can be expressed as 8882014 _x_ . In 9 years, you can set up the equation relating their ages: Ruby’s age in 9 years will be 8882014 _x_ + 9 and Sam’s age will be _x_ + 9. Use the relationship that in 9 years, Ruby’s age will be three times Sam’s age to form the equation: 

8882014 _x_ + 9 = 3( _x_ + 9) From there, solve for _x_ . 

**Diverse** -model approach seeks to generate more varied prompts by drawing from a diverse set of LLMs. Before each iteration of repeated sampling, a thinker model is randomly chosen from the available model options whenever a problem is provided. 

#### 3.2.2. RANDOM QUERY REPHRASER

An alternative approach to diversifying the prompts for any given task is to rephrase the query at each iteration. To accomplish this, we introduce the RandQReph strategy, where an LLM, acting as a narrator, is tasked with rephrasing the input question during each BoN sampling. Similar to RandIdeaInj, this strategy includes three variations: a **single** -model strategy, where the generative model itself rephrases the question; a **dual** -model strategy, where a separate model acts as the rephraser; and a **diverse** -model strategy, where the rephraser is randomly selected from a set of LLMs for each iteration. The rephrased question **q** _[′] k_ replaces the original question **q** _k_ , forming the query pair ( **p** _,_ **q** _[′] k_[)][ at the] _[ k]_[-th sampling.][Additionally, query rephrasing] can be achieved through back-translation (Beddiar et al., 2021), a process where the query is translated from the target language back to the source language. This technique generates slightly modified versions of the original text while preserving its core meaning, thereby expanding the dataset with diverse wording while maintaining contextual consistency. 

## 4. Theoretical Analysis

In this section, we analyze the perturbation injection method and present a theoretical result stating its improvement over unperturbed input texts. For technical details and proof of theorem, please refer to Appendix A. 

For notational simplicity, we use **r** = [ **p** _,_ **q** ] to denote concatenated prompt-question pairs, and write **r** _∼R_ := _{_ **p** _} × Q_ with **q** _∼Q_ . To further formalize our setting, consider a prompt perturbation distribution _d_ ( _·_ ) that randomly injects perturbations into **r** = [ **p** _,_ **q** ] to get **r** _[′] ∼ d_ ( **r** ). 

We base our theory on two natural assumptions on prompt perturbation distribution _d_ and prior input distribution _R_ . Our first assumption stipulates that perturbed inputs are reasonably diverse in performance, which comes naturally from the diversity of the perturbed inputs themselves. 

**Assumption 4.1.** The log probability that the response to input **r** fails the verifier $l(\mathbf{r}) = \log \mathbb{P}_{\mathbf{s}\sim\text{LLM}(\cdot|\mathbf{r})}\big[V(\mathbf{s})=0\big]$ has constant-level first and second moments under perturbed distribution **r** _[′] ∼ d_ ( **r** 0) for any original input **r** 0. 

The second assumption requires perturbed prompts to have similar performances compared with unperturbed prompts under simple resampling strategies. It is natural to assume that perturbed inputs have a similar utility compared to unperturbed ones; if not so, one should consider the perturbation harmful and use different perturbation methods. 

**Assumption 4.2.** The Pass@k or EM@k failure rate of the LLM between prompt-question input pairs from the original distribution **r** = ( **p** _,_ **q** ) _∼R_ and the perturbed distribution **r** _[′] ∼ d_ ( **r** ) _,_ **r** _∼R_ have a close-to-1 ratio. 

With these assumptions, we give the following theorem (see detailed proof in Appendix A) quantifying the improvement in Pass@k for perturbation injection. 

**Theorem 4.3.** _Consider sampling original input_ **r** _∼R and perturbed prompt-question pair_ **r** _k ∼ d_ ( **r** ) _for k_ = 1 _, · · · , N . Define Ninj[k][and][N] reg[ k][to][be][the][probabilities][of] generating responses that fail to pass based on inputs with and without injection, respectively. Then_ 

$$
N_{\text{inj}}^{k} \leq N_{\text{reg}}^{k} / C_{k},
$$

_where Ck_ = _O_ ( _k_ ) _is greater than_ 1 _and increasing in k. Remark_ 4.4 _._ The main implications of this theorem are two-fold. First, since _Ck ≥_ 1, perturbed inputs should **always perform better** than non-perturbed inputs. In practice, sometimes the assumptions don’t hold strictly (especially Assumption 4.2 when the perturbation harms prediction), but in general perturbing inputs improve performance. Second, despite the fact that error rate strictly decreases as _k_ increases for both perturbed and unperturbed inputs, the **decrease is substantially faster** for perturbed inputs, meaning larger _k_ ’s result in greater improvements. 

## 5. Experiments

### 5.1. Datasets

We evaluate DivSampling across six benchmarks including reason, math and coding: **(a)** Multiple choice questionsanswering on **MMLU-Pro** (Wang et al., 2024b), a dataset curated by eliminating some trivial and noisy questions from MMLU (Hendrycks et al., 2020) while incorporating more reasoning-focused problems. For evaluation, we randomly select 150 samples from the dataset. **(b)** Math problem-solving on **GSM-hard** (Gao et al., 2023) and **MATH** (Hendrycks et al., 2021b). GSM-Hard increases the computational complexity of GSM8K (Cobbe et al., 2021) by replacing numerical values with larger numbers. MATH consists of competitive-level mathematical problems requiring high levels of reasoning ability and mathematical knowledge. We randomly sample 100 problems from both GSM-Hard and MATH for evaluation. **(c)** Code generation on **Humaneval** (Chen et al., 2021), **MBPP** (Austin et al., 2021) and **APPS** (Hendrycks et al., 2021a). HumanEval includes 164 human-generated Python problems, while MBPP consists of 399 problems covering basic algorithmic and functional programming tasks. APPS features challenging code competition problems. Due to budget constraints, we randomly sample 200 problems from the 10,000 available problems in APPS for evaluation. 

### 5.2. Experiment Details

For simplicity, we configured the models with a temperature of 0.4 for the reasoning dataset MMLU-Pro, a uniform temperature of 0.2 for the math task datasets GSMHard and MATH, and a temperature of 0.6 for all code 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** MMLU-Pro GSM-Hard MATH 0.70 0.70 0.60 0.65 0.65 0.55 0.60 0.60 0.55 0.50 0.55 0.50 None 0.45 None 0.50 None 0.45 Role 0.40 Role 0.45 Role 0.40 Instruction Instruction Instruction Jabberwocky 0.35 Jabberwocky 0.40 Jabberwocky 0.35 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K Humaneval MBPP APPS 0.80 0.84 0.30 0.82 0.75 0.80 0.25 0.78 0.70 0.76 0.20 None None None 0.74 Role Role Role 0.65 Instruction 0.72 Instruction 0.15 Instruction Jabberwocky 0.70 Jabberwocky Jabberwocky 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K EM@K EM@K EM@K Pass@K Pass@K Pass@K

_Figure 2._ EM@k or Pass@k graphs of Role, Instruction, and Jabberwocky methods versus direct sampling across six datasets using GPT-3.5-turbo. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** MMLU-Pro GSM-Hard Humaneval 0.8 0.65 0.85 0.7 0.60 0.80 0.6 0.55 0.75 0.5 None 0.50 None 0.70 None Single Single Single 0.4 DualDiverse 0.45 DualDiverse 0.65 DualDiverse 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K EM@K EM@K Pass@K

_Figure 3._ EM@k or Pass@k graphs of Single, Dual and Diverse strategies of RandIdeaInj versus direct sampling on the MMLU-Pro, GSM-Hard and Humaneval using GPT-3.5-turbo. In the Dual strategy, GPT-4o-mini serves as the thinker. The Diverse method utilizes a set of four models, with GPT-3.5-turbo, GPT-4o-mini (OpenAI, 2023b), and Llama-3.1-8B-Instruct (Meta, 2024) consistently included across all datasets. The fourth model varies by dataset: Qwen2.5-7B-Instruct (Yang et al., 2024a) for MMLU-Pro, Qwen2.5-Math-7B-Instruct (Yang et al., 2024b) for GSM-Hard, and Qwen2.5-Coder-7B-Instruct (Hui et al., 2024) for HumanEval. In each iteration, a thinker is randomly selected from the set of four models. 

generation benchmarks including Humaneval, MBPP, and APPS. These temperature settings were determined through a coarse hyperparameter sweep from _T ∈{_ 0 _._ 0 _,_ 0 _._ 2 _, ...,_ 1 _._ 2 _}_ . In the decoding-phase, we use top- _p_ sampling with a fixed value of 1.0 across all experiments. All method evaluations are allocated the same search budget of 10 solutions. DivSampling is assessed in comparison to the direct sampling without perturbation, which is referred to as **None** across the experiments. We run experiments on a server with 4 NVIDIA A100 GPUs, each one with 80GB RAM. 

### 5.3. Results of Task-Agnostic Approaches

We evaluate the Role, Instruction, and Jabberwocky strategies in Section 3.2 across six benchmarks spanning reasoning, mathematics, and code generation, comparing them to direct sampling without perturbations. Figure 2 shows their 

scaling curves of evaluation on GPT-3.5-turbo (OpenAI, 2023a). We find that these injection strategies yield improvements across all tasks, with a notable 8.6% increase in EM@10 on GSM-Hard and the most significant gains on APPS, achieving approximately a 53.7% improvement in Pass@10 over direct sampling. We encourage readers to check Appendix C for more results of other models. 

### 5.4. Results of Random Idea Injection

The evaluation involves a range of RandIdeaInj strategies in Section 3.2.1, including the single-model approach, dual-model approach, and diverse-model approach, evaluated across the benchmarks MMLU-Pro, GSMHard, and HumanEval. Figure 3 displays the scaling curves of evaluations conducted with the generative model GPT-3.5-turbo. RandIdeaInj exhibits consistent 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Combination of Role and RandIdeaInj Combination of Instruction and RandIdeaInj Combination of Jabberwocky and RandIdeaInj 0.45 None 0.40 None Role Instruction 0.40 Role+Single 0.35 Instruction+Single 0.3 0.35 Role+Dual Instruction+Dual 0.30 Role+Diverse 0.30 Instruction+Diverse 0.2 0.25 None 0.250.20 0.20 0.1 JJabberwocky+Singleabberwocky 0.15 0.15 0.0 Jabberwocky+Dual Jabberwocky+Diverse 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K Figure 4. Pass@k graphs of Role, Instruction, and Jabberwocky, along with their combinations with RandIdeaInj on APPS using GPT-3.5-turbo.. GPT-4o-mini serves as the thinker model in each combination of the Dual strategy. For the verse strategy, each combination has a set of 4 choices: GPT-3.5-turbo,, GPT-4o-mini,, Llama-3.1-8B-Instruct and Qwen2.5-Coder-7B-Instruct, with a thinker model randomly selected from this set in each iteration of repeated sampling., with a thinker model randomly selected from this set in each iteration of repeated sampling. 0.40 None GPT-4o-mini 0.18 None Llama-3.1-8B 0.225 None Qwen2.5-Coder 0.60 Claude-3.5-Sonnet 0.35 RoleRole+Dual 0.160.14 Role Role+Dual 0.2000.175 RoleRole+Dual 0.55 0.12 0.150 0.30 0.50 0.10 0.125 0.25 0.08 0.1000.075 0.45 None Role 0.06 Role+Dual 0.050 0.40 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K K Pass@K Pass@K Pass@K Pass@K Pass@K Pass@K Pass@K

_Figure 4._ Pass@k graphs of Role, Instruction, and Jabberwocky, along with their combinations with RandIdeaInj on APPS using GPT-3.5-turbo.. GPT-4o-mini serves as the thinker model in each combination of the Dual strategy. For the Diverse strategy, each combination has a set of 4 choices: GPT-3.5-turbo,, GPT-4o-mini,, Llama-3.1-8B-Instruct and Qwen2.5-Coder-7B-Instruct, with a thinker model randomly selected from this set in each iteration of repeated sampling., with a thinker model randomly selected from this set in each iteration of repeated sampling. 

_Figure 5._ Expanded Pass@k graphs of Role, along with its combination with Dual strategy in RandIdeaInj using various models. In each Dual strategy combination, GPT-4o-mini serves as the thinker. 

improvement with idea-injected prompts, achieving a 13.5% increase in reasoning on MMLU-Pro, a 15.5% increase in the mathematics on GSM-Hard, and a 15.4% increase in coding on the Humaneval dataset, over the direct sampling. See Appendix D for more results of RandIdeaInj from other models. 

### 5.5. Results of Combining Injection Strategies

We show the Pass@k results for combining Role, Instruction, and Jabberwocky injections with three RandIdeaInj strategies on the APPS dataset, using GPT-3.5-turbo, as shown in Figure 4. We find that combining the injections significantly enhances performance, achieving maximum relative improvements in Pass@10 of 75.6%, 73.2%, and 75.6% over the direct sampling. We extend our evaluation of the combined Role and Dual strategies to additional models, presenting the resulting scaling curves in Figure 5. Notably, the Pass@10 relative improvement reaches 40.0% with Llama-3.1-8B-Instruct. 

### 5.6. Results of Random Query Rephraser

We show Pass@k results of three types of RandQReph in Section 3.2.2 from different models on APPS in Figure 6. The best-performing strategy exhibits an relative improvement in Pass@10 over direct sampling, achieving 11.6% for GPT-4o-mini, 28.0% for Llama-3.1-8B-Instruct, 18.4% for Qwen2.5-Coder-7B-Instruct, and a notable 63.4% for GPT-3.5-turbo. In addition, Figure 7 illustrates the scaling curves of back-translation, which show a 29.3% rela- 

tive improvement in Pass@10 compared to direct sampling. 

### 5.7. Effects of Temperature Sweeping

We show the temperature sweeping of the task-agnostic strategy Role and its combination with Dual, ranging from 0.0 to 1.2 in increments of 0.2, on the APPS dataset in Figure 8. Our findings show that Role achieves Pass@k relative improvements over direct sampling without perturbation, with its best Pass@10 improving by 15.9% compared to the best Pass@10 of direct sampling. Furthermore, Role+Dual enhances performance beyond Role, achieving an 8.6% improvement in its best Pass@10 compared to the best Pass@10 of the Role method. 

### 5.8. Scalability

Multi-round Debate (Du et al., 2023) is a strategy that relies on an additional model or agent to provide a reference answer. In literature, debating also shows effectiveness in improve LLM performance. Intuitively, debating is also one kind of diversity injection in prompt. In the Multi-round Debate (Du et al., 2023), the primary model updates its response in the following round based on that reference, ultimately producing a refined answer. We assess the scalability of our method versus Debate by comparing the proportion of problems solved when both approaches use the same number of output tokens. The evaluation is performed on Humaneval, with GPT-3.5-turbo serving as the generative model. GPT-4o-mini is employed as the thinker model for idea generation in the Dual strategy of RandIdeaInj and as the reference model in the Debate strategy. The re- 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** GPT-3.5-turbo GPT-4o-mini Llama-3.1-8B Qwen2.5-Coder 0.16 0.225 0.30 0.375 0.14 0.25 0.325 0.12 0.175 0.20 0.10 0.15 NRandQReph-Singleone 0.275 None RandQReph-Single 0.08 None RandQReph-Single 0.125 None RandQReph-Single 0.10 RandQRephRandQReph-Diverse-Dual 0.225 RandQReph-DualRandQReph-Diverse 0.060.04 RandQReph-Dual RandQReph-Diverse 0.075 RandQReph-DualRandQReph-Diverse 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K K Pass@K Pass@K Pass@K Pass@K

_Figure 6._ Pass@k graphs on APPS using the models GPT-3.5-turbo, GPT-4o-mini, Llama-3.1-8B-Instruct, and Qwen2.5-Coder-7B-Instruct. The Dual method employs GPT-3.5-turbo as the rephraser for GPT-4o-mini; otherwise, GPT-4o-mini acts as the rephraser. The Diverse method has a set of 4 models: GPT-3.5-turbo, GPT-4o-mini, Llama-3.1-8B-Instruct and Qwen2.5-Coder-7B-Instruct, with a randomly selected rephraser from the set in each iteration of repeated sampling. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Scaling Curves of Back-Translation 0.250 0.225 0.200 0.175 None 0.150 BTrans-Single BTrans-Dual 0.125 BTrans-Diverse 2 4 6 8 10 K Pass@K

_Figure 7._ Pass@k graph of back-translations on APPS using GPT-3.5-turbo. A GPT-4o-mini serves as the translator in the Dual strategy. The Diverse strategy randomly selects a translator model from four options: GPT-3.5-turbo, GPT-4o-mini, Llama-3.1-8B-Instruct, and Qwen2.5-Coder-7B-Instruct. 

sults in Figure 9 tell that the Dual strategy consistently outperforms the Debate strategy when using the same number of output tokens, showing its superior scalability compared to the Debate method. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Temperature Effects on Role vs. None 1.2 0.30 1.0 0.25 0.8 0.6 0.20 0.4 0.15 None 0.2 Role 0.0 1 2 3 4 5 6 7 8 9 10 K Temperature Effects on Role+Dual vs. Role 1.2 0.35 1.0 0.30 0.8 0.25 0.6 0.20 0.4 0.15 Role 0.2 Role+Dual 0.0 1 2 3 4 5 6 7 8 9 10 K Pass@K Temperature Pass@K Temperature

_Figure 8._ Sweep over temperature in 0.2 increments from 0.0 to 1.2 on APPS using GPT-3.5-turbo. Role exhibits Pass@k improvements at higher temperatures, while Role+Dual achieves further improvements. 

### 5.9. Results of DivSampling on top of CoT

We evaluate various injection strategies, including Role, Instruction, Jabberwocky, and their combinations with Dual, applied on top of Chain-of-Thought (CoT) (Wei et al., 2022; Wang et al., 2023b). The performances from GPT-3.5-turbo on the APPS are presented in Figure 10. CoT is implemented by prompting the generative model to break down its solution into a step-by-step manner: 

#### Example CoT Prompt

**Prompt:** When you receive a problem description, methodically break down the implementation into distinct, logical steps within the Python code itself. Use comments within your code to clearly delineate these steps, focusing exclusively on the logic and structure necessary to solve the problem as described. Make sure each part of your solution is self-contained within a Python code block, illustrating the solution’s development in a step-by-step manner... 

Results in Figure 10 show that even simple task-agnostic 

approaches, as well as their combinations with the Dual strategy of RandIdeaInj, lead to improvements over standard CoT. Notably, on top of CoT, DivSampling achieves a 13.1% relative improvement over direct sampling. 

## 6. Additional Related Work

**Scaling Inference Computation** has explored diverse strategies for enhancing LLM capabilities through adaptive test-time compute allocation (Snell et al., 2024; Brown et al., 2024; Manvi et al., 2024; Guan et al., 2025; Chen et al., 2024b). Typically, LLM inference involves decomposing complex questions into sequential intermediate steps that lead to the final answer, exemplified by chain-of-thought (CoT) prompting (Wei et al., 2022; Sprague et al., 2024; Wang & Zhou, 2024) and its variants (Kojima et al., 2022; Zhou et al., 2023; Wang et al., 2023d; Li et al., 2023). How- 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Scaling curve with output tokens 0.8 0.7 0.6 0.5 None Debate(round1) 0.4 Debate(round2) 0.3 Debate(round3) Debate(round4) 0.2 Dual 0 2000 4000 6000 8000 10000 Number of output tokens used Proportion correct

_Figure 9._ Proportion of problems solved vs. number of tokens used. Results are from GPT-3.5-turbo. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** DivSampling on top of CoT 0.35 0.30 0.25 None Role 0.20 Role+Dual Instruction 0.15 Instruction+Dual Jabberwocky 0.10 Jabberwocky+Dual 2 4 6 8 10 K Pass@K

_Figure 10._ Pass@k graphs of DivSampling built upon CoT using GPT-3.5-turbo on APPS. The task-agnostic approaches are combined with the Dual strategy of RandIdeaInj, where GPT-4o-mini acts as the thinker for idea generation. 

**Prompting for Self-improvement** , beginning with STaR (Zelikman et al., 2022), relies on solutions generated by the LLM to augment data in fine-tuning processes. For instance, reinforced self-training methods (Gulcehre et al., 2023; Hosseini et al., 2024a; Singh et al., 2024; Aksitov et al., 2023) introduce mechanisms to curate new highquality examples, mainly by sampled CoT solutions, and then iteratively enrich the training dataset for enhancing LLM capabilities. However, these methods typically rely on either labeled preference data (Gulcehre et al., 2023) or a reward model (Guan et al., 2024; Zelikman et al., 2022; Hosseini et al., 2024a). In contrast, recent work like selfcorrection (Kumar et al., 2024; Zelikman et al., 2024; Hosseini et al., 2024b; Xi et al., 2024) and self-rewarding (Yuan et al., 2024; Chen et al., 2024c; Huang et al., 2024) use LLM themselves to evaluate the generated solutions. In domains with fine-grained sampled CoT step, existing advances training a model-based verifier (Cobbe et al., 2021; Lightman et al., 2023; Wang et al., 2024a; Min et al., 2024) or using tree search to collect preferences (Xie et al., 2024a) decide on a final answer that can improve performance on reasoning tasks relative to taking a single sample. Nevertheless, the strategies still require initial human annotation for finetuning, e.g., seed data (Chen et al., 2024c; Lee et al., 2024). Our self-improvement differs from previous methods since DivSampling does not necessary require any ground truth as ORM, where performance can be improved only by injecting a diverse set of perturbations. Other strategies, such as LLM-as-a-judger, majority voting can also be adopted. 

## 7. Conclusion

ever, with the increasing number of steps in a single chain, these methods often suffer from error propagation and struggle with complex computations (Chen et al., 2023). To overcome the limitation, CoT (Li et al., 2024) has been improved with search methods (Zhang et al., 2024c), such as beam search (Xie et al., 2024c) and Best-of-N (Snell et al., 2024), which leverage reward models to select the most promising candidates. Later, tree search-based algorithms, such as MCTS and A* (Yao et al., 2024b; Luo et al., 2024; Zhang et al., 2024a; Hao et al., 2023; Zhou et al., 2024; Choi et al., 2023; Yao et al., 2024a; Chen et al., 2024d; Xie et al., 2024b; Zhang et al., 2025) produce diversity paths in inference computation, allowing for exploration at different levels. All of these methods show that utilizing inference time techniques for extended search computation leads to performance gains in a variety of tasks. While improved performance through scaling inference comes with increased computational costs (Wu et al., 2025), the diversified sampling remains underexplored. This paper systematically analyzes this relationship and empirically demonstrates that sampling from diversified natural language prompts can mitigate performance shortfalls in reasoning, math, and code generation tasks. 

In this paper, we introduce DivSampling, a novel and generalizable prompt perturbation framework designed to address the inherent limitations of uniform LLM outputs during inference by injecting diversity into prompt-based sampling. By leveraging task-agnostic and task-specific strategies—including Role, Instruction, Jabberwocky, Random Idea Injection, and Random Query Rephraser—DivSampling broadens the distribution of candidate responses, leading to improvements across diverse tasks such as reasoning, mathematics, and code generation. Our empirical evaluation demonstrates significant enhancements in solution accuracy (Pass@k), confirming that prompt diversification effectively breaks the local clustering problem commonly observed in traditional sampling methods. Theoretical analysis further reinforces that increased diversity reduces error rates linearly with the number of diverse prompts. By optimizing test-time inference without additional training, DivSampling offers a scalable, efficient solution for boosting LLM performance and practical applicability in real-world tasks. 

## 8. Impact Statements

This work contributes to the field of LLM by proposing DivSampling, a novel sampling strategy aimed at enhancing the effectiveness of test-time scaling across various reasoning, math, and code generation tasks. By addressing the diversity issue in LLM inference, this technique has the potential to improve computational efficiency, reducing energy consumption associated with redundant computations, and enhancing model performance in real-world applications. No immediate or specific ethical concerns are identified in the context of this research. 

## References

- Aksitov, R., Miryoosefi, S., Li, Z., Li, D., Babayan, S., Kopparapu, K., Fisher, Z., Guo, R., Prakash, S., Srinivasan, P., et al. Rest meets react: Self-improvement for multi-step reasoning llm agent. _arXiv preprint arXiv:2312.10003_ , 2023. 

- Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- Beddiar, D. R., Jahan, M. S., and Oussalah, M. Data expansion using back translation and paraphrasing for hate speech detection, 2021. URL https://arxiv.org/ abs/2106.04681. 

- Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Re,´ C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling. _arXiv preprint arXiv:2407.21787_ , 2024. 

- Chen, G., Liao, M., Li, C., and Fan, K. Alphamath almost zero: process supervision without process. _arXiv preprint arXiv:2405.03553_ , 2024a. 

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _Transactions on Machine Learning Research_ , 2023. 

- Chen, X., Xu, J., Liang, T., He, Z., Pang, J., Yu, D., Song, L., Liu, Q., Zhou, M., Zhang, Z., Wang, R., Tu, Z., Mi, H., and Yu, D. Do not think that much for 2+3=? on the overthinking of o1-like llms, 2024b. URL https: //arxiv.org/abs/2412.21187. 

- Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q. Selfplay fine-tuning converts weak language models to strong 

language models. In _Forty-first International Conference on Machine Learning_ , 2024c. URL https:// openreview.net/forum?id=O4cHTxW9BS. 

- Chen, Z., White, M., Mooney, R., Payani, A., Su, Y., and Sun, H. When is tree search useful for llm planning? it depends on the discriminator, 2024d. URL https: //arxiv.org/abs/2402.10890. 

- Choi, S., Fang, T., Wang, Z., and Song, Y. Kcts: Knowledgeconstrained tree search decoding with token-level hallucination detection. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 14035–14053, 2023. 

- Cideron, G., Agostinelli, A., Ferret, J., Girgin, S., Elie, R., Bachem, O., Perrin, S., and Rame, A.´ Diversity-rewarded cfg distillation, 2024. URL https://arxiv.org/ abs/2410.06084. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, 

- Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948. 

- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. _arXiv preprint arXiv:2305.14325_ , 2023. 

- Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., et al. Codebert: A pre-trained model for programming and natural languages. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pp. 1536–1547, 2020. 

- Gandhi, K., Lee, D., Grand, G., Liu, M., Cheng, W., Sharma, A., and Goodman, N. D. Stream of search (sos): Learning to search in language. _arXiv preprint arXiv:2404.03683_ , 2024. 

- Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023. 

- Guan, X., Liu, Y., Lu, X., Cao, B., He, B., Han, X., Sun, L., Lou, J., Yu, B., Lu, Y., and Lin, H. Search, verify and feedback: Towards next generation post-training paradigm of foundation models via verifier engineering, 2024. URL https://arxiv.org/abs/2411.11504. 

- Guan, X., Zhang, L. L., Liu, Y., Shang, N., Sun, Y., Zhu, Y., Yang, F., and Yang, M. rstar-math: Small llms can master math reasoning with self-evolved deep thinking, 2025. URL https://arxiv.org/abs/2501.04519. 

- Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., et al. Reinforced self-training (rest) for language modeling. _arXiv preprint arXiv:2308.08998_ , 2023. 

- Hao, S., Gu, Y., Ma, H., Hong, J., Wang, Z., Wang, D., and Hu, Z. Reasoning with language model is planning with world model. In _Empirical Methods in Natural Language Processing_ , pp. 8154–8173, 2023. 

- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., et al. Measuring coding challenge competence with apps. _arXiv preprint arXiv:2105.09938_ , 2021a. 

- Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021b. 

- Hosseini, A., Yuan, X., Malkin, N., Courville, A., Sordoni, A., and Agarwal, R. V-star: Training verifiers for self-taught reasoners. _arXiv preprint arXiv:2402.06457_ , 2024a. 

- Hosseini, A., Yuan, X., Malkin, N., Courville, A., Sordoni, A., and Agarwal, R. V-star: Training verifiers for self-taught reasoners, 2024b. URL https://arxiv. org/abs/2402.06457. 

- Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., and Zhou, D. Large language models cannot self-correct reasoning yet, 2024. URL https: //arxiv.org/abs/2310.01798. 

- Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., et al. Qwen2. 5-coder technical report. _arXiv preprint arXiv:2409.12186_ , 2024. 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35: 22199–22213, 2022. 

- Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J. D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., et al. Training language models to self-correct via reinforcement learning. _arXiv preprint arXiv:2409.12917_ , 2024. 

- Lee, K.-H., Fischer, I., Wu, Y.-H., Marwood, D., Baluja, S., Schuurmans, D., and Chen, X. Evolving deeper llm thinking, 2025. URL https://arxiv.org/abs/ 2501.09891. 

- Lee, N., Wattanawong, T., Kim, S., Mangalam, K., Shen, S., Anumanchipalli, G., Mahoney, M. W., Keutzer, K., and Gholami, A. Llm2llm: Boosting llms with novel iterative data enhancement. _arXiv preprint arXiv:2403.15042_ , 2024. 

- Li, Y., Lin, Z., Zhang, S., Fu, Q., Chen, B., Lou, J.-G., and Chen, W. Making language models better reasoners with step-aware verifier. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5315–5333, 2023. 

- Li, Z., Liu, H., Zhou, D., and Ma, T. Chain of thought empowers transformers to solve inherently serial problems, 2024. URL https://arxiv.org/abs/ 2402.12875. 

- Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- Luo, L., Liu, Y., Liu, R., Phatale, S., Guo, M., Lara, H., Li, Y., Shu, L., Zhu, Y., Meng, L., Sun, J., and Rastogi, A. Improve mathematical reasoning in language models by automated process supervision, 2024. URL https: //arxiv.org/abs/2406.06592. 

- Manvi, R., Singh, A., and Ermon, S. Adaptive inferencetime compute: Llms can predict if they can do better, even mid-generation. _arXiv preprint arXiv:2410.02725_ , 2024. 

Meta. Meta-llama 3.1-8b instruct, 2024. URL https://huggingface.co/meta-llama/ Meta-Llama-3.1-8B-Instruct. Accessed: 2024-09-03. 

- Min, Y., Chen, Z., Jiang, J., Chen, J., Deng, J., Hu, Y., Tang, Y., Wang, J., Cheng, X., Song, H., Zhao, W. X., Liu, Z., Wang, Z., and Wen, J.-R. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems, 2024. URL https://arxiv.org/ abs/2412.09413. 

- Nori, H., Usuyama, N., King, N., McKinney, S. M., Fernandes, X., Zhang, S., and Horvitz, E. From medprompt to o1: Exploration of run-time strategies for medical challenge problems and beyond, 2024. URL https://arxiv.org/abs/2411.03590. 

- OpenAI. Gpt-3.5-turbo, 2023a. URL https: //platform.openai.com/docs/models/ gpt-3-5. 

OpenAI. Gpt-4o-mini, 2023b. URL https: //platform.openai.com/docs/models/ gpt-4. 

- Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Garcia, X., Liu, P. J., Harrison, J., Lee, J., Xu, K., et al. Beyond human data: Scaling self-training for problemsolving with language models. _Transactions on Machine Learning Research_ , 2024. 

- Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. _arXiv preprint arXiv:2408.03314_ , 2024. 

- Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling test-time compute optimally can be more effective than scaling LLM parameters. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https: //openreview.net/forum?id=4FWAwZtd2n. 

- Sprague, Z., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa, M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., and Durrett, G. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning, 2024. URL https://arxiv.org/abs/2409.12183. 

- Wang, E., Cassano, F., Wu, C., Bai, Y., Song, W., Nath, V., Han, Z., Hendryx, S., Yue, S., and Zhang, H. Planning in natural language improves LLM search for code generation. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https: //openreview.net/forum?id=48WAZhwHHw. 

- Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. _arXiv preprint arXiv:2312.08935_ , 2023a. 

- Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 9426–9439, 2024a. 

- Wang, X. and Zhou, D. Chain-of-thought reasoning without prompting, 2024. URL https://arxiv.org/abs/ 2402.10200. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023b. URL https: //openreview.net/forum?id=1PL1NIMMrw. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023c. URL https: //openreview.net/forum?id=1PL1NIMMrw. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In _International Conference on Learning Representations_ , 2023d. 

- Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. _arXiv preprint arXiv:2406.01574_ , 2024b. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., brian ichter, Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain of 

thought prompting elicits reasoning in large language models. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), _Advances in Neural Information Processing Systems_ , 2022. 

- Wu, Y., Sun, Z., Li, S., Welleck, S., and Yang, Y. Inference scaling laws: An empirical analysis of computeoptimal inference for problem-solving with language models, 2024. URL https://arxiv.org/abs/ 2408.00724. 

- Wu, Y., Sun, Z., Li, S., Welleck, S., and Yang, Y. Inference scaling laws: An empirical analysis of computeoptimal inference for LLM problem-solving. In _International Conference on Learning Representations_ , 2025. URL https://openreview.net/forum? id=VNckp7JEHn. 

- Xi, Z., Yang, D., Huang, J., Tang, J., Li, G., Ding, Y., He, W., Hong, B., Do, S., Zhan, W., Wang, X., Zheng, R., Ji, T., Shi, X., Zhai, Y., Weng, R., Wang, J., Cai, X., Gui, T., Wu, Z., Zhang, Q., Qiu, X., Huang, X., and Jiang, Y.-G. Enhancing llm reasoning via critique models with test-time and training-time supervision, 2024. URL https://arxiv.org/abs/2411.16579. 

- Xiang, V., Snell, C., Gandhi, K., Albalak, A., Singh, A., Blagden, C., Phung, D., Rafailov, R., Lile, N., Mahan, D., Castricato, L., Franken, J.-P., Haber, N., and Finn, C. Towards system 2 reasoning in llms: Learning how to think with meta chain-of-thought, 2025. URL https: //arxiv.org/abs/2501.04682. 

- Xie, Y., Goyal, A., Zheng, W., Kan, M.-Y., Lillicrap, T. P., Kawaguchi, K., and Shieh, M. Monte carlo tree search boosts reasoning via iterative preference learning. _arXiv preprint arXiv:2405.00451_ , 2024a. 

- Xie, Y., Goyal, A., Zheng, W., Kan, M.-Y., Lillicrap, T. P., Kawaguchi, K., and Shieh, M. Monte carlo tree search boosts reasoning via iterative preference learning, 2024b. URL https://arxiv.org/abs/2405.00451. 

- Xie, Y., Kawaguchi, K., Zhao, Y., Zhao, J. X., Kan, M.Y., He, J., and Xie, M. Self-evaluation guided beam search for reasoning. _Advances in Neural Information Processing Systems_ , 36, 2024c. 

- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. _arXiv preprint arXiv:2412.15115_ , 2024a. 

- Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. Qwen2. 5-math technical report: Toward mathematical expert model via selfimprovement. _arXiv preprint arXiv:2409.12122_ , 2024b. 

- Yao, H., Huang, J., Wu, W., Zhang, J., Wang, Y., Liu, S., Wang, Y., Song, Y., Feng, H., Shen, L., and Tao, D. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search, 2024a. URL https://arxiv.org/abs/2412.18319. 

- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024b. 

- Yuan, W., Pang, R. Y., Cho, K., Li, X., Sukhbaatar, S., Xu, J., and Weston, J. E. Self-rewarding language models. In _International Conference on Machine Learning_ , 2024. 

- Zelikman, E., Wu, Y., Mu, J., and Goodman, N. Star: Bootstrapping reasoning with reasoning. _Advances in Neural Information Processing Systems_ , 35:15476–15488, 2022. 

- Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., and Goodman, N. D. Quiet-star: Language models can teach themselves to think before speaking, 2024. URL https://arxiv.org/abs/2403.09629. 

- Zeng, Z., Cheng, Q., Yin, Z., Wang, B., Li, S., Zhou, Y., Guo, Q., Huang, X., and Qiu, X. Scaling of search and learning: A roadmap to reproduce o1 from reinforcement learning perspective, 2024. URL https://arxiv. org/abs/2412.14135. 

- Zhang, D., Zhoubian, S., Hu, Z., Yue, Y., Dong, Y., and Tang, J. Rest-mcts*: Llm self-training via process reward guided tree search, 2024a. URL https: //arxiv.org/abs/2406.03816. 

- Zhang, L., Hosseini, A., Bansal, H., Kazemi, M., Kumar, A., and Agarwal, R. Generative verifiers: Reward modeling as next-token prediction, 2024b. URL https://arxiv.org/abs/2408.15240. 

- Zhang, X., Du, C., Pang, T., Liu, Q., Gao, W., and Lin, M. Chain of preference optimization: Improving chainof-thought reasoning in llms, 2024c. URL https:// arxiv.org/abs/2406.09136. 

- Zhang, Z., Zheng, C., Wu, Y., Zhang, B., Lin, R., Yu, B., Liu, D., Zhou, J., and Lin, J. The lessons of developing process reward models in mathematical reasoning, 2025. URL https://arxiv.org/abs/2501.07301. 

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https: //arxiv.org/abs/2306.05685. 

- Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., and Wang, Y.-X. Language agent tree search unifies 

reasoning, acting, and planning in language models. In _International Conference on Machine Learning_ , 2024. 

- Zhou, D., Scharli, N., Hou, L., Wei, J., Scales, N., Wang, X.,¨ Schuurmans, D., Cui, C., Bousquet, O., Le, Q. V., et al. Least-to-most prompting enables complex reasoning in large language models. In _International Conference on Learning Representations_ , 2023. 

## Appendix: DivSampling

## A. Theoretical Details

In this section, for clarity, we use upper case letters _K_ to denote an LLM’s total number of attempts, and lower case letters _k_ to denote the attempt subscripts. This is in slight contrast with the main paper where _k_ is used to denote number of attempts when discussing EM@k and Pass@k metrics, and _N_ is used to denote the number of attempts for the final model. 

We first restate Assumption 4.1 and Assumption 4.2 in more technical detail: 

**Assumption A.1** (Restatement of Assumption 4.1) **.** Consider the log probability that the response to an input **r** fails the verifier: 

$$
q(\mathbf{r}) = \log \mathbb{P}_{s\sim\text{LLM}(\cdot|\mathbf{r})}\big[V(s)=0\,\big|\,\mathbf{r}\big],
$$

then the perturbed input distribution _d_ ( **r** ) satisfies that its first and second moments are lower bounded for any input **r** , i.e. there exists constants _µ_ ˆ1 _,_ ˆ _µ_ 2 _>_ 0 such that 

$$
\begin{aligned}
\mathbb{E}_{\mathbf{r}'\sim d(\mathbf{r})}\big|q(\mathbf{r}')-\bar{q}\big| &\geq \hat{\mu}_1, \\
\mathbb{E}_{\mathbf{r}'\sim d(\mathbf{r})}\big(q(\mathbf{r}')-\bar{q}\big)^2 &\geq \hat{\mu}_2,
\end{aligned} \tag{1}
$$

where _q_ ¯ = E **r** _′∼d_ ( **r** ) _q_ ( **r** _[′]_ ) is the mean value. 

**Assumption A.2** (Restatement of Assumption 4.2) **.** The failure rate of the LLM with _K_ attempts between inputs from the original distribution **r** _∼R_ and the perturbed distribution **r** _[′] ∼ d_ ( **r** ) _,_ **r** _∼R_ have a close-to-1 ratio. Specifically, there exists a small constant _ϵ_ such that 

$$
1-\epsilon \leq \frac{\mathbb{E}_{\mathbf{r}'\sim d(\mathbf{r}),\,\mathbf{r}\sim\mathcal{R}}\big[\exp(Kq(\mathbf{r}'))\big]}{\mathbb{E}_{\mathbf{r}\sim\mathcal{R}}\big[\exp(Kq(\mathbf{r}))\big]} \leq 1+\epsilon,
$$

where notice 

$$
\exp(Kq(\mathbf{r})) = \mathbb{P}_{s\sim\text{LLM}(\cdot|\mathbf{r})}^{K}\big[V(s)=0\big]
$$

is the failure rate with repeated sampling on a single input. 

_Remark_ A.3 _._ We actually only need the right hand side, but we include both a lower and an upper bound here for a more comprehensive comparison between perturbed and unperturbed inputs. 

Now we fully state Theorem 4.3 and present its proof. 

**Theorem A.4** (Restatement of Theorem 4.3) **.** _For a testing input distribution R, define_ 

$$
N_{\text{inj}}^K = \mathbb{P}\bigg[V(\mathbf{s}_k)=0,\ \forall k\in[K]\ \bigg|\ \mathbf{s}_k\sim\text{LLM}(\cdot|\mathbf{r}_k),\ \mathbf{r}_k\sim d(\mathbf{r}),\ \forall k\in[K],\ \mathbf{r}\sim\mathcal{R}\bigg] \tag{2}
$$

_and_ 

$$
N_{\text{reg}}^K = \mathbb{P}\bigg[V(\mathbf{s}_k)=0,\ \forall k\in[K]\ \bigg|\ \mathbf{s}_k\sim\text{LLM}(\cdot|\mathbf{r}),\ \forall k\in[K],\ \mathbf{r}\sim\mathcal{R}\bigg] \tag{3}
$$

_to be the probabilities of failing Pass@K for prompts with and without injection, respectively. Then_ 

$$
N_{\text{inj}}^K \leq N_{\text{reg}}^K / C_K,
$$

_where CK_ = _O_ ( _K_ ) _is greater than_ 1 _and increasing in K._ 

_Proof of Theorem 4.3._ First, using the fact that **s** _k_ are i.i.d. samples, we rewrite the two values with log probabilities _q_ ( **r** ): 

$$
\begin{aligned}
N_{\text{inj}}^K &= \mathbb{P}\bigg[V(\mathbf{s}_k)=0,\ \forall k\ \bigg|\ \mathbf{s}_k\sim\text{LLM}(\cdot|\mathbf{r}_k),\ \mathbf{r}_k\sim d(\mathbf{r}),\ \mathbf{r}\sim\mathcal{R}\bigg] \\
&= \mathbb{E}\bigg[\prod_{k=1}^K \mathbb{P}\big[V(\mathbf{s}_k)=0\ \big|\ \mathbf{s}_k\sim\text{LLM}(\cdot|\mathbf{r}_k)\big]\ \bigg|\ \mathbf{r}_k\sim d(\mathbf{r}),\ \mathbf{r}\sim\mathcal{R}\bigg] \\
&= \mathbb{E}\bigg[\prod_{k=1}^K \exp(q(\mathbf{r}_k))\ \bigg|\ \mathbf{r}_k\sim d(\mathbf{r}),\ \mathbf{r}\sim\mathcal{R}\bigg] \\
&= \mathbb{E}\bigg[\exp\bigg(\sum_{k=1}^K q(\mathbf{r}_k)\bigg)\ \bigg|\ \mathbf{r}_k\sim d(\mathbf{r}),\ \mathbf{r}\sim\mathcal{R}\bigg],
\end{aligned} \tag{4}
$$

Next, denoting $q_k = q(\mathbf{r}_k)$, we will draw a connection between $\exp\big(\sum_{k=1}^K q_k\big)$ and $(1/K)\sum_{k=1}^K \exp(Kq_k)$. For this we denote $\bar{q} = (1/K)\sum_{k=1}^K q_k$ and write

$$
\begin{aligned}
\frac{\frac{1}{K}\sum_{k=1}^K \exp(Kq_k)}{\exp\big(\sum_{k=1}^K q_k\big)} &= \frac{1}{K}\sum_{k=1}^K \exp\big(Kq_k - K\bar{q}\big) \\
&= \frac{1}{K}\sum_{k=1}^K \big[g\big(Kq_k - K\bar{q}\big) + \big(Kq_k - K\bar{q}\big) + 1\big] \\
&= 1 + \frac{1}{K}\sum_{k=1}^K g\big(Kq_k - K\bar{q}\big),
\end{aligned} \tag{5}
$$

where _g_ ( _x_ ) = _e[x] − x −_ 1 is a non-negative function. Using basic analysis, one can easily prove _g_ ( _x_ ) _≥_ min _{_ 0 _._ 25 _x_[2] _,_ 0 _._ 5 _|x|}_ for any _x ∈_ R, which gives us from Equation 5 that 

$$
\begin{aligned}
\frac{\frac{1}{K}\sum_{k=1}^K \exp(Kq_k)}{\exp\big(\sum_{k=1}^K q_k\big)} &\geq 1 + \min\bigg\{\frac{0.5}{K}\sum_{k=1}^K \big|Kq_k - K\bar{q}\big|,\ \frac{0.25}{K}\sum_{k=1}^K \big(Kq_k - K\bar{q}\big)^2\bigg\} \\
&= 1 + \min\bigg\{0.5\sum_{k=1}^K \big|q_k - \bar{q}\big|,\ 0.25K\sum_{k=1}^K \big(q_k - \bar{q}\big)^2\bigg\},
\end{aligned}
$$

and so from Assumption 4.1 and the central limit theorem, there exists _cK_ = _O_ ( _K_ ) such that the left hand side above is at least 1 + _cK_ with high probability.[1] Thus continuing from 4, we finally have 

$$
\begin{aligned}
N_{\text{inj}}^K &\leq \mathbb{E}\bigg[\frac{\frac{1}{K}\sum_{k=1}^K \exp(Kq(\mathbf{r}_k))}{1+c_K}\ \bigg|\ \mathbf{r}_k\sim d(\mathbf{r}),\ \mathbf{r}\sim\mathcal{R}\bigg] \\
&= \frac{1}{K(1+c_K)}\sum_{k=1}^K \mathbb{E}\big[\exp(Kq(\mathbf{r}_k))\ \big|\ \mathbf{r}_k\sim d(\mathbf{r}),\ \mathbf{r}\sim\mathcal{R}\big] \\
&\leq \frac{1+\epsilon}{K(1+c_K)}\sum_{k=1}^K \mathbb{E}\big[\exp(Kq(\mathbf{r}_k))\ \big|\ \mathbf{r}_k\sim\mathcal{R}\big] \\
&= \frac{1+\epsilon}{1+c_K}\mathbb{E}\big[\exp(Kq(\mathbf{r}))\ \big|\ \mathbf{r}\sim\mathcal{R}\big] \\
&= \frac{1}{1+(c_K-\epsilon)/(1+\epsilon)}\mathbb{P}^K\bigg[V(\mathbf{s})=0\ \bigg|\ \mathbf{s}\sim\text{LLM}(\cdot|\mathbf{r}),\ \mathbf{r}\sim\mathcal{R}\bigg] \\
&= \frac{1}{1+(c_K-\epsilon)/(1+\epsilon)}\mathbb{P}\bigg[V(\mathbf{s}_k)=0,\ \forall k\in[K]\ \bigg|\ \mathbf{s}_k\sim\text{LLM}(\cdot|\mathbf{r}),\ \forall k\in[K],\ \mathbf{r}\sim\mathcal{R}\bigg] \\
&= \frac{1}{1+(c_K-\epsilon)/(1+\epsilon)}N_{\text{reg}}^K,
\end{aligned}
$$

where the second inequality uses Assumption 4.2. Therefore letting _CK_ = 1 + ( _cK − ϵ_ ) _/_ (1 + _ϵ_ ), we finish the proof of the Theorem. 

## B. Details of Metrics

For each of our metrics, the solver is allowed _k_ submissions for each, denoted by [ **s** ] _k ∼_ LLM( _·|_ **r** _, k_ ) given input **r** . We consider testing the model on a set of tasks consisting of prompts and questions _X_ = _{_ **r** = [ **p** _,_ **q** ] _}_ . **EM@k Rate** . For reasoning and math tasks, if at least one submission _s[′] ∈_ [ **s** ] _k_ matches the ground truth, the task is considered solved. The EM@k rate is defined as the proportion of tasks solved as 

$$
\textbf{EM@k} = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{r}\in\mathcal{X}}\mathbf{1}\big(\exists\,\mathbf{s}\in[\mathbf{s}]_k,\ \text{s.t.}\ \mathbf{s}=\mathbf{H}\ \big|\ [\mathbf{s}]_k\sim\text{LLM}(\cdot|\mathbf{r},k)\big),
$$

1This can be more rigidly proven using mathematical languages from probability theory, but the proof is unnecessarily tedious for our purposes, and the intuition behind such a proof is exactly as explained here. 

where 1 ( _·_ ) is the indicator function and **H** is the ground truth. 

**Pass@k Rate** . For code generation tasks, if at least one submission _s[′] ∈_ [ **s** ] _k_ passes all hidden tests **H** _c_ , the task is considered solved. The Pass@k rate is defined as 

$$
\textbf{Pass@k} = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{r}\in\mathcal{X}}\mathbf{1}\big(\exists\,\mathbf{s}'\in[\mathbf{s}]_k,\ \text{s.t.}\ \mathbf{s}'\text{ passes all }\mathbf{H}_c\ \big|\ [\mathbf{s}]_k\sim\text{LLM}(\cdot|\mathbf{r},k)\big).
$$

**TF-IDF Similarity** measures the importance of terms in a document relative to a collection of documents, which computes the average cosine similarity between TF-IDF representations of solution pairs: 

$$
\textbf{tf-idf sim.} = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{x}\in\mathcal{X}}\frac{1}{k(k-1)}\sum_{\substack{\mathbf{s},\mathbf{s}'\in[\mathbf{s}]_k \\ \mathbf{s}\neq\mathbf{s}'}}\frac{\operatorname{tf\text{-}idf}(\mathbf{s})\cdot\operatorname{tf\text{-}idf}(\mathbf{s}')}{\|\operatorname{tf\text{-}idf}(\mathbf{s})\|\,\|\operatorname{tf\text{-}idf}(\mathbf{s}')\|}.
$$

**BERT Cosine Similarity** is an average cosine score between the embeddings of candidate solution pairs, where embeddings are performed using CodeBERT (Feng et al., 2020), a pre-trained model for understanding code semantically: 

$$
\textbf{BERT sim.} = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{x}\in\mathcal{X}}\frac{1}{k(k-1)}\sum_{\substack{\mathbf{s},\mathbf{s}'\in[\mathbf{s}]_k \\ \mathbf{s}\neq\mathbf{s}'}}\frac{\operatorname{CodeBERT}(\mathbf{s})\cdot\operatorname{CodeBERT}(\mathbf{s}')}{\|\operatorname{CodeBERT}(\mathbf{s})\|\,\|\operatorname{CodeBERT}(\mathbf{s}')\|}.
$$

**Levenshtein Similarity** is based on the Levenshtein distance, which measures the minimum number of single-character edits (insertions, deletions, or substitutions) required to transform one string into another: 

$$
\textbf{lev. sim.} = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{x}\in\mathcal{X}}\frac{1}{k(k-1)}\sum_{\substack{\mathbf{s},\mathbf{s}'\in[\mathbf{s}]_k \\ \mathbf{s}\neq\mathbf{s}'}}\frac{\operatorname{Levenshtein\ Distance}(\mathbf{s},\mathbf{s}')}{\max(|\mathbf{s}|,|\mathbf{s}'|)}.
$$

**Token Sequence Similarity** measures the overlap between two sequences of tokens (e.g., programming language tokens), 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** MMLU-Pro GSM-Hard MATH 0.80 0.68 0.85 0.75 0.66 0.80 0.64 0.70 0.62 0.75 0.65 None 0.60 None None Role Role Role 0.60 Instruction 0.58 Instruction 0.70 Instruction Jabberwocky 0.56 Jabberwocky Jabberwocky 0.65 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K Humaneval MBPP APPS 0.90 0.84 NoneRole 0.375 0.88 Instruction 0.350 0.82 Jabberwocky 0.325 0.86 0.80 0.300 0.84 None 0.78 0.275 None 0.82 Role Instruction 0.76 0.250 RoleInstruction Jabberwocky 0.74 0.225 Jabberwocky 0.80 2 4 6 8 10 2 4 6 8 10 1 2 3 4 5 6 7 8 9 K K K EM@K EM@K EM@K Pass@K Pass@K Pass@K

_Figure 11._ EM@k or Pass@k graphs of Role, Instruction, and Jabberwocky methods versus direct sampling across six datasets using GPT-4o-mini. 

denoted by _T_ ( **s** ) for output **s** : 

$$
\textbf{seq. sim.} = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{x}\in\mathcal{X}}\frac{1}{k(k-1)}\sum_{\substack{\mathbf{s},\mathbf{s}'\in[\mathbf{s}]_k \\ \mathbf{s}\neq\mathbf{s}'}}\frac{|T(\mathbf{s})\cap T(\mathbf{s}')|}{|T(\mathbf{s})\cup T(\mathbf{s}')|}.
$$

*[figure omitted — see original PDF]*

> **Figure text (extracted):** MMLU-Pro GSM-Hard MATH 0.60 0.75 0.7 0.55 0.70 0.6 0.50 0.65 0.45 0.60 0.5 None None None Role 0.40 Role 0.55 Role 0.4 InJabberwockystruction 0.35 Instruction Jabberwocky 0.50 Instruction Jabberwocky 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K Humaneval MBPP APPS 0.75 0.12 0.75 0.70 0.10 0.65 0.70 0.08 None None None 0.60 Role 0.65 Role Role Instruction Instruction 0.06 Instruction 0.55 Jabberwocky Jabberwocky Jabberwocky 0.60 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K EM@K EM@K EM@K Pass@K Pass@K Pass@K

_Figure 12._ EM@k or Pass@k graphs of Role, Instruction, and Jabberwocky methods versus direct sampling across six datasets using Llama-3.1-8B-Instruct. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** MMLU-Pro GSM-Hard Humaneval 0.85 0.66 0.90 0.80 0.88 0.64 0.75 0.86 0.70 0.62 0.84 0.65 None 0.60 None 0.82 None 0.60 Single Single Single Dual 0.58 Dual 0.80 Dual 0.55 Diverse Diverse Diverse 0.78 2 4 6 8 10 1 2 3 4 5 6 7 8 9 2 4 6 8 10 K K K EM@K EM@K Pass@K

_Figure 13._ EM@k or Pass@k graphs of Single, Dual and Diverse strategies of RandIdeaInj versus direct sampling on the MMLU-Pro, GSM-Hard and Humaneval benchmarks using GPT-4o-mini. In the Dual strategy, GPT-3.5-turbo serves as the thinker. The Diverse method utilizes a set of four models, with GPT-3.5-turbo, GPT-4o-mini, and Llama-3.1-8B-Instruct consistently included across all datasets. The fourth model varies by dataset: Qwen2.5-7B-Instruct for MMLU-Pro, Qwen2.5-Math-7B-Instruct for GSM-Hard, and Qwen2.5-Coder-7B-Instruct for HumanEval. In each iteration, a thinker is randomly selected from the set of four models. 

## C. Additional Results of Task-Agnostic Approaches

We evaluate the Role, Instruction, and Jabberwocky strategies across six benchmarks, measuring their EM@k rate for reasoning and math tasks and their Pass@k rate for code generation tasks, in comparison to the direct sampling. The results in Figure 11, generated using GPT-4o-mini, show an relative improvement of 6.2% in EM@10 on the GSM-Hard dataset and 11.6% in Pass@10 on the APPS dataset. The results in Figure 12, generated using Llama-3.1-8B-Instruct, show a 2.8% relative improvement in EM@10 on the MMLU-Pro dataset, a 15.7% relative improvement in EM@10 on GSM-Hard, and a 4.1% relative improvement in Pass@10 on Humaneval. 

## D. Additional Results of Random Idea Injection

We evaluate RandIdeaInj by generating 10 solutions with GPT-4o-mini with each strategy on the MMLU-Pro, GSMHard, and Humaneval datasets. GPT-3.5-turbo serves as the thinker model in the Dual strategy for idea generation. The results, shown in Figure 13, indicate that RandIdeaInj achieves a 6.8% relative improvement in EM@10 on MMLU-Pro and a 4.3% relative improvement in Pass@10 on Humaneval. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** GPT-3.5-turbo GPT-4o-mini Llama-3.1-8B 0.8 0.80 0.7 0.7 0.75 0.6 0.0 0.70 0.0 0.6 0.0 0.2 0.2 0.2 (a)t 0.50.4 0.4 0.6 0.81.0 0.650.60 0.40.60.81.0 0.50.4 0.40.60.81.0 1.2 1.2 1.2 0.55 0.3 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K MMLU-Pro GPT-3.5-turbo GPT-4o-mini Llama-3.1-8B 0.65 0.0 0.70 0.0 0.60 0.0 0.2 0.2 0.2 0.55 0.60 0.4 0.4 0.4 0.6 0.65 0.6 0.6 0.50 0.55 0.8 0.8 0.8 0.50 1.0 1.2 0.60 1.01.2 0.45 1.01.2 0.40 0.45 0.55 0.35 0.40 2 4 6 8 10 2 4 6 8 10 2 4 6 8 10 K K K GSM-Hard GPT-3.5-turbo GPT-4o-mini Llama-3.1-8B 0.85 0.0 0.80 0.2 0.88 0.75 0.4 0.70 0.75 0.60.8 0.86 0.00.2 0.65 0.00.2 0.70 1.0 1.2 0.84 0.4 0.60 0.4 0.6 0.6 0.65 0.82 0.8 0.55 0.8 1.0 1.0 0.60 0.80 1.2 0.50 1.2 2 4 6 8 10 1 2 3 4 5 6 7 8 9 2 4 6 8 10 K K K Humaneval EM@K EM@K EM@K EM@K EM@K EM@K Pass@K Pass@K Pass@K

_Figure 14._ Scaling curves of GPT-3.5-Turbo, GPT-4o-mini, and Llama-3.1-8B-Instruct across different temperatures on the MMLU-Pro, GSM-Hard, and Humaneval benchmark. 

## E. Temperatures

We present the scaling curves for GPT-3.5-Turbo, GPT-4o-mini, and Llama-3.1-8B-Instruct at temperature settings ranging from 0.0 to 1.2 (in increments of 0.2) on the MMLU-Pro, GSM-Hard, and Humaneval benchmarks in Figure 14. 

## F. Examples of Prompt Injections

We show examples of injected-prompts of Jabberwocky, Role and Instruction. 

Example Jabberwocky Poem Injection 

**Jabberwocky 1** : ’Twas brillig, and the slithy toves. Did gyre and gimble in the wabe: **Jabberwocky 2** : All mimsy were the borogoves, And the mome raths outgrabe. **Jabberwocky 3** : Beware the Jabberwock, my son! The jaws that bite, the claws that catch! 

#### Example Role Prompt Injection

#### Roles for Reasoning

**Role 1** : You are a problem solver. You are analytical, logical, detail-oriented. You thrive on tackling complex problems and finding efficient solutions, enjoy the challenge of debugging and often see issues as puzzles to be solved, and are methodical in your approach and persistent in your efforts to overcome obstacles. **Role 2** : You are a pragmatist. You are practical, results-oriented, efficient. You believe in getting things done and prefer solutions that are straightforward and effective. You are less concerned with perfection and more focused on delivering reliable solution. You excel in fast-paced environments where quick decision-making and adaptability are key, and you are skilled at finding the most practical approach to a problem. 

#### Roles for Math

**Role 1** : You are a curious explorer of mathematics. You approach math with wonder and enthusiasm. You’re eager to learn new techniques and test out fresh ideas, always refining your approach. You don’t fear complex or unfamiliar problems but see them as opportunities to expand your understanding. 

**Role 2** : You are a rigorous communicator. You excel at explaining the reasoning behind each step in simple, understandable terms. You guide others through your thought process so they can follow exactly how you arrived at a result. You consider your audience’s perspective and make math accessible. 

#### Roles for Coding

**Role 1** : You are an innovator. You are creative, visionary, adaptable. You are always looking for new ways to apply technology. You are not just interested in how things work but also in how they can be improved or transformed. You enjoy pioneering new techniques and technologies and are comfortable with experimentation and risk-taking. 

**Role 2** : You are a builder. You are hands-on, practical, resourceful. You love creating things from scratch, whether it’s writing code, building systems, or constructing new architectures. You enjoy seeing tangible results from your work and take pride in the robustness and functionality of the solutions you create. You are a maker at heart, always eager to bring ideas to life. 

#### Example Instruction Prompt Injection

#### Instructions for Reasoning

**Instruction 1** : Identify Key Information and Gaps: Note down all pertinent details provided in the scenario. Identify what information is known and what is missing or needs to be inferred. 

**Instruction 2** : Develop a Reasoning Strategy: Choose an appropriate approach to address the question. This might involve logical deduction, applying specific reasoning frameworks, or constructing an argument based on evidence from the text. 

#### Instructions for Math

**Instruction 1** : Check for Assumptions and Constraints: Make sure you understand any conditions, assumptions, or limitations stated in the problem. 

**Instruction 2** : Identify Known and Unknown Variables: Highlight or list all the information given in the problem and determine what needs to be found.: 

#### Instructions for Coding

**Instruction 1** : Write code with explicit, detailed comments and verbose variable/function names. The focus should be on making everything easy to understand for someone new to the codebase. 

**Instruction 2** : Use concise, readable expressions, and rely on built-in Python idioms. Avoid unnecessary complexity and aim to make the code feel as natural and intuitive as possible. 

