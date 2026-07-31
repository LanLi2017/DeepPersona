# Evolutionary System Prompt Learning for Reinforcement Learning in LLMs

**Lunjun Zhang**[1] **Ryan Chen**[2 3] **Bradly C. Stadie**[2 3] 

## Abstract

Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose **Evolutionary System Prompt Learning (E-SPL)** , a method for jointly improving model contexts and model weights. In each RL iteration, E-SPL samples trajectories under multiple system prompts in parallel, then jointly applies RL updates to LLM weights and evolutionary updates to system prompts. System prompts evolve via mutation and crossover, two genetic operators driven by LLM self-reflection; selection is based on relative performance ratings updated across RL iterations. E-SPL encourages a natural division between declarative knowledge encoded in prompts and procedural knowledge encoded in weights, resulting in improved performance across reasoning and agentic tasks. For instance, in an easy-to-hard (AIME _→_ BeyondAIME) generalization setting, E-SPL improves RL success rate from 38 _._ 8% _→_ 45 _._ 1% while also outperforming reflective prompt evolution (40 _._ 0%). Overall, our results demonstrate that RL and system prompt evolution are deeply synergistic, and combining the two yields consistent gains in sample efficiency and generalization. Code: github.com/LunjunZhang/E-SPL 

## 1. Introduction

Evolutionary algorithms (EA) and reinforcement learning (RL) have long been studied as complementary paradigms for optimizing intelligent behaviors. Evolutionary methods excel at population-level exploration over discrete, high- 

> 1Department of Computer Science, University of Toronto 2Department of Statistics and Data Science, Northwestern University[3] Bridgewater AIA Labs. Correspondence to: Lunjun Zhang _<_ lunjun@cs.toronto.edu _>_ . 

_Preprint. February 26, 2026._ 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** AIME → BeyondAIME Easy-to-Hard Generalization Agent Evolution Tree Eval on BeyondAIME Evolution + RL RL Only Prompt Evolution

_Figure 1._ **Evolving System Prompts helps RL generalize better** . 

level structures (Koza, 1994; Back¨ et al., 1997; Lehman & Stanley, 2011a), while reinforcement learning is well suited for fine-grained optimization of policy parameters through trial and error (Williams, 1992; Sutton et al., 1999; Silver et al., 2016). Historically, however, these two approaches have largely operated on different representational levels: evolution over programs (Langdon & Poli, 2013) and architectures (Stanley & Miikkulainen, 2002) on the one hand, and gradient-based learning over a fixed set of parameters (Schulman et al., 2015; Mnih et al., 2015) on the other. Bridging these representational levels in a unified and scalable way remains an open challenge, particularly in the context of large language models (LLMs) (Brown et al., 2020; Achiam et al., 2023), where both the model weights (Jaech et al., 2024) and the model contexts (Wei et al., 2022; Zhou et al., 2022) critically shape behavior. 

Recent work on prompt evolution and LLM agents demonstrates that prompt (or context) engineering is a powerful mechanism for improving test-time performance. Techniques such as self-reflection (Shinn et al., 2023; Madaan et al., 2023; Cai et al., 2025) and population-based search (Guo et al., 2023; Fernando et al., 2023; Romera-Paredes et al., 2024; Agrawal et al., 2025) show that prompts can encode substantial declarative knowledge and behavioral priors without modifying model weights. At the same time, RL has emerged as the predominant post-training approach to fundamentally improve reasoning capabilities (Jaech et al., 2024; Guo et al., 2025; Chen et al., 2025; Comanici et al., 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** RL sampling for each problem x Evolution updates system prompts Sample N rollouts Compute Value estimate Select s 1, …, s M per from LLM s i πθ rewards under each s i Selection { y 1, j } [N] j =1 [∼] [π] θ [( ⋅∣] [s] 1 [,] [ x] [)] RR (( xx …,, y y 1,21,1)) mean V 1 Mutation R ( x , y 1, N ) Crossover … … System PromptsPopulation of { yM , j } [N] j =1 [∼] [π] θ [( ⋅∣] [s] M [,] [ x] [)] RR (( xx …,, y yMM ,1,2)) mean VM Update TrueSkill Ratings {( s i , ( μi , σi ))} R ( x , yM , N ) RL updates 1 M N Policy Gradient ∇ θ J ( x ) = θ ← θ + α ∇ θ J weights θ NM [∑] i =1 [∑] j =1 [(] [R] [(] [x] [,] [ y][i] [,] [j] [) −] [V][i] [)∇] [θ] [log] [ π][θ] [(] [y][i] [,] [j] [ ∣] [s] [i] [,] [ x] [)]

_Figure 2._ **Evolutionary System Prompt Learning (E-SPL)** jointly optimizes model contexts and model weights to enhance LLM self-improvement. Evolution updates system prompts; RL updates weights. The learned system prompts can encode _declarative knowledge_ via articulated principles and strategies, while RL gradients can further hone the model’s _procedural knowledge_ for reliable execution. 

2025). Despite their complementary strengths, existing approaches typically optimize model contexts and weights in isolation, leaving their potential synergies unexplored. 

In this work, we argue that effective self-improving agents should jointly evolve their high-level behavioral specifications and their low-level parametric policies. This perspective motivates **Evolutionary System Prompt Learning (E-SPL)** , a framework that tightly couples reinforcement learning over model weights with evolutionary search over system prompts. By maintaining a population of system prompts that are evaluated, rated, and evolved alongside RL training, E-SPL enables structured exploration over declarative knowledge while encoding procedural knowledge into weights. This joint optimization yields a synergistic dynamics in which improved prompts accelerate learning, and improved weights amplify the benefits of prompt evolution. 

Experimentally, E-SPL enhances RL performance of from 56 _._ 3% _→_ 60 _._ 6% on AIME 2025, 50 _._ 0% _→_ 52 _._ 7% on HMMT 2025 November test, and notably, 38 _._ 8% _→_ 45 _._ 1% on AIME _→_ BeyondAIME in an easy-to-hard generalization setting, using a base model of DeepSeek v3.1 (DeepSeek-AI, 2024). E-SPL significantly outperforms prompt evolution as well, showing that weight updates are often necessary for improving the model’s tacit knowledge and intuition. On an agentic search task (Jin et al., 2025), we improved RL performance from 44 _._ 2% _→_ 48 _._ 6% on OpenAI’s gpt-oss-120b (Agarwal et al., 2025) with limited data. Together, these results show that combining RL and system prompt evolution unlocks a synergistic form of selfimprovement that neither approach achieves in isolation. 

## 2. Related Work

**Prompt Evolution and Contextual Memory** A growing body of work treats prompts and contexts as optimizable objects that can encode high-level behaviors without modifying model weights. Evolutionary and search-based approaches such as EvoPrompt (Guo et al., 2023), PromptBreeder (Fernando et al., 2023), Automatic Prompt Engineer (Zhou et al., 2022), and FunSearch (Romera-Paredes et al., 2024) demonstrate that population-based search using operators like selection, mutation, and crossover can substantially improve performance at test time. Iterative context refinement methods, including Reflexion (Shinn et al., 2023), Self-Refine (Madaan et al., 2023), Trainingfree GRPO (Cai et al., 2025), Dynamic Cheatsheet (Suzgun et al., 2025), and ACE (Zhang et al., 2025b) further show that self-reflection can drive continued self-improvement. Agentic frameworks such as ReAct (Yao et al., 2022) and AutoGen (Wu et al., 2024) highlight that system prompts and interaction protocols play a central role in steering tool use and long-horizon behaviors. Besides, LLM-based memory mechanisms (Packer et al., 2023; Zhong et al., 2024; Lee et al., 2024; Xu et al., 2025) are also related to our work, since memory is key to helping the agent remember what strategies have worked in the past. A system prompt is often not just a foundational set of human-engineered instructions, but an agentic memory of desired behaviors and reward preferences (Ouyang et al., 2025). However, these approaches operate with frozen model parameters, limiting their ability to improve implicit knowledge acquired through experience. 

**Scaling RL for LLMs** Reinforcement learning has become the primary mechanism for further improving a pre- 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** HMMT 23/34 -> 25 AIME -> BeyondAIME Mutation + Crossover Mutation Only Mutation + Crossover Mutation Only Mutation Crossover scores

_Figure 3._ **Evolutionary trees of E-SPL** . During RL, E-SPL creates an evolutionary tree of system prompts, by re-suing the same data already generated by RL with minimal additional computational overhead. Each genetic operator (mutation or crossover) only requires a sampling server for LLM self-reflection with different context construction strategies, which can be concurrent with RL gradient updates. 

#### Discovered System Prompts for Solving Math Problems (deepseek v3.1)

**[1]** For generating functions with a numerator (1 _− x[M]_ ) _[k]_ and a denominator as a product of geometric series, the coefficient of _x[N]_ is a sum over _j_ of binomial coefficients times the number of solutions to a Diophantine equation. If _N < M_ , only the _j_ = 0 term contributes. For _N ≥ M_ , multiple _j_ may contribute, and the signs from the binomial coefficients must be considered. 

**[2]** After computations, **verify results for consistency across domains: check denominators, symmetry, and physical plausibility. Use alternative methods to recompute critical quantities and detect patterns such as periodicity.** 

**[3]** Avoid detours and unjustified guesses, including contest heuristics without validation. **Trust sound derivations; if inconsistencies persist, explore alternative configurations or identities.** 

**[4]** To verify modular injectivity, factor differences and check non-vanishing of the cofactor. For complex cases, test small primes; derivative condition is sufficient but not necessary. 

**[5]** For sums over integer pairs with conditions like gcd( _a, b_ ) = 1, consider using Mobius inversion or changing the¨ order of summation to simplify the expression. 

**[6]** When free parameters appear but a fixed answer is required, **look for cancellations forced by constraints or boundary conditions. Verify that the conclusion is consistent with all given conditions.** 

_Figure 4._ **Discovered strategies in learned System Prompts for solving math problems.** Those explicit behavior specifications include: useful heuristics and tips for various categories of problems, self-verification strategies such as checking for consistency and plausibility, a list of common failure modes to avoid, etc. Note that RL is done under diverse system prompts, and does not overfit to any particular one. 

trained LLM’s reasoning and agentic capabilities. Early preference-based methods demonstrated the effectiveness of RL for alignment and instruction following (Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022a;b), while more recent work on RL has focused on improving reasoning and problem-solving, starting with o1 (Jaech et al., 2024), R1 (Shao et al., 2024; Guo et al., 2025), and various further improvements (Team et al., 2025; Yu et al., 2025; Chen et al., 2025; Zheng et al., 2025; Khatri et al., 2025) to policy gradients (Williams, 1992; Schulman et al., 2017) on LLMs. However, these methods typically assume a fixed system prompt or interface during training, treating contexts 

as exogenous rather than as a programmatic and optimizable object. While recent work (Ye et al., 2025) has attempted to discover new prompts within RL loops, those prompts are generated as additional synthetic training data, which are fundamentally different from _system prompts_ that encode declarative knowledge, specify desired behaviors, and articulate strategic principles. Our work is also inspired by recent observations on the interplay between system prompts and RL (Tan et al., 2025; Azarbal et al., 2025; MacDiarmid et al., 2025), which suggested that certain system-level contexts can meaningfully improve RL generalization by reducing optimization pressure to globally update the model. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Mutation RL Value Estimate under Agent Trajectories Self-reflect on Edit the previous each of s 1, …, s M under s k past mistakes best prompt V 1, …, VM ( x , yk ,1, R ( x , yk ,1)) Population of ( x , yk ,2, R ( x , yk ,2)) Self Reflection Propose Edits to System Prompt s mutate System Prompts{( s i , ( μi , σi ))} k = argmax iVi s k ( x , yk , N , R …( x , yk , N )) s k Add μ ′ = μ σ ′ = σ + Δ σ ( s mutate , ( μ ′, σ ′))

_Figure 5._ **Mutation operator** in E-SPL. The highest-performing prompt in each iteration undergoes **LLM self-reflection** on group-wise agent trajectories and their outcomes. An LLM-generated diff edits the parent into a child system prompt, removing ineffective rules and converting observed mistakes into improved declarative instructions, yielding a new prompt that enters the evolutionary population. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Crossover RL M Problems where each Analyze why each Edit the previous System b Problems s i was the top prompt s i did best on ρi best prompt Prompts x 1 x 2 x 3 x 4 Avg ( s i , ρi ) Population of s 1 0.9 0.2 0.5 0.6 0.55 ( s 1, ( x 1)) Self Reflection Propose Edits to System Prompts System Prompt s 2 0.1 0.3 0.8 0.4 0.40 ( s 2, ( x 3)) {( s i , ( μi , σi ))} s 3 0.2 0.6 0.7 0.9 0.60 ( s 3, ( x 2, x 4)) Overall best s k s crossover Score Matrix μ ′ = (∑ i σμi [2] i [)(] [∑] [i] σ 1 i [2] [)] σ ′ = 1/( ∑ i 1/ σi [2] ) [+ Δ] [σ] [2] ( s crossover , ( μ Add′, σ ′))

_Figure 6._ **Crossover operator** in E-SPL. System prompts are compared based on their problem-wise performance within the current RL batch. Guided by these differential strengths and weaknesses, an **LLM self-reflection** process selectively recombines the most effective complementary segments from multiple parent prompts, yielding a new child prompt that enters the evolutionary population. 

**Combining Evolutionary Search with RL** Hybrid approaches combining Evolution and RL have long been studied, including ERL (Khadka & Tumer, 2018), PBT (Jaderberg et al., 2017), and CEM-RL (Pourchot & Sigaud, 2018), which all leverage population dynamics to complement gradient-based optimization. Open-ended evolutionary methods such as Novelty Search (Lehman & Stanley, 2011b), MAP-Elites (Mouret & Clune, 2015), and POET (Wang et al., 2019) emphasize exploration over diverse environments and behaviors, but do not operate at the level of prompts and contexts. More recent LLM-guided systems revive program-level evolution, with FunSearch (RomeraParedes et al., 2024) and AlphaEvolve (Novikov et al., 2025) demonstrating that evolutionary search over code and algorithms can yield novel discoveries. The Darwin Godel¨ Machine (DGM) (Zhang et al., 2025a) draws inspiration from Godel machine (¨ Schmidhuber, 2003) to include self-referential self-modification in agentic prompting, but does not include model weight updates during the selfmodification process. In contrast, E-SPL jointly optimizes system prompts and model weights in a single training loop. 

**Genetic Programming (GP)** is a learning paradigm where candidate programs are iteratively refined through mu- 

tation and crossover operators (Koza, 1994; Banzhaf et al., 1998; Langdon & Poli, 2013) to improve performance. Modern GP approaches have demonstrated strong results across domains such as symbolic regression (Ma et al., 2022), complex scheduling (Nguyen et al., 2017), and algorithmic discovery (Chen et al., 2023). A large language model under a unique system prompt and a particular set of tools can be seen as a program (Lehman et al., 2023; Shojaee et al., 2024; Romera-Paredes et al., 2024; Opsahl-Ong et al., 2024), and the LLM can be used to flexibly design the mutation and crossover operators that continuously generate new members of the population (Hemberg et al., 2024; Agrawal et al., 2025; Novikov et al., 2025). While prior work (Surina et al., 2025) has also considered combining LLM-based genetic operators (Romera-Paredes et al., 2024) with DPO (Rafailov et al., 2023), their focus was primarily on combinatorial optimization tasks, where LLM generates specific pieces of code for problems like Traveling Salesman Problem (TSP) (Lin, 1965). By contrast, we tackle the much more general setting of LLM post-training, where the LLM needs to generate system prompts that guide the next iteration of itself for general-purpose problem solving. Compared to AI Scientist approaches (Gottweis et al., 2025), we do not use additional agentic scaffold on top of RL. 

_Figure 7._ **An example of the system prompt discovered by E-SPL for RL with search engines. Here, the discovered system prompt specifies a workflow that includes: generative self-verification, detailed instructions on tool use, critical formatting rules, heuristics for determining the reliability of various information sources, actionable guidance for avoiding common failure modes, etc.** 

System Prompt for RL Agent with Search Engine (openai gpt-oss-120b) 

#### System Prompt _Before_ E-SPL

You are an expert assistant who solves tasks using a Wikipedia search tool. Here is an example of solving a real question: 

“Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?” _· · ·_ Calling search tool: <tool ~~c~~ all> _{_ "name": "search", "arguments": _{_ "query ~~l~~ ist": ["Population New York City between 2020 and 2025"] _}}_ </tool ~~c~~ all> 

#### System Prompt _After_ E-SPL

You are an expert research assistant. _· · ·_ **Workflow (follow** _**exactly**_ **in this order)** _**· · ·**_ 

**3. Call the search tool once** — Use **exactly** this syntax 

**4. Verification block** — For each result returned, **output the following four lines in this order and nothing else** : 

Source: <domain> Summary: <one-sentence summary> Status: VERIFIED (or NOT VERIFIED) Phrase: "<exact phrase>" (only if Status is VERIFIED) 

#### A result is VERIFIED only if:

- the exact target phrase appears verbatim in the snippet, **and** 

- the domain ends with .org, .edu, .gov, is Wikipedia, **or** a major reputable outlet (e.g., nytimes.com, bbc.com, reuters.com, theguardian.com) 

If the domain is not on this list, mark the result **NOT VERIFIED** even if the phrase is present. 

**5. Refine query (optional)** — **If all results are NOT VERIFIED, modify the query and repeat Step 3.** 

#### Critical formatting rules (must be obeyed)

- All tool calls must be wrapped exactly as <tool ~~c~~ all> _{_ ... _}_ </tool ~~c~~ all> 

- The only user-visible output after verification is the single Answer: line; no bullet points, markdown, blank lines, or extra text 

#### Positive example (correct flow)

User: "What other name is used for the 4 × 400 m relay?" Query: "4 x 400 metres relay" <tool ~~c~~ all> _{_ "name": "search", "arguments": _{_ "query ~~l~~ ist": ["4 x 400 metres relay"] _}}_ </tool ~~c~~ all> Source: en.wikipedia.org Summary: The 4 x 400 metres relay is also known as the "long relay". Status: VERIFIED Phrase: "long relay" _· · ·_ 

**Follow exactly this loop for every user query. Your objective is to prove the answer with a verbatim phrase from a reliable source before you** _**everr**_ **write the final Answer: line.** 

## 3. Method

In this section, we describe how to combine reinforcement learning (RL) on LLM weights with an evolutionary algorithm (EA) on LLM system prompts. In a nutshell, we condition RL weight updates on a population of system prompts, and reuse the resulting RL data to jointly evolve this population via two genetic operators, mutation and crossover, both of which are based on LLM self-reflection. **We design the evolutionary algorithm to be directly compatible with RL post-training pipelines, allowing RL to benefit from an evolving repertoire of declarative knowledge** while incurring minimal additional overhead. 

### 3.1. Reinforcement Learning for Weight Update

The RL component of E-SPL optimizes the LLM policy _πθ_ to maximize rewards on dataset _D_ under a variety of system prompts. Each system prompt, denoted as **s** , is a list of strategic instructions that contain declarative knowledge on how to solve various categories of problems. The system prompts are not given _a priori_ , but are instead discovered by the LLM through self-reflection on its own experience during the RL training process. Each RL iteration uses _M_ system prompts _{_ **s** _i}[M] i_ =1[in][parallel,][so][that][the][model] weights do not overfit to any particular system prompt; this also induces direct evolutionary competition among system prompts. For each problem _x ∼D_ in the batch, and for each system prompt **s** _i_ , we sample _N_ trajectories 

$$y_{i,j} \sim \pi_{\theta}(\cdot \mid \mathbf{s}_{i}, x) \quad \text{for } j=1,\dots,N \tag{1}$$

Then, we compute the outcome reward _ri,j_ for each trajectory and the value function estimate _Vi_ accordingly: 

$$r_{i,j} = R(x,y_{i,j}) \qquad V_{i} = \tfrac{1}{N} \sum_{j=1}^{N} r_{i,j} \tag{2}$$

The policy gradient objective for each problem _x_ is: 

$$\mathcal{J}(\theta, x) = \tfrac{1}{NM} \sum_{i=1}^{M} \sum_{j=1}^{N} ( r_{i,j} - V_{i} ) \log \pi_{\theta}( y_{i,j}\mid \mathbf{s}_{i}, x)$$

We perform the policy gradient update to LLM weights _θ_ on a batch of problems _B ∼D_ , while regularizing the policy _πθ_ to be close to the reference policy _π_ ref: 

$$\theta \leftarrow \theta + \alpha \, \tfrac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \nabla_{\theta} \big(\mathcal{J}(\theta, x) - \beta \, \mathrm{KL}( \pi_{\theta}, \pi_{\text{ref}}) \big) \tag{3}$$

When using low-rank adaptation (Hu et al., 2022), the KL regularization is implicit (since _πθ_ cannot deviate too much from _π_ ref by construction) and thus often ignored ( _β_ = 0). 

Performing policy gradients conditioned on learned system prompts also encourages a separation between two sources of knowledge: _declarative knowledge_ that can be explicitly articulated and stored as principles in the system prompt, versus _procedural knowledge_ that is tacit, intuitive, and only editable in weights. Conditioning the model on declarative knowledge in prompts reduces the optimization pressure to globally update the model in weights, and allows the weight updates to focus on acquiring procedural knowledge. 

### 3.2. Evolutionary Algorithm for Context Update

Our evolutionary algorithm ( **EA** ) optimizes the LLM system prompt to best elicit model capabilities and encode explicit knowledge learned from experience. Like in classical EA, we keep a population of candidates, and: (1) **evaluate the fitness** of different system prompts continuously during training; (2) **select** the successful ones from the population; (3) **reproduce** new system prompts from selected parents to add to the population via mutation and crossover – both grounded in LLM self-reflection but differing in their context construction strategies. The population _S_ starts with a single root, and gradually grows into an evolutionary tree. 

**Fitness Evaluation** Our EA estimates the competence of each system prompt using the same trajectories already generated by RL. To ensure seamless integration into existing RL pipelines, we eliminate the need for a separate fitness evaluation on a held-out validation set that is typical of prompt evolution algorithms. Instead, evolution directly leverages the noisy train-time batch statistics generated during RL rollouts, incurring negligible additional cost. 

Reusing on-policy batch statistics comes with an important consequence: absolute returns are not directly comparable across training iterations, as system prompts are evaluated on different data batches and under different model weights _θ_ . However, within each RL iteration, all sampled system prompts are evaluated under matched conditions – same batch, same weights. We therefore treat each RL iteration as a **tournament** round among selected system prompts, where only their relative ordering is meaningful across time. 

To convert the relative ordering into persistent, comparable ratings, we use the Bayesian skill rating system TrueSkill (Herbrich et al., 2006), which models each rating as a Gaussian distribution _N_ ( _µ, σ_[2] ). Concretely, given value estimates [ _V_ 1 _, . . . , VM_ ] for the sampled prompts [ **s** 1 _, . . . ,_ **s** _M_ ] and their current ratings [( _µ_ 1 _, σ_ 1) _, . . . ,_ ( _µM , σM_ )], we compute the sorted ordering (in descending order) for _{_ **s** _i}[M] i_ =1[:] 

$$\mathbf{v} \leftarrow \mathrm{argsort}([V_{1},\dots, V_{M}])$$

and update their TrueSkill ratings associated with each sampled prompt **s** _i_ in the population _S_ accordingly (see §B): 

$$\{(\mu_{i}, \sigma_{i})\}_{i=1}^{M} \leftarrow \mathrm{TrueSkill.update}(\{(\mu_{i}, \sigma_{i})\}_{i=1}^{M}, \mathbf{v})$$

which performs approximate message passing between those ratings. This procedure yields a persistent estimate of each prompt’s competence that can aggregate evidence across iterations despite changing model weights. 

**Selection** We select the system prompts used in each RL iteration based on their TrueSkill scores. We use an optimistic estimate of each system prompt **s** _i_ ’s skill level, which forms our sampling distribution for selection: 

$$\mathrm{score}_{i} = \mu_{i}+\lambda \sigma_{i} \qquad p_{i} \propto \exp ( \mathrm{score}_{i} /T ) \tag{4}$$

**Algorithm 1 — Evolutionary System Prompt Learning for RL**

```text
Input: Problem set D(·), reward function R(·,·).
Input: RL policy π_θ(·); reference policy π_ref(·).
Input: RL hyper-parameters: N, α, β.
Input: Initial system prompt s_root; M, p_crossover.
Input: Initial (μ_0, σ_0) for TrueSkill; Δσ; λ.
Initialize S = {(s_root, μ_0, σ_0)}.                          # Population to Evolve
while not converged do
    Sample a problem x ~ D
    Sample M system prompts {(s_i, μ_i, σ_i)}_{i=1}^M ~ S,
        each with probability  p_i ∝ exp(μ_i + λ σ_i)
    for i = 1 to M do
        Sample N solutions {y_{i,j}}_{j=1}^N :  y_{i,j} ~ π_θ(· | s_i, x)
        Reward r_{i,j} = R(x, y_{i,j}),  value V_i = (1/N) Σ_{j=1}^N r_{i,j}
    end for
    J(θ) = (1/NM) Σ_i Σ_j (r_{i,j} − V_i) log π_θ(y_{i,j} | s_i, x)
    θ ← θ + α ∇_θ ( J(θ) − β KL(π_θ, π_ref) )                 # RL update
    v ← argsort([V_1, …, V_M])                                # rank the s_i
    {(μ_i, σ_i)}_{i=1}^M ← TrueSkill.update({(μ_i, σ_i)}_{i=1}^M, v)   # update ratings in S
    k ← argmax_i V_i                                          # only mutate the best prompt s_k
    Create mutation self-reflection prompt Ψ_reflect using [(x, y_{k,1}, r_{k,1}), …, (x, y_{k,N}, r_{k,N})].
    ℓ ~ π_ref(· | Ψ_reflect)
    diff ~ π_ref(· | s_k, ℓ)
    s_mutate ← git.apply(s_k, diff)                           # Mutation step
    S.append((s_mutate, μ_k, σ_k + Δσ))
    if p_crossover > random.rand() then
        Let B = {x_b}_{b=1}^B be a batch of B past problems
        Let Ω = {(s_i, μ_i, σ_i)}_{i=1}^M
        (s', μ', σ') = Crossover(B, R, Ω, π_θ, π_ref)         # Alg 2
        S.append((s', μ', σ'))
    end if
end while
```
 

where _λ >_ 0 controls the degree of optimism in estimating skill levels, and _T_ is a temperature term; both higher _λ_ and higher _T_ can encourage exploration. In addition, we apply a sliding window of size _K_ , similar to a replay buffer, so that we only sample from the latest _K_ system prompts. This strategy favors system prompts that perform well while maintaining a high degree of exploration. 

**Mutation** After evaluating the empirical performance of selected system prompts _{_ **s** _i}[M] i_ =1[, we identify the best:] 

$$k \leftarrow \mathrm{argmax}_{i} V_{i} \tag{5}$$

and only apply mutation on this best candidate **s** _k_ . This directly adds additional evolutionary pressure on the population: in each iteration, **only the one system prompt that outperforms all of its competitively selected peers** 

**Algorithm 2 — Crossover(B, R, Ω, π_θ, π_ref)**

```text
Input: List of B problems: B; reward function R(·,·).
Input: List of M system prompts: Ω.
Input: RL policy π_θ(·), reference policy π_ref(·).
Let Φ ∈ ℝ^{M×B}        # Φ_{i,b}: empirical return of i-th prompt in Ω on b-th problem in B
for (s_i, μ_i, σ_i) in Ω do
    for x_b in B do
        Sample {y_{i,b,j}}_{j=1}^N ~ π_θ(· | s_i, x_b)        # reuse RL rollouts from Alg 1
        Φ_{i,b} = (1/N) Σ_{j=1}^N R(x, y_{i,b,j})
    end for
end for
k ← argmax_i Σ_b Φ_{i,b}                                      # s_k is the overall top prompt
φ = [ ]                                                       # problems each s_i was the top prompt for
for (s_i, μ_i, σ_i) in Ω do
    ρ_i = [ ]
    for x_b in B do
        if i = argmax_z Φ_{z,b} then
            ρ_i.append(x_b)                                   # s_i is the top prompt for x_b
        end if
    end for
    φ.append((s_i, ρ_i))                                      # analyze why s_i did best on ρ_i
end for
Create crossover self-reflection prompt Ψ_cr using φ.
diff_cr ~ π_ref(· | s_k, Ψ_cr)
s_crossover ← git.apply(s_k, diff_cr)                         # Crossover step
μ' = (Σ_i μ_i/σ_i²) / (Σ_i 1/σ_i²),   σ' = sqrt( 1/(Σ_i 1/σ_i²) + Δσ² )
Return (s_crossover, μ', σ')
```

**gets to produce a child of its own through mutation** . To run mutation on **s** _k_ , we first ask the reference policy _π_ ref to summarize each trajectory and self-reflect (using mutation self-reflection prompt Ψreflect) on the mistakes that _πθ_ made in those trajectories generated under the system prompt **s** _k_ : 

$$\ell \sim \pi_{\text{ref}}(\cdot \mid \Psi_{\text{reflect}}, (x, y_{k,1}, r_{k,1}), \dots, (x, y_{k,N}, r_{k,N}))$$

where the model reasons across all the rollouts _{yk,j}[N] j_ =1 and their outcomes _{rk,j}[N] j_ =1[to] **[identify][past][mistakes] and potential strategies to avoid them going forward** . Next, we use the lessons _ℓ_ from self-reflection to propose edits on the system prompt **s** _k_ in a format similar to git diff: 

$$\mathrm{diff} \sim \pi_{\text{ref}}(\cdot \mid \mathbf{s}_{k}, \ell) \qquad \mathbf{s}_{\text{mutate}} \leftarrow \mathrm{git.apply}(\mathbf{s}_{k}, \mathrm{diff})$$

Finally, we add **s** mutate to the population _S_ with TrueSkill rating ( _µk, σk_ + ∆ _σ_ ); under mutation, a child inherits the parent’s rating but with increased uncertainty. This favors the child in the next round of selection according to Eq (4). 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** HMMT 23/34 -> 25 AIME -> BeyondAIME Learned TrueSkill Test Success Rates Learned TrueSkill Test Success Rates Best Best Best Best Mutation Crossover scores

_Figure 8._ **TrueSkill scores learned from train-time rankings are predictive of relative test-time performance of system prompts.** E-SPL treats each RL iteration as a set of relative comparisons between system prompts for TrueSkill message passing. The visualization shows that E-SPL effectively aggregates noisy train-time comparisons into a stable and informative criterion for evolutionary selection. 

**Crossover** enables selective recombination of knowledge from multiple parents in producing an offspring. In contrast, mutation alone isolates evolutionary branches, requiring each lineage to independently rediscover similar declarative knowledge. Crossover therefore facilitates rapid dissemination of effective strategies across the population. 

Our crossover operator first computes _Vi_ in Eq (2) for each problem in a batch _B_ , yielding a score matrix Φ _∈_ R _[M][×][B]_ where the rows represent _M_ sampled system prompts _{_ **s** _i}[M] i_ =1[and the columns represent] _[ B]_[problems in the batch] _{xb}[B] b_ =1[.][Each entry][ Φ] _[i,b]_[ represents the average reward of] problem _xb_ obtained under **s** _i_ during this RL iteration. 

The matrix Φ explicitly captures the problem-wise strengths of each system prompt by identifying which system prompts outperform others on specific problems. Incorporated into the crossover self-reflection prompt Ψcr, this information grounds LLM self-reflection in concrete examples of differential advantages and enables targeted recombination of complementary prompt segments. **The goal of the crossover operator is to produce children that inherit and recombine the strongest traits of multiple parents.** In Algorithm 2, after crossover self-reflection, we apply the proposed git diff edits to **s** _k_ to obtain **s** crossover, which is added to the population _S_ with a TrueSkill rating initialized as a precision-weighted average of the parents’ ratings. 

**Self-reflection is key to genetic operators** Both mutation and crossover are LLM-based self-reflection processes that diagnose failures and synthesize structured edits to produce children for the population; they differ only in how the selfreflection context is constructed. Mutation reflects on agent trajectories from a single best-performing system prompt to repair its weaknesses, while crossover reflects on comparisons across multiple system prompts in their problem-wise performance to recombine complementary segments. 

### 3.3. Algorithmic Summary

Evolutionary System Prompt Learning ( **E-SPL** ) augments RL post-training with an evolutionary algorithm based on self-reflection from experience. In parallel with RL updates, E-SPL uses the relative ordering of each system prompt’s average reward to obtain TrueSkill ratings, which provide numerical scores that inform evolutionary selection. To create new system prompts for the population, E-SPL applies mutation and crossover operators based on LLM self-reflection. As a result, E-SPL brings the benefits of evolution to RL with minimal additional computational overhead. 

## 4. Experiments

We evaluate E-SPL on mathematical reasoning and agentic search benchmarks, with the goal of answering the following questions: **(1)** Does jointly evolving system prompts and model weights outperform optimizing either alone? **(2)** How does E-SPL affect the learning speed and generalization of RL? **(3)** What are some emergent patterns in the system prompts discovered via E-SPL? **(4)** What roles do mutation and crossover play in the evolutionary dynamics of E-SPL? 

### 4.1. Experimental Setup

**Tasks and datasets.** We first focus on mathematical reasoning tasks with limited data, a regime in which inductive biases from evolved system prompts are expected to play a particularly important role. Our main evaluations include: **(1) DAPO100** _→_ **AIME25** , which follows the protocols from (Cai et al., 2025) that trains on 100 selected problems from the DAPO dataset (Yu et al., 2025) and tests on AIME 2025 competition; **(2) HMMT 23/24** _→_ **25** , which uses all of HMMT 2023, 2024, and February 2025 problems as training set, and November 2025 problems as test set; **(3) AIME** _→_ **BeyondAIME** , which uses past years of 

_Figure 9._ **An example of how the RL model learns to utilize learned system prompts** (numbered G1 _. . ._ G30) **to solve problems.** 

Example of How E-SPL Shapes Model Behavior 

#### Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below: 

**[G1]. When optimizing sums, set parameters to cancel dominant terms. For quadratic forms, choose parameters that minimize the leading coefficients, reducing the problem to analyzing remainder terms.** 

**[G22]. In modular arithmetic and sums, decompose terms via remainders or moduli. Use periodicity for remainder sums and CRT for coprime factors, ensuring multiplicative solutions for square-free moduli.** 

#### Problem

Let Z _/n_ Z be the set of all integers modulo _n_ (hence Z _/n_ Z contains _n_ elements). Find the second smallest composite number _n_ , such that there exists a bijective function _g_ : Z _/n_ Z _→_ Z _/n_ Z such that the 101 functions 

$$g(x),\ g(x)+x,\ g(x)+2x,\ \cdots,\ g(x)+100x$$

are all bijective from Z _/n_ Z to Z _/n_ Z. 

#### Model Output

**Understanding the Condition:** Each function _hk_ ( _x_ ) = _g_ ( _x_ ) + _kx_ (for _k_ = 0 _,_ 1 _, . . . ,_ 100) must be a permutation of Z _/n_ Z. Since _g_ is bijective, _h_ 0( _x_ ) = _g_ ( _x_ ) is bijective by assumption. For _k ≥_ 1, _hk_ is bijective if and only if for every _a ∈_ Z _/n_ Z, the equation _g_ ( _x_ ) + _kx_ = _a_ has a unique solution _x_ . 

**Alternatively** , since _g_ is a permutation, we can set _y_ = _g_ ( _x_ ) and then _x_ = _g[−]_[1] ( _y_ ), so _hk_ ( _g[−]_[1] ( _y_ )) = _y_ + _kg[−]_[1] ( _y_ ) _._ But perhaps a better approach is to consider the differences. **Note:** _hk_ is bijective if and only if the map _x ↦ g_ ( _x_ )+ _kx_ is injective. That is, if _g_ ( _x_ )+ _kx_ = _g_ ( _y_ )+ _ky_ then _x_ = _y_ . Equivalently, _g_ ( _x_ ) _− g_ ( _y_ ) = _−k_ ( _x − y_ ) _._ Since _g_ is arbitrary (but fixed for all _k_ ), we need to choose _g_ such that for each _k_ = 1 _, . . . ,_ 100, the equation _g_ ( _x_ ) _− g_ ( _y_ ) = _−k_ ( _x − y_ ) implies _x_ = _y_ . 

**Alternatively** , we can think about the differences: for each nonzero _d ∈_ Z _/n_ Z, we require that _g_ ( _x_ + _d_ ) _− g_ ( _x_ ) = _−kd_ for all _x_ and for all _k_ = 0 _, . . . ,_ 100. Actually, injectivity of _hk_ is equivalent to: if _x_ = _y_ , then _g_ ( _x_ ) + _kx ̸_ = _g_ ( _y_ ) + _ky,_ i.e., _g_ ( _x_ ) _− g_ ( _y_ ) _̸_ = _−k_ ( _x − y_ ). So for each nonzero _d_ , and for each _k_ , we require that _g_ ( _x_ + _d_ ) _− g_ ( _x_ ) _̸_ = _−kd_ for all _x_ . **But** since _g_ is a permutation, this is not easy to control. **Another idea** : Consider the composition _hk ◦ g[−]_[1] . Let _f_ = _g[−]_[1] . Then _hk_ ( _g[−]_[1] ( _y_ )) = _y_ + _kf_ ( _y_ ) _· · · . . ._ 

**But let’s think** : For _n_ to work, it must be that all prime factors are at least 103? But that gives large numbers. Maybe there is a trick: The number 101 is prime. So perhaps _n_ must be coprime to 101! or something. **Another idea** : Consider the function for _k_ = 101? Not required. 

**Given the instructions, we should use [G22]: In modular arithmetic, use CRT.** 

If _n_ = product of primes, then by CRT, the condition holds if and only if it holds for each prime power factor. So for each _p[a]_ dividing _n_ , there exists a bijection _gp_ on _Z/p[a]_ such that the 101 functions are bijective. For a prime _p >_ 101, take _g_ ( _x_ ) = _x_ works. For _p <_ = 101, we need to see if there is a bijection on _Z/p[a]_ . So the smallest composite would be one that has no prime factor _<_ = 101, i.e., all factors _>_ = 103. So the smallest is 103[2] = 10609. But then the second smallest would be 103 _×_ 107 = 11021. _. . ._ 

**Final Answer:** 

11021 

**Ground truth answer:** 11021 

_Figure 10._ **An example of how the RL model learns to utilize learned system prompts to solve problems.** 

Example of How E-SPL Shapes Model Behavior 

#### Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below: 

**[G26]. After deriving results, perform sanity checks: test small instances, ensure bounds, and verify extreme cases. Re-examine complex steps if results seem off.** 

Problem 

Real numbers _x_ 1 _, x_ 2 _, · · · , xn_ satisfy _|x_ 1 _|_ + _|x_ 2 _|_ + _· · ·_ + _|x_ 3526 _|_ = 1. Let _M_ be the maximal possible value of 

1 

_|x_ 1 _|_ + _|x_ 1 _− x_ 2 _|_ + _|x_ 1 + _x_ 2 _− x_ 3 _|_ + _· · ·_ + _|x_ 1 + _x_ 2 + _· · ·_ + _x_ 3525 _− x_ 3526 _|_ 

Find _M_ mod 1000. 

Model Output 

#### Let’s analyze the problem step by step:

Let _Sk_ = _x_ 1 + _x_ 2 + _· · ·_ + _xk_ for _k_ = 1 _,_ 2 _, . . . ,_ 3526, with _S_ 0 = 0. Then the denominator becomes: _|S_ 1 _|_ + _|S_ 1 _− S_ 2 _|_ + _|S_ 2 _− S_ 3 _|_ + _· · ·_ + _|S_ 3525 _− S_ 3526 _|._ But note: _S_ 1 = _x_ 1, _S_ 1 _− S_ 2 = _−x_ 2, _S_ 2 _− S_ 3 = _−x_ 3, _· · ·_ , _S_ 3525 _− S_ 3526 = _−x_ 3526. So the denominator is: _|S_ 1 _|_ + _|x_ 2 _|_ + _|x_ 3 _|_ + _· · ·_ + _|x_ 3526 _|._ However, this is not exactly matching the given expression. Actually, the given terms are: _. . . . . ._ 

The correct maximum is actually 2, achieved by the first construction with all terms zero except the last. Therefore, _M_ = 2. Then _M_ mod 1000 = 2. 

**So the answer is 2.** This is the simplest. 

**Given the instructions [G26]: perform sanity checks.** 

For _n_ = 2: variables _x_ 1, _x_ 2 with _|x_ 1 _|_ + _|x_ 2 _|_ = 1. _D_ = 1 _/_ ( _|x_ 1 _|_ + _|x_ 1 _− x_ 2 _|_ ). _· · ·_ So for _n_ = 2, maximum is 2. For _n_ = 3: maximize _D_ = 1 _/_ ( _|x_ 1 _|_ + _|x_ 1 _− x_ 2 _|_ + _|x_ 1 + _x_ 2 _− x_ 3 _|_ ). _· · ·_ So for _n_ = 3, we can get _D_ = 4. In general, for _n_ , the construction is: _x_ 1 = _a_ , _x_ 2 = _a_ , _x_ 3 = 2 _a_ , _x_ 4 = 4 _a_ , ..., _xn_ = 2 _[n][−]_[2] _a_ . Then _|x_ 1 _|_ = _|a|_ , _|x_ 1 _− x_ 2 _|_ = 0, _|x_ 1 + _x_ 2 _− x_ 3 _|_ = 0, ..., _|x_ 1 + _..._ + _xn−_ 1 _− xn|_ = _|_ (2 _[n][−]_[1] _−_ 1) _a −_ 2 _[n][−]_[2] _a|_ = _|_ (2 _[n][−]_[2] _−_ 1) _a|_ . **Wait, careful** : _Sn−_ 1 = _a_ + _a_ + 2 _a_ + 4 _a_ + _..._ + 2 _[n][−]_[3] _a_ = (2 _[n][−]_[2] _−_ 1) _a_ + _a_ = 2 _[n][−]_[2] _a_ . Then _Sn−_ 1 _− xn_ = 2 _[n][−]_[2] _a −_ 2 _[n][−]_[2] _a_ = 0. So all terms after the first are 0 except the second which is 0. So denominator = _|a|_ . Constraint: _|a|_ + _|a|_ + _|_ 2 _a|_ + _|_ 4 _a|_ + _..._ + _|_ 2 _[n][−]_[2] _a|_ = _|a|_ (2 + (2 _[n][−]_[1] _−_ 2)) = _|a| ∗_ 2 _[n][−]_[1] = 1, so _|a|_ = 2[1] _[−][n]_ . Then _D_ = 2 _[n][−]_[1] . For _n_ = 3526, _D_ = 2[3525] . **This is huge, and integer. And it is better than the previous construction.** 

So the maximum is _M_ = 2[3525] . Then _M_ mod 1000 = 2[3525] mod 1000. Now, we need to compute 2[3525] mod 1000. _· · ·_ So 2[3525] _≡_ 432 (mod 1000). **So the answer is 432** . This is the intended solution. 

**Final Answer** : 

432 

**Ground truth answer:** 432 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** DAPO100 AIME 25 HMMT 23/24 25 AIME BeyondAIME 65 55 52.7 45.0 45.1 60 60.6 50 48.6 50.0 57.3 56.3 47.0 47.3 42.5 55 54.6 52.6 45 40.0 39.1 40.0 38.4 38.8 50 48.6 40 39.3 37.5 36.5 45 35 35.0 Base model Self-Reflection Evolution RL (CISPO) RL (GRPO) Evolution + RL Base modelSelf-ReflectionEvolutionRL (CISPO)RL (GRPO)Evolution + RL Base modelSelf-ReflectionEvolutionRL (CISPO)RL (GRPO)Evolution + RL Base modelSelf-ReflectionEvolutionRL (CISPO)RL (GRPO)Evolution + RL Success Rate %

_Figure 11._ Results on mathematical reasoning datasets using DeepSeek-v3.1. **Evolution + RL** (our E-SPL method) performs best compared to self-reflection (prompting only), evolution (prompting only), and RL (weight update only). The results show that coupling RL with evolution on system prompts unlocks a synergistic form of self-improvement that neither approach achieves in isolation. 

AIME (AIME 22, AIME 23, and AIME 24) as the training set, and BeyondAIME (ByteDance-Seed, 2025) as test set, which is a collection of original problems with difficulty levels greater than or equal to that of the hardest AIME problems each year. In addition, we also evaluate on **(4) Agentic Search** on Natural Questions (NQ) (Kwiatkowski et al., 2019) and HotpotQA (Yang et al., 2018), where we subsample 196 problems for training and 64 problems for test; the task is to teach an LLM model to use search engines to answer complex multi-hop questions (Jin et al., 2025). 

**Models** On math tasks, we use the base model DeepSeek v3.1 (DeepSeek-AI, 2024) for our experiments, which is a 671B MoE (Shazeer et al., 2017) model with 37B active parameters. On agentic search tasks, we use OpenAI’s gptoss-120b (Agarwal et al., 2025), which is a 120B MoE model with 5B active parameters. All finetuning is done via low-rank adaptation (Hu et al., 2022) with a rank of 32. 

**Methods compared** We compare the following variants: 

- **RL only** : standard RL post-training with a fixed system prompt; we benchmark the popular **GRPO** (Shao et al., 2024) and **CISPO** (Chen et al., 2025) algorithms; 

- **Self-reflection** : iterative prompt refinement via reflection (Shinn et al., 2023; Cai et al., 2025; Ouyang et al., 2025), without weight updates. This is essentially our mutation operator on its own, with frozen weights, no crossover operator, and a population size of 1. 

- **Evolution** : which runs our evolutionary algorithm (EA) for system prompt search with frozen model weights. Our evolutionary algorithm is very similar to GEPA (Agrawal et al., 2025) except that we do not continuously run evaluation on a separate feedback set, and instead only use train-time rollout statistics just like in standard RL. Like GEPA, our EA works well on its own, and can sometimes outperforms RL. 

- **Evolution + RL (E-SPL)** : our full method, jointly evolving model contexts and updating model weights. 

### 4.2. Main Results

Figure 11 summarizes performance on math reasoning benchmarks using DeepSeek-v3.1 model. Figure 13 summarizes our results on agentic search tasks using OpenAI gptoss-120b. Across all datasets, **Evolution + RL** (E-SPL) consistently outperforms baselines that optimize either prompts (evolution-only) or weights (RL-only) in isolation. 

- On **AIME 2025** , E-SPL improves success rate from 56 _._ 3% _→_ 60 _._ 6% compared to the RL-only (GRPO). 

- On **HMMT 2025** , performance improves from 50 _._ 0% _→_ 52 _._ 7% after applying E-SPL to RL. 

- On an easy-to-hard generalization (Sun et al., 2024) setting, **AIME** _→_ **BeyondAIME** , the gains are the most pronounced, where E-SPL improves RL from 38 _._ 8% _→_ 45 _._ 1% in success rate. 

- On **agentic search** tasks, E-SPL improves RL from 44 _._ 2% _→_ 48 _._ 6%, which demonstrates that E-SPL is not limited to mathematical reasoning, but is also effective in agentic reasoning with tool use. 

- On all tasks above, E-SPL outperforms evolutiononly as well, with the relative improvements ranging from +5 _._ 8% to +12 _._ 8% on the mathematical reasoning tasks, and +42 _._ 9% on the agentic search task. 

Evolution-only consistently outperforms self-reflection, and can sometimes outperform RL-only. On both DAPO100 _→_ AIME25 and AIME _→_ BeyondAIME, evolution outperforms RL; on HMMT 23/24 _→_ 25, GRPO slightly outperforms evolution. This finding is consistent with prior findings in GEPA (Agrawal et al., 2025), and validates the design of our proposed evolutionary algorithm on its own. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Self Improvement Progress Comparison DAPO100 AIME 25 HMMT 23/24 25 AIME BeyondAIME 0.62 0.54 0.46 0.60 0.52 0.44 0.58 0.50 0.56 0.48 0.42 0.46 0.54 0.40 0.44 0.52 0.42 0.38 0.50 0.40 0.48 0.36 2 4 6 8 10 12 14 16 5 10 15 20 25 30 5 10 15 20 25 30 Batch Iterations Batch Iterations Batch Iterations Self-Reflection Evolution RL (CISPO) RL (GRPO) Evolution + RL Success Rate of Best Agent

_Figure 12._ **E-SPL learns faster, and achieves better asymptotic performance that neither RL nor prompt evolution alone could achieve.** By studying the learned system prompts (e.g. Figure 4 and Figure 9), we hypothesize that E-SPL achieves this by encouraging a natural division between declarative knowledge (factual information or global heuristics that can be verbally explained) in prompts and procedural knowledge (practical, instinctive “know-how” that relies on intuition developed through repetition) in weights. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Agentic Search gpt-oss-120b (NQ + HotpotQA) 0.486 0.45 0.442 0.34 0.30 0.15 0.066 0.00 Base model Evolution RL onlyEvolution + RL Success Rate

_Figure 13._ E-SPL outperforms both (prompt) Evolution and RL on agentic search tasks with openai gpt-oss-120b model. 

Figure 12 visualizes the learning curves across three mathematical reasoning settings. E-SPL demonstrates better sample efficiency and higher asymptotic performance compared to both RL-only and evolution-only baselines. Evolutiononly often learns the fastest early on since it only has to update model contexts, but obtains lower asymptotic performance because it does not update model weights and therefore struggles with improving procedural knowledge. 

In agentic search, we find that RL can significantly outperform prompt evolution (44 _._ 2% versus 34 _._ 0%); whereas in math reasoning, RL often performs on-par with evolution (Figure 11). It can often be hard to predict _a priori_ whether RL or prompt evolution will perform better on a particular domain, since the learning bottleneck could be an arbitrary mix of declarative and procedural knowledge. E-SPL naturally combines the strengths of both, allowing continual self-improvement to proceed along whichever axis of knowledge is most limiting for the task at hand. 

Overall, the results above suggest that evolving system prompts and applying RL to weights play complementary 

roles in LLM self-improvement. System prompts progressively accumulates explicit, declarative knowledge – such as high-level principles, robust workflows, corner cases, and failure modes (see Figure 4 and Figure 7) – while RL updates the weights to acquire procedural knowledge that is tacit and intuitive. By conditioning RL on increasingly informative system prompts, E-SPL allows weight updates to focus on intuitive skills, leading to better sample efficiency. 

### 4.3. Emergent Patterns in Learned System Prompts

We observe several recurring patterns that emerge in the learned system prompts, which we discuss below. 

**Generative Self-Verification** We find that a repeated theme in learned system prompts is about learning to selfverify a solution in a generative manner (Zhang et al., 2024). 

For instance, in the agentic search setting, the learned system prompt (Figure 7) defines a workflow that explicitly labels each search result as VERIFIED versus NOT VERIFIED based on two criteria: whether the target phrase appears verbatim in the snippet returned by search engine, and whether the domain name ends with .org, .edu, .gov, is a major reputable news outlet or Wikipedia. It goes on to specify that if all results have NOT VERIFIED status, then the model should repeat the search and continue the loop in its chain of thought. The learned system prompt defines the objective as _‘to prove the answer with a verbatim phrase from a reliable source before you everr write the final Answer: line’_ . **The RL process, in turn, does not need to discover new workflows from scratch and encode them in weights, but rather refines and perfects the execution of the workflows specified in the discovered system prompts.** Such workflows emerge through E-SPL and are absent from the initial system prompt. 

In solving math problems, a similar self-verification strat- 

**Table 1 — Mistakes in Learned E-SPL System Prompts.** The discovered system prompts sometimes include **heuristics that are not always true but still helpful** to problem-solving. Unlike RL weight updates, learned system prompts are interpretable and can be monitored and corrected. Two subtle mistakes in discovered principles, with their **corrected** versions (see §C.1 for explanations):

**Principle 1 (number theory).**
- *Discovered:* For a polynomial $P(n) = \prod_{i=1}^m (n - a_i)$, the largest integer $d$ that divides $P(n)$ for all integers $n$ is $d = \prod_{p} p^{e_p}$, where $e_p = \min_{r \bmod p}\bigl(\#\{\, i : a_i \equiv r \pmod p \,\}\bigr)$. Only primes $p \le \max(a_i)$ need be considered, as for larger $p$ the $a_i$ are distinct modulo $p$, so $e_p = 0$. For primes where $i \mapsto a_i \bmod p$ is a permutation (e.g., $a_i = i^3$ and $p \equiv 2 \bmod 3$), $e_p = \lfloor m/p \rfloor$.
- *Corrected:* …the largest integer $d$ is $d = \prod_{p} p^{e_p}$, where $e_p = \min_{n \in \mathbb{Z}} v_p\!\bigl(P(n)\bigr)$. Moreover, one always has the lower bound $e_p \ge \min_{r \bmod p}\bigl(\#\{\, i : a_i \equiv r \pmod p \,\}\bigr)$. For this mod-$p$ bound it suffices to consider primes $p \le m$, since if $p>m$ some residue class occurs zero times and the minimum is $0$. For the permutation case the bound gives $e_p \ge \lfloor m/p \rfloor$.

**Principle 2 (graph theory).**
- *Discovered:* For graphs with odd-degree vertices, find a minimum $T$-join by *pairing odd vertices along efficient paths*. In symmetric graphs such as grids, *pair boundary vertices* optimally to minimize total length.
- *Corrected:* For graphs with odd-degree vertices, find a minimum $T$-join by *pairing odd vertices via a minimum-weight perfect matching on shortest-path distances*. In symmetric graphs such as grids, *restrict the matching to symmetry-respecting pairings* to minimize total length.

_Table 1._ The discovered system prompts sometimes include **heuristics that are not always true but still helpful to problem-solving** . Unlike RL updates to model weights, the learned system prompts are interpretable and thus can be monitored and corrected. Here we showcase two subtle mistakes in two discovered principles, and what their **corrected** versions should be (see §C.1 for explanations). 

egy emerged. In Figure 10, the learned system prompt has discovered the following instruction: _‘After deriving results, perform sanity checks: test small instances, ensure bounds, and verify extreme cases. Re-examine complex steps if results seem off.’_ The full example in Figure 10 illustrates how the RL model has learned to use this particular self-verification strategy to solve a particular problem on BeyondAIME test set: after initially concluding that _‘So the answer is_ 2 _’_ , the model explicitly cites this strategy from the system prompt, and follows the exact instruction to perform sanity checks by starting from small instances of _n_ = 2 and _n_ = 3, before moving on to the general case and saying _‘Wait, careful’_ when it identified its own mistake. The model then self-corrects and concludes with _‘So the answer is 432. This is the intended solution’_ , which is correct. This example explains E-SPL’s effectiveness: high-level strategies are specified by the system prompt, while RL fine-tunes execution. Self-rewrite of the system prompt induces more systematic behavior changes than weight updates alone. 

**Codified Domain Expertise and Heuristics** The learned system prompts also have many strategies that are highly domain-specific. For instance, Figure 9 showcases learned system prompts such as _‘When optimizing sums, set parameters to cancel dominant terms. For quadratic forms, ...’_ (which is specific to problems that optimize sums) and _‘In modular arithmetic and sums, decompose terms via remainders or moduli. Use periodicity for remainder sums and CRT for coprime factors, ensuring multiplicative solutions for square-free moduli.’_ (which points out the importance of Chinese Remainder Theorem, or _CRT_ for short, in solving modular arithmetic). Example 9 shows that the RL model persistently explores many potential direc- 

tions, repeating phrases like _‘Alternatively’_ , _‘Another idea’_ , _‘But let’s think’_ , before simplifying the problem to a form of modular arithmetic and recognizing that it should apply _CRT_ in that situation according to the system prompt. Example 16 shows that the RL model cited a helpful heuristic in the system prompt suggesting the use of Mobius inver-¨ sion under specific conditions, which became a crucial step in the solution. Those examples demonstrate that **E-SPL can effectively codify declarative knowledge into the system prompt – specialized domain expertise and helpful problem-solving heuristics under various circumstances – so that RL can focus on honing core reasoning skills** rather than memorizing domain knowledge. 

**Failure Mode: Missing Caveats** Not all discovered strategies from E-SPL can hold true in the most general cases. In Table 1, we study two cases where the learned system prompt included heuristics that are not strictly true in general but are still helpful in limited scenarios. For instance, in analyzing polynomials of a particular form, the discovered strategy misunderstood the lower bound of a min as the actual min. In the other case, the discovered principle for graph theory vaguely outlined an algorithm for finding a minimum _T_ -join by _‘pairing odd vertices along efficient paths’_ ; the actual optimal solution requires _pairing odd vertices via a minimum-weight perfect matching on shortestpath distances_ . The likely reason for those mistakes is that E-SPL over-generalized some particular training data points or made mistakes in its self-reflection process. E-SPL is still missing caveats in many of its generalized principles. 

However, we argue that such mistakes in acquired knowledge are more transparent and interpretable than RL gradient updates, because they can be more easily monitored 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Effect of Using Crossover in E-SPL HMMT 23/24 25 AIME BeyondAIME (Same Difficulty) (Easy-to-Hard Generalization) 0.525 0.44 0.500 0.42 0.475 0.450 0.40 0.425 0.38 0.400 0.36 5 10 15 20 25 30 5 10 15 20 25 30 Batch Iterations Batch Iterations Mutation + Crossover + RL Mutation + RL RL Only (GRPO) Success Rate of Best Agent

_Figure 14._ **Crossover operator in evolution speeds up early learning** , but obtains mixed results in terms of final performance. On AIME _→_ BeyondAIME, Mutation + RL gets to 45 _._ 1% success rate, while Mutation + Crossover + RL gets to 42 _._ 5%; on HMMT, using Crossover operator results in slightly better performance (52 _._ 7% vs 51 _._ 6%). In either case, Evolution + RL outperforms RL-only. 

and edited by humans. A model trained by RL alone will also pick up on certain assumptions that are only partially correct, but even the best mechanistic interpretability tools today (Gao et al., 2024; Dunefsky et al., 2024; Kharlapenko et al., 2025) still cannot interpret those weight updates or edit the behaviors in a straightforward manner. By comparison, E-SPL represents a part of learned knowledge as explicit, propositional statements, which are much more suited for further verification, formalization (Trinh et al., 2024; Hubert et al., 2025), and scalable oversight (Leike et al., 2018; Bowman et al., 2022; McAleese et al., 2024). 

### 4.4. Ablation Studies

**Effect of Crossover** Figure 14 analyzes the contribution of genetic operators. Adding **crossover** to mutation accelerates early learning, consistent with its role in propagating high-fitness substructures across evolutionary branches. However, its effect on final performance is mixed: on AIME _→_ BeyondAIME, mutation-only achieves higher final accuracy, while on HMMT, mutation + crossover performs slightly better. **Both variants outperform RL-only** , confirming the value of evolutionary search over system prompts regardless of genetic operator choice. 

Figure 3 visualizes evolutionary trees produced during training. As shown, the crossover operator enables knowledge transfer across different branches of the evolutionary tree, but at the cost of reduced _speciation_ – long-term separation between evolutionary lineages. To better understand the tradeoffs of using crossover, we draw attention to prior work (Golberg, 1989; Eshelman, 1991; Mahfoud, 1995; Vose, 1999) that supports our observations: 

- crossover can accelerate short-term progress by mixing useful substructures, 

- but it can also wash out subpopulation structure, reduce long-run diversity, and increase the risk of premature homogenization and convergence. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Sampling Policy for Evolution (Reference vs Online RL Policy) 0.60 Evolution with RL Model 0.55 0.527 Evolution with Ref Model 0.510 0.50 0.45 0.425 0.416 0.40 0.35 HMMT BeyondAIME Success Rate

_Figure 15._ Using a fixed reference policy rather than the online RL policy for prompt evolution is important to performance. 

Mutation-only behaves more like independent local hillclimbs along separate lineages. Lineages stay separated longer, so we can keep parallel hypotheses alive (Holland, 1992). Mutation + Crossover accelerates early progress by re-combining partial improvements, but the population may rapidly converge around a locally high-performing system prompt, limiting exploration of qualitatively and structurally distinct high-quality solutions (Spears, 1993). Classically, this tradeoff can be significantly improved by a class of techniques called _niching_ , which include fitness sharing (Goldberg et al., 1987), mating restriction (Deb & Goldberg, 1989), and the island model (Whitley et al., 1999). We leave the application of niching to E-SPL for future work. 

**Sampling Policy for Evolution** Figure 15 evaluates the effect of using a fixed reference policy _π_ ref for system prompt evolution, rather than the continually changing RL policy _πθ_ . Using the fixed reference policy _π_ ref for evolutionary updates consistently improves the final performance of E-SPL. This supports our design choice to keep evolutionary operators stationary: allowing the editor itself to drift alongside the policy introduces instability and degrades the quality of system prompt edits. A fixed reference policy provides a stable basis for self-reflection, mutation, and crossover. 

_Figure 16._ **An example of how the RL model learns to utilize learned system prompts to solve problems** (also see Figure 22). 

Example of How E-SPL Shapes Model Behavior 

#### Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below: _· · ·_ **[G2]. For sums over integer pairs with conditions like gcd(** _**a, b**_ **) = 1, consider using Mobius inversion¨ or changing the order of summation to simplify the expression.** _· · ·_ 

Problem 

Let _S_ be the set of all positive integers less than 143 that are relatively prime to 143. Compute the number of ordered triples ( _a, b, c_ ) of elements of _S_ such that _a_ + _b_ = _c_ . 

Model Output **Let’s solve the problem step by step: Step 1** : Factorize 143 = 11 _×_ 13. So, _S_ consists of positive integers _<_ 143 that are coprime to both 11 and 13. _· · ·_ **Step 5** : **Use Mobius inversion (Principle¨ [G2]).** Let _f_ ( _n_ ) be the number of pairs ( _a, b_ ) _∈ S × S_ with _a_ + _b_ = _n_ . We want $\sum_{n=1}^{142}$ _[f]_[(] _[n]_[)] _[ ·]_ **[ 1]**[gcd(] _[n,]_[143)=1][.] _· · · · · ·_ **Step 11** : Apply inclusion-exclusion. _· · ·_ Therefore, the number of ordered triples ( _a, b, c_ ) is 5940. **Final Answer** : 5940 **Ground truth answer:** 5940 

## 5. Future Directions

The fundamental reason why E-SPL can work is that the base LLMs already have strong in-context learning and instruction following abilities. The RL process in itself does not fully take advantage of this – RL performs trial and error conditioned on an exogenous piece of context, without having the model think about how it can directly re-program itself by re-writing its own system instructions. E-SPL is a step towards augmenting RL with _self-rewrite_ , by jointly evolving a population of system prompts for selfconditioning. We outline a few promising future directions: 

**Agentic Retrieval for Long System Prompts** To build agents that can absorb an arbitrary amount of knowledge, the learned system prompt will eventually need a more refined sub-structure with many long components. A natural solution is to store certain components of the system prompt in the local file-system, and have the model access them in an agentic manner using a terminal and grep commands. 

**Self-Write as an RL Problem** So far, the LLM used for self-reflection and self-rewrite does not self-improve, which means that eventually it might become a bottleneck. It is possible to extend RL to the entire self-reflection and selfwrite process, but this would require additional research. 

**Self-Referential Recursive Self-Improvement** Our current context update and weight update algorithms are fundamentally limited; ultimately, the model should be allowed to make paradigm-level changes to its own learning algorithm. This can be achieved by allowing the model to directly modify its own training script and weights, thereby realizing fully self-referential self-improvement (Schmidhuber, 2003; 2005). A system capable of such root-level selfmodification, unlocking sustained, open-ended, unbounded improvements in its own capabilities, is also called Seed AI. (Yudkowsky, 2007). 

## 6. Conclusion

We have shown that unifying RL and evolutionary prompt search significantly enhances LLM self-improvement. E- SPL demonstrates that RL and Evolution are not competing paradigms but deeply synergistic ones. On a variety of reasoning and agentic tasks, our unified approach leads to better sample efficiency, asymptotic performance, and generalization. While the current method is not yet fully self-referential, it takes a meaningful step in that direction by allowing the model to self-modify its own system-level instructions. We believe further unification of Evolution and RL for self-rewrite can pave the way toward Seed AI. 

## References

- Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker, B., Bao, H., et al. gpt-oss-120b & gpt-oss-20b model card. _arXiv preprint arXiv:2508.10925_ , 2025. 

- Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. _arXiv preprint arXiv:2507.19457_ , 2025. 

- Azarbal, A., Gillioz, V., Ivanov, V., Woodworth, B., Drori, J., Wichers, N., Ebtekar, A., Cloud, A., and Turner, A. M. Recontextualization mitigates specification gaming without modifying the specification. _arXiv preprint arXiv:2512.19027_ , 2025. 

- Back, T., Fogel, D. B., and Michalewicz, Z.¨ Handbook of evolutionary computation. _Release_ , 97(1):B1, 1997. 

- Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_ , 2022a. 

- Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. Constitutional ai: Harmlessness from ai feedback. _arXiv preprint arXiv:2212.08073_ , 2022b. 

- Banzhaf, W., Francone, F. D., Keller, R. E., and Nordin, P. _Genetic programming: an introduction: on the automatic evolution of computer programs and its applications_ . Morgan Kaufmann Publishers Inc., 1998. 

- Bowman, S. R., Hyun, J., Perez, E., Chen, E., Pettit, C., Heiner, S., Lukosiˇ ut¯ e,˙ K., Askell, A., Jones, A., Chen, A., et al. Measuring progress on scalable oversight for large language models. _arXiv preprint arXiv:2211.03540_ , 2022. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33: 1877–1901, 2020. 

- ByteDance-Seed. Beyondaime: Advancing math reasoning evaluation beyond high school olympiads. _Hugging Face repository_ , 2025. 

- Cai, Y., Cai, S., Shi, Y., Xu, Z., Chen, L., Qin, Y., Tan, X., Li, G., Li, Z., Lin, H., et al. Training-free group relative policy optimization. _arXiv preprint arXiv:2510.08191_ , 2025. 

- Chen, A., Li, A., Gong, B., Jiang, B., Fei, B., Yang, B., Shan, B., Yu, C., Wang, C., Zhu, C., et al. Minimaxm1: Scaling test-time compute efficiently with lightning attention. _arXiv preprint arXiv:2506.13585_ , 2025. 

- Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., et al. Symbolic discovery of optimization algorithms. _Advances in neural information processing systems_ , 36:49205–49233, 2023. 

- Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _arXiv preprint arXiv:2507.06261_ , 2025. 

- Deb, K. and Goldberg, D. E. An investigation of niche and species formation in genetic function optimization. In _Proceedings of the third international conference on Genetic algorithms_ , pp. 42–50, 1989. 

- DeepSeek-AI. Deepseek-v3 technical report, 2024. URL https://arxiv.org/abs/2412.19437. 

- Dunefsky, J., Chlenski, P., and Nanda, N. Transcoders find interpretable llm feature circuits. _Advances in Neural Information Processing Systems_ , 37:24375–24410, 2024. 

- Eshelman, L. J. Preventing premature convergence in genetic algorithms by preventing incest. In _Proceedings of Fourth International Conference on Genetic Algorithms, 1991_ , 1991. 

- Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktaschel, T.¨ Promptbreeder: Self-referential self-improvement via prompt evolution. _arXiv preprint arXiv:2309.16797_ , 2023. 

- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. _arXiv preprint arXiv:2406.04093_ , 2024. 

- Golberg, D. E. Genetic algorithms in search, optimization, and machine learning. _Addion wesley_ , 1989(102):36, 1989. 

- Goldberg, D. E., Richardson, J., et al. Genetic algorithms with sharing for multimodal function optimization. In _Genetic algorithms and their applications: Proceedings_ 

- _of the Second International Conference on Genetic Algorithms_ , volume 4149, pp. 414–425. Lawrence Erlbaum, Hillsdale, NJ, 1987. 

- Gottweis, J., Weng, W.-H., Daryin, A., Tu, T., Palepu, A., Sirkovic, P., Myaskovsky, A., Weissenberger, F., Rong, K., Tanno, R., et al. Towards an ai co-scientist. _arXiv preprint arXiv:2502.18864_ , 2025. 

- Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_ , 2025. 

- Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., Liu, G., Bian, J., and Yang, Y. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. _arXiv preprint arXiv:2309.08532_ , 2023. 

- Hemberg, E., Moskal, S., and O’Reilly, U.-M. Evolving code with a large language model. _Genetic Programming and Evolvable Machines_ , 25(2):21, 2024. 

- Herbrich, R., Minka, T., and Graepel, T. Trueskill™: a bayesian skill rating system. _Advances in neural information processing systems_ , 19, 2006. 

- Holland, J. H. _Adaptation in natural and artificial systems: an introductory analysis with applications to biology, control, and artificial intelligence_ . MIT press, 1992. 

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. _ICLR_ , 1(2):3, 2022. 

- Hubert, T., Mehta, R., Sartran, L., Horvath, M. Z.,´ Zu[ˇ] ziˇ c, G.,´ Wieser, E., Huang, A., Schrittwieser, J., Schroecker, Y., Masoom, H., et al. Olympiad-level formal mathematical reasoning with reinforcement learning. _Nature_ , pp. 1–3, 2025. 

- Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., Vinyals, O., Green, T., Dunning, I., Simonyan, K., et al. Population based training of neural networks. _arXiv preprint arXiv:1711.09846_ , 2017. 

- Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. _arXiv preprint arXiv:2412.16720_ , 2024. 

- Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., Zamani, H., and Han, J. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. _arXiv preprint arXiv:2503.09516_ , 2025. 

- Kalyanakrishnan, S., Tewari, A., Auer, P., and Stone, P. Pac subset selection in stochastic multi-armed bandits. In _ICML_ , volume 12, pp. 655–662, 2012. 

- Kaufmann, E., Cappe,´ O., and Garivier, A. On bayesian upper confidence bounds for bandit problems. In _Artificial intelligence and statistics_ , pp. 592–600. PMLR, 2012. 

- Khadka, S. and Tumer, K. Evolution-guided policy gradient in reinforcement learning. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- Kharlapenko, D., Shabalin, S., Barez, F., Conmy, A., and Nanda, N. Scaling sparse feature circuit finding for incontext learning. _arXiv preprint arXiv:2504.13756_ , 2025. 

- Khatri, D., Madaan, L., Tiwari, R., Bansal, R., Duvvuri, S. S., Zaheer, M., Dhillon, I. S., Brandfonbrener, D., and Agarwal, R. The art of scaling reinforcement learning compute for llms. _arXiv preprint arXiv:2510.13786_ , 2025. 

- Kingma, D. P. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Koza, J. R. Genetic programming as a means for programming computers by natural selection. _Statistics and computing_ , 4(2):87–112, 1994. 

- Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al. Natural questions: a benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 7:453–466, 2019. 

- Langdon, W. B. and Poli, R. _Foundations of genetic programming_ . Springer Science & Business Media, 2013. 

- Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), _Proceedings of the 17th International Conference on Machine Learning (ICML 2000)_ , pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann. 

- Lee, K.-H., Chen, X., Furuta, H., Canny, J., and Fischer, I. A human-inspired reading agent with gist memory of very long contexts. _arXiv preprint arXiv:2402.09727_ , 2024. 

- Lehman, J. and Stanley, K. O. Abandoning objectives: Evolution through the search for novelty alone. _Evolutionary computation_ , 19(2):189–223, 2011a. 

- Lehman, J. and Stanley, K. O. Abandoning objectives: Evolution through the search for novelty alone. _Evolutionary computation_ , 19(2):189–223, 2011b. 

- Lehman, J., Gordon, J., Jain, S., Ndousse, K., Yeh, C., and Stanley, K. O. Evolution through large models. In _Handbook of evolutionary machine learning_ , pp. 331– 366. Springer, 2023. 

- Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., and Legg, S. Scalable agent alignment via reward modeling: a research direction. _arXiv preprint arXiv:1811.07871_ , 2018. 

- Lin, S. Computer solutions of the traveling salesman problem. _Bell System Technical Journal_ , 44(10):2245–2269, 1965. 

- Ma, H., Narayanaswamy, A., Riley, P., and Li, L. Evolving symbolic density functionals. _Science Advances_ , 8(36): eabq0279, 2022. 

- MacDiarmid, M., Wright, B., Uesato, J., Benton, J., Kutasov, J., Price, S., Bouscal, N., Bowman, S., Bricken, T., Cloud, A., et al. Natural emergent misalignment from reward hacking in production rl. _arXiv preprint arXiv:2511.18397_ , 2025. 

- Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with selffeedback. _Advances in Neural Information Processing Systems_ , 36:46534–46594, 2023. 

- Mahfoud, S. W. _Niching methods for genetic algorithms_ . University of Illinois at Urbana-Champaign, 1995. 

- McAleese, N., Pokorny, R. M., Uribe, J. F. C., Nitishinskaya, E., Trebacz, M., and Leike, J. Llm critics help catch llm bugs. _arXiv preprint arXiv:2407.00215_ , 2024. 

- Minka, T. P. _A family of algorithms for approximate Bayesian inference_ . PhD thesis, Massachusetts Institute of Technology, 2001. 

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. _nature_ , 518(7540): 529–533, 2015. 

- Mouret, J.-B. and Clune, J. Illuminating search spaces by mapping elites. _arXiv preprint arXiv:1504.04909_ , 2015. 

- Nguyen, S., Mei, Y., and Zhang, M. Genetic programming for production scheduling: a survey with a unified framework. _Complex & Intelligent Systems_ , 3(1):41–66, 2017. 

- Novikov, A., Vu, N., Eisenberger, M., Dupont, E., Huang,˜ P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J., Mehrabian, A., et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. _arXiv preprint arXiv:2506.13131_ , 2025. 

- Opsahl-Ong, K., Ryan, M. J., Purtell, J., Broman, D., Potts, C., Zaharia, M., and Khattab, O. Optimizing instructions and demonstrations for multi-stage language model programs. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pp. 9340–9366, 2024. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35:27730–27744, 2022. 

- Ouyang, S., Yan, J., Hsu, I., Chen, Y., Jiang, K., Wang, Z., Han, R., Le, L. T., Daruki, S., Tang, X., et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. _arXiv preprint arXiv:2509.25140_ , 2025. 

- Packer, C., Fang, V., Patil, S., Lin, K., Wooders, S., and Gonzalez, J. Memgpt: Towards llms as operating systems. _ArXiv_ , 2023. 

- Pourchot, A. and Sigaud, O. Cem-rl: Combining evolutionary and gradient-based methods for policy search. _arXiv preprint arXiv:1810.01222_ , 2018. 

- Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. _Advances in neural information processing systems_ , 36: 53728–53741, 2023. 

- Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J., Ellenberg, J. S., Wang, P., Fawzi, O., et al. Mathematical discoveries from program search with large language models. _Nature_ , 625 (7995):468–475, 2024. 

- Schmidhuber, J. Godel¨ machines: self-referential universal problem solvers making provably optimal selfimprovements. _arXiv preprint cs/0309048_ , 2003. 

- Schmidhuber, J. Completely self-referential optimal reinforcement learners. In _International Conference on Artificial Neural Networks_ , pp. 223–233. Springer, 2005. 

- Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In _International conference on machine learning_ , pp. 1889–1897. PMLR, 2015. 

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_ , 2017. 

- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_ , 2024. 

- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. _arXiv preprint arXiv:1701.06538_ , 2017. 

- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. _Advances in Neural Information Processing Systems_ , 36:8634–8652, 2023. 

- Shojaee, P., Meidani, K., Gupta, S., Farimani, A. B., and Reddy, C. K. Llm-sr: Scientific equation discovery via programming with large language models. _arXiv preprint arXiv:2404.18400_ , 2024. 

- Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. Mastering the game of go with deep neural networks and tree search. _nature_ , 529(7587):484–489, 2016. 

- Spears, W. M. Crossover or mutation? In _Foundations of genetic algorithms_ , volume 2, pp. 221–237. Elsevier, 1993. 

- Srinivas, N., Krause, A., Kakade, S. M., and Seeger, M. Gaussian process optimization in the bandit setting: No regret and experimental design. _arXiv preprint arXiv:0912.3995_ , 2009. 

- Stanley, K. O. and Miikkulainen, R. Evolving neural networks through augmenting topologies. _Evolutionary computation_ , 10(2):99–127, 2002. 

- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. _Advances in neural information processing systems_ , 33:3008–3021, 2020. 

- Sun, Z., Yu, L., Shen, Y., Liu, W., Yang, Y., Welleck, S., and Gan, C. Easy-to-hard generalization: Scalable alignment beyond human supervision. _Advances in Neural Information Processing Systems_ , 37:51118–51168, 2024. 

- Surina, A., Mansouri, A., Quaedvlieg, L., Seddas, A., Viazovska, M., Abbe, E., and Gulcehre, C. Algorithm discovery with llms: Evolutionary search meets reinforcement learning. _arXiv preprint arXiv:2504.05108_ , 2025. 

- Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. _Advances in neural information processing systems_ , 12, 1999. 

- Suzgun, M., Yuksekgonul, M., Bianchi, F., Jurafsky, D., and Zou, J. Dynamic cheatsheet: Test-time learning with adaptive memory. _arXiv preprint arXiv:2504.07952_ , 2025. 

- Tan, D., Woodruff, A., Warncke, N., Jose, A., Riche,´ M., Africa, D. D., and Taylor, M. Inoculation prompting: Eliciting traits from llms during training can suppress them at test-time. _arXiv preprint arXiv:2510.04340_ , 2025. 

- Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5: Scaling reinforcement learning with llms. _arXiv preprint arXiv:2501.12599_ , 2025. 

- Trinh, T. H., Wu, Y., Le, Q. V., He, H., and Luong, T. Solving olympiad geometry without human demonstrations. _Nature_ , 625(7995):476–482, 2024. 

- Vose, M. D. _The simple genetic algorithm: foundations and theory_ . MIT press, 1999. 

- Wang, R., Lehman, J., Clune, J., and Stanley, K. O. Poet: open-ended coevolution of environments and their optimized solutions. In _Proceedings of the genetic and evolutionary computation conference_ , pp. 142–151, 2019. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Whitley, D., Rana, S., and Heckendorn, R. B. The island model genetic algorithm: On separability, population size and convergence. _Journal of computing and information technology_ , 7(1):33–47, 1999. 

- Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine learning_ , 8(3):229–256, 1992. 

- Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In _First Conference on Language Modeling_ , 2024. 

- Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. A-mem: Agentic memory for llm agents. _arXiv preprint arXiv:2502.12110_ , 2025. 

- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., and Manning, C. D. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In _Proceedings of the 2018 conference on empirical methods in natural language processing_ , pp. 2369–2380, 2018. 

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In _The eleventh international conference on learning representations_ , 2022. 

- Ye, Z., Agarwal, R., Liu, T., Joshi, R., Velury, S., Le, Q. V., Tan, Q., and Liu, Y. Reward-guided prompt evolving in reinforcement learning for llms. In _Forty-second International Conference on Machine Learning_ , 2025. 

- Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. _arXiv preprint arXiv:2503.14476_ , 2025. 

- Yudkowsky, E. Levels of organization in general intelligence. In _Artificial general intelligence_ , pp. 389–501. Springer, 2007. 

- Zhang, J., Hu, S., Lu, C., Lange, R., and Clune, J. Darwin godel machine: Open-ended evolution of self-improving agents. _arXiv preprint arXiv:2505.22954_ , 2025a. 

- Zhang, L., Hosseini, A., Bansal, H., Kazemi, M., Kumar, A., and Agarwal, R. Generative verifiers: Reward modeling as next-token prediction. _arXiv preprint arXiv:2408.15240_ , 2024. 

- Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., et al. Agentic context engineering: Evolving contexts for self-improving language models. _arXiv preprint arXiv:2510.04618_ , 2025b. 

- Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. _arXiv preprint arXiv:2507.18071_ , 2025. 

- Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pp. 19724–19731, 2024. 

- Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., and Ba, J. Large language models are humanlevel prompt engineers. In _The eleventh international conference on learning representations_ , 2022. 

- Zhu, R. J. and Qiu, Y. Ucb exploration for fixedbudget bayesian best arm identification. _arXiv preprint arXiv:2408.04869_ , 2024. 

## A. Appendix Contents

### A.1. LLM-Based Self-Reflection for Mutation and Crossover

E-SPL implements both mutation and crossover as LLM-driven editing operators over past system prompts. The mutation operator consists of (i) a self-reflection stage that diagnoses performance from RL rollouts (which has two steps: trajectory summarization and self-critique), followed by (ii) a patch synthesis stage that applies structured edits to the parent prompt using a git-style diff (which has two steps: local edit proposals and local-to-global aggregation). On the other hand, the crossover operator has a single stage. 

**Mutation is Multi-Stage.** In each RL iteration of Alg. 1, we identify the best-performing system prompt 

$$k \leftarrow \arg\max_i V_i, \qquad V_i = \tfrac{1}{N}\sum_{j=1}^N R(x,y_{i,j})$$

where the returns are computed under matched conditions for all sampled prompts. 

**Self-Reflection Module in Mutation** Let _Ek_ = _{_ ( _x, yk,j, rk,j_ ) _}[N] j_ =1[denote the rollouts][produced under] **[s]** _[k]_[.][The self-] reflection prompt Ψreflect instructs the reference model _π_ ref to summarize these trajectories ( **Trajectory Summarization** , §A.2), attribute successes and failures to specific reasoning behaviors ( **Self-Critique** , §A.3), and produce lessons _ℓ_ : 

$$\ell \sim \pi_{\text{ref}}(\cdot \mid \Psi_{\text{reflect}}, \mathcal{E}_k)$$

**Local Edit Proposal (§A.4)** Given the reflection output _ℓ_ , we generate concrete edits to the parent system prompt: 

$$\mathrm{diff} \sim \pi_{\text{ref}}(\cdot \mid \mathbf{s}_k, \ell)$$

for each problem _x_ in the batch _B_ . Each diff is a list of local edits. Now we need a mechanism to aggregate those individual per-problem edits across a batch of problems. 

**Local-to-Global Aggregation (§A.5)** To reconcile overlapping edits to the system prompt proposed by parallel selfreflection processes, E-SPL aggregates these edits using a consolidation prompt that produces a global revision plan consisting of: 

- **modify** : refine an existing principle using accumulated evidence, 

- **merge** : combine semantically similar principles into a more general form. 

Each resulting principle in the system prompt should be short, abstract, and strategy-focused, preventing uncontrolled growth of the system prompt length. We then apply the global editing operations: 

$$\mathbf{s}_{\text{mutate}} \leftarrow \mathrm{git.apply}(\mathbf{s}_k, \mathrm{diff})$$

The resulting child system prompt is added to the population with inherited rating and increased uncertainty as in Alg. 1. The diff format is especially important as system prompts grow longer. 

**Crossover is Single-Stage.** The crossover operator (Alg. 2) differs from the mutation operator in the evidence used for reflection. For a batch of problems _B_ , we compute the empirical return matrix 

$$\Phi_{i,b} = \tfrac{1}{N}\sum_{j=1}^N R(x_b,y_{i,b,j})$$

by reusing rollouts already generated by RL. For each system prompt **s** _i_ , we identify which problems _ρi_ in the batch _B_ that **s** _i_ did best on, yielding a list _**φ**_ of prompt-specific wins. The crossover reflection prompt Ψcr is constructed from _**φ**_ to contrast the differential strengths of multiple parent prompts. 

The reference model then proposes edits applied to the overall top-performing prompt **s** _k_ : 

$$\mathrm{diff}_{\text{cr}} \sim \pi_{\text{ref}}(\cdot \mid \mathbf{s}_k, \Psi_{\text{cr}}), \qquad \mathbf{s}_{\text{crossover}} \leftarrow \mathrm{git.apply}(\mathbf{s}_k, \mathrm{diff}_{\text{cr}})$$

producing a child prompt that selectively recombines complementary instruction segments from multiple parents. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Add new system prompt Population of Add new system prompt System Prompts {( s i , ( μi , σi ))} Adjust ratings Select M System Prompts s 1 s 2 … s M Batch of b Batch of b Batch of b Batch of b Same batch of problems b Problems Problems Problems Problems Trajectories Trajectories Trajectories Sample from πθ ∼ πθ ( ⋅∣ x 1: b , s 1) ∼ πθ ( ⋅∣ x 1: b , s 2) … ∼ πθ ( ⋅∣ x 1: b , s M ) Performance of each s i AccuracyMean Per-problem Accuracy AccuracyMean Per-problem Accuracy … AccuracyMean Per-problem Accuracy Sort Mean Accuracy to Find problems where get relative rankings each was the top prompt s i Get top system prompt Update TrueSkill in the batch: s k ratings Self Reflection Module Self Reflection Module in Mutation in Crossover Add new system prompt from mutation New s mutate Add new system prompt from crossover New s crossover RL Update Policy Gradient Update to πθ

_Figure 17._ **Detailed overview of the E-SPL pipeline** , which illustrates how Figure 2, Figure 5, and Figure 6 fit into a single algorithm. 

### A.2. Trajectory Summarization in Mutation Self-Reflection

Before proposing local edits, E-SPL first converts raw RL rollouts into structured trajectory summaries. For the selected parent prompt **s** _k_ , all sampled solutions are grouped by problem instance _x_ . Let _Rx_ = _{_ ( _x, yk,j, rk,j_ ) _}[N] j_ =1[denote the group] of rollouts produced for problem _x_ under **s** _k_ . E-SPL only performs self-reflection on problems where the model has both made some mistakes and achieved some success (0 _< Vi <_ 1). Self-reflection focuses on cases where we can _contrast_ successes and failures under identical instructions. 

For each rollout in a problem-specific group, the reference model _π_ ref is prompted to generate a step-by-step summary describing: (i) the reasoning steps taken, (ii) which principles or strategies in the system prompt were (implicitly) used, and (iii) where detours or errors occurred, given the rollout outcome and ground-truth answer. 

These summaries preserve the full reasoning outcomes while explicitly highlighting failure modes, producing a compact representation suitable for higher-level self-critique. 

### A.3. Self-Critique in Mutation Self-Reflection

For each problem _x_ , E-SPL constructs a self-critique prompt that jointly presents: 

- multiple trajectory summaries from _Rx_ , labeled as successful or incorrect, 

- the current system prompt **s** _k_ , 

- the ground-truth answer. 

*[figure omitted — see original PDF]*

> **Figure text (extracted):** Self Reflection Module in Mutation Problem x 1 Agent Trajectories ∼ π ( ⋅∣ s k , x 1) SummarizationTrajectory Self Critique Local EditsPropose Problem x 2 Agent Trajectories ∼ π ( ⋅∣ s k , x 2) SummarizationTrajectory Self Critique Local EditsPropose Local-to-Global Aggregation Edits to s k ⋯ ⋯ ⋯ ⋯ ⋯ Problem xb Agent Trajectories ∼ π ( ⋅∣ s k , xb ) SummarizationTrajectory Self Critique Local EditsPropose

_Figure 18._ **Self-Reflection Module in Mutation** (where the best-performing system prompt **s** _k_ is edited) consists of four sequential modules: trajectory summarization (§A.2), self-critique (§A.3), local edit proposal (§A.4), local-to-global aggregation (§A.5). 

The reference model _π_ ref is instructed to analyze these contrasted trajectories to: 

1. identify reasoning patterns that consistently led to success, 

2. diagnose systematic failure modes, and consider additional guidance that would be needed to avoid those failure modes, 

3. determine whether existing principles in the system prompt are insufficient or require further editing. 

Based on this analysis, the model proposes a number of structured update operations: 

- **add** : introduce a new general principle (or high-level strategy / repeatable workflow) to the system prompt; 

- **modify** : refine an existing principle to better encode successful strategies or avoid observed errors. 

Each proposed principle is designed to express an abstract decision heuristic rather than problem-specific calculations, ensuring transferability across tasks. 

### A.4. Local Edit Proposal Module

To stabilize evolution and prevent overly aggressive prompt drift, E-SPL limits the number of edits proposed per problem instance. Specifically, each per-problem self-critique stage is restricted to produce at most _K_ ops edit operations, where _K_ ops is a tunable hyperparameter (set to _K_ ops = 2 in our experiments). The goal of this design is to encourage incremental knowledge accumulation and localized corrections driven by concrete failures. 

Empirically, bounding the number of local edits encourages steady refinement of the principle set in the system prompt while preserving useful structure learned in earlier generations. 

### A.5. Local-to-Global Aggregation Module

During a training step, multiple problems may independently propose local edits. Directly applying all such edits can lead to conflicting updates, redundancy, or uncontrolled growth of the principle set in the system prompt. 

To address this, E-SPL performs a batch aggregation stage that selectively combines the local edit proposals into a global revision plan. Given the selected system prompt **s** _k_ and the collection of proposed edits, the reference model is prompted to: 

- refine individual edits to incorporate repeated evidence, 

- merge semantically overlapping principles or strategies into more general formulations. 

The resulting revision plan consists of two types of operations: 

- **modify** operation on an existing principle, which aggregates all proposed edits to that principle across the batch, 

- **merge** operation, which replaces several related principles with a unified higher-level strategy / more universal rule across domains. 

All final principles in the system prompt are encouraged to be short, abstract, and strategy-oriented, yielding a compact declarative knowledge base that evolves steadily throughout training. 

We use the following hyper-parameters: 

- LoRA (Hu et al., 2022) rank of 32, learning rate _α_ =4e-5, using Adam optimizer (Kingma, 2014) with _β_ = (0 _._ 9 _,_ 0 _._ 95). 

- Batch size of 10 (problems per batch). Group size _N_ = 5 for the number of rollouts to sample per system prompt. 

- Sliding window size _K_ = 10 for the population _S_ . Number of system prompts per batch _M_ = 3. 

- Uncertainty in genetic operators: ∆ _σ_ = 1, uncertainty bonus: _λ_ = 2 _._ 0; the crossover probability _p_ crossover = 0 _._ 2. 

- TrueSkill (all defaults): _µ_ 0 = 25, _σ_ 0 = 25 _/_ 3, beta=25 _/_ 6, tau=25 _/_ 300, draw probability=0 _._ 10. 

- We find that a simplified selection strategy is quite effective: always add the system prompt with the highest score (score _i_ = _µi_ + _λσi_ ) to the batch, but randomly sample from the rest of the sliding window. 

- _K_ ops = 2, where _K_ ops is the maximum number of local edits per problem in the mutation operator. 

#### Root node system prompt for agentic search:

SEARCH_TASK_INSTRUCTIONS = """You are an expert assistant who solves tasks using a Wikipedia search tool. 

Here are instructions for how to solve a problem: 

1. Think step by step before calling the tool and after you receive the result of the tool call. Decide what queries to call the tool with. 

2. Call the tool with the queries you have decided on. 

3. Think step by step again after you receive the result of the tool call. If you have the information you need, you can stop here. 

4. Otherwise, come up with new queries that combine information from the previous results. 5. Include your final answer after the "Answer:" prefix. The answer should be between one to five words. 

Here is an example of solving a real question: 

"Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?" 

1. Think step by step: In order to answer this question, I need to know the population of New York City and San Francisco between 2020 and 2025. I will search for the population of New York City in each year 

2. Calling search tool: <tool_call>{"name": "search", "arguments": {"query_list": ["Population New York city between 2020 and 2025"]}}</tool_call> (Output omitted for brevity) 

3. Think step by step again: I have the population of New York City in each year, and I see that the population of New York City grew the most in 2024. I need to know the population of San Francisco in 2024. I will search for the population of San Francisco in each year. 

<tool_call>{"name": "search", "arguments": {"query_list": ["Population San Francisco between 2023 and 2024"]}}</tool_call> (Output omitted for brevity) 

4. Answer: The population of New York City grew the most in 2024, and the population of San Francisco changed by XXXX in 2024. """ 

## B. TrueSkill-Based Fitness Modeling for E-SPL

To aggregate noisy batch-wise prompt comparisons into persistent competence estimates, we adopt the Bayesian skill rating model **TrueSkill** (Herbrich et al., 2006). Each system prompt is treated as a player whose latent competence evolves over time and is updated through relative performance evidence obtained within each RL iteration. Unless otherwise stated, we use default TrueSkill hyperparameters: _µ_ 0 = 25 _._ 0, _σ_ 0 = 25 _._ 0 _/_ 3, _β_ = 25 _._ 0 _/_ 6, _τ_ = 25 _._ 0 _/_ 300, and draw probability parameter _p_ draw = 0 _._ 1. **For completeness** , we review the mathematical details of an off-the-shelf TrueSkill system below. 

**Skill and Performance Model.** Each system prompt **s** _i_ is associated with a Gaussian skill variable 

$$z_i \sim \mathcal{N}(\mu_i,\sigma_i^2)$$

where _µi_ denotes estimated competence and _σi_ its uncertainty. Observed performance is modeled as a noisy skill: 

$$p_i = z_i + \epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0,\beta^2)$$

so that the marginal performance variance is 

$$\mathrm{Var}(p_i)=\sigma_i^2+\beta^2$$

**Dynamics.** Before each rating update, we apply Gaussian drift _τ_ in skill space to model temporal uncertainty growth: 

$$\sigma_i^2 \leftarrow \sigma_i^2 + \tau^2$$

**Ranking Observations and Tie Encoding.** Within each RL iteration, matched rollouts yield scalar returns _{Vi}[M] i_ =1[for] selected prompts. Sorting these values produces a ranking from best to worst. Ties are encoded by an integer array ties such that adjacent positions _i_ and _i_ + 1 are treated as a draw if ties[ _i_ ] = ties[ _i_ + 1], and otherwise as a win for position _i_ over _i_ + 1. This yields a chain of _M −_ 1 adjacent pairwise constraints. 

**Draw Margin.** We first compute a global draw margin in performance-difference units: 

$$\epsilon_{\mathrm{abs}} = \Phi^{-1}\!\left(\frac{p_{\text{draw}}+1}{2}\right)\sqrt{2}\,\beta$$

We denote _ϕ_ ( _·_ ) and Φ( _·_ ) as the probability density function (pdf) and cumulative distribution function (cdf) of the standard normal distribution _N_ ; the Φ _[−]_[1] ( _·_ ) function above is the inverse (probit) function of Φ. 

For each pairwise comparison with total standard deviation _c_ , the normalized margin used in inference is 

$$\epsilon = \frac{\epsilon_{\mathrm{abs}}}{c}$$

**Truncated Gaussian Correction Functions.** For a normalized performance difference variable 

$$t=\frac{\mu_i-\mu_j}{c}$$

the one-sided (win/loss) correction terms are 

$$\begin{aligned} v_{\text{win}}(t,\epsilon) &= \frac{\phi(t-\epsilon)}{\Phi(t-\epsilon)}, \\ w_{\text{win}}(t,\epsilon) &= v_{\text{win}}(t,\epsilon)\big(v_{\text{win}}(t,\epsilon)+t-\epsilon\big) \end{aligned}$$

and the two-sided (draw) terms are 

$$\begin{aligned} v_{\text{draw}}(t,\epsilon) &=\frac{\phi(-\epsilon-t)-\phi(\epsilon-t)}{\Phi(\epsilon-t)-\Phi(-\epsilon-t)}, \\ w_{\text{draw}}(t,\epsilon) &=v_{\text{draw}}^2+\frac{(\epsilon-t)\phi(\epsilon-t)-(-\epsilon-t)\phi(-\epsilon-t)}{\Phi(\epsilon-t)-\Phi(-\epsilon-t)} \end{aligned}$$

where _ϕ_ and Φ denote the standard normal pdf and cdf. 

**Approximate Inference for Full Rankings.** For rankings with more than two prompts, we perform EP-style iterative message passing (Minka, 2001) over adjacent constraints in performance space. 

**Performance-space priors.** For each system prompt **s** _i_ and its corresponding rating _N_ ( _µi, σi_[2][)][, define] 

$$v_i=\sigma_i^2+\beta^2, \qquad \pi_i^{(0)}=\frac{1}{v_i}, \qquad \tau_i^{(0)}=\frac{\mu_i}{v_i}$$

**Messages.** For each adjacent constraint connecting system prompts _i_ = _k_ and _j_ = _k_ + 1, we maintain Gaussian messages _mk→i_ = ( _πk→i, τk→i_ ) and _mk→j_ , initialized to zero. 

**Cavity distributions.** When updating for the _k_ -th constraint, cavity parameters are formed by summing all other incoming messages: 

$$\pi_i^{\setminus k}=\pi_i^{(0)}+\sum_{c\neq k}\pi_{c\to i}, \qquad \tau_i^{\setminus k}=\tau_i^{(0)}+\sum_{c\neq k}\tau_{c\to i}$$

and similarly for _j_ . The corresponding cavity moments are 

$$\mu_i^{\setminus k}=\frac{\tau_i^{\setminus k}}{\pi_i^{\setminus k}}, \qquad \sigma_{i,\setminus k}^2=\frac{1}{\pi_i^{\setminus k}}$$

**Moment update.** Let 

$$c=\sqrt{\sigma_{i,\setminus k}^2+\sigma_{j,\setminus k}^2}, \qquad t=\frac{\mu_i^{\setminus k}-\mu_j^{\setminus k}}{c}, \qquad \epsilon=\frac{\epsilon_{\mathrm{abs}}}{c}$$

If the pair is a draw, use ( _v_ draw _, w_ draw); otherwise use ( _v_ win _, w_ win). The updated performance-space moments are 

$$\begin{aligned} \mu_i^{\text{post}} &= \mu_i^{\setminus k}+\frac{\sigma_{i,\setminus k}^2}{c}v, & \mu_j^{\text{post}} &= \mu_j^{\setminus k}-\frac{\sigma_{j,\setminus k}^2}{c}v, \\ \sigma_{i,\text{post}}^2 &= \sigma_{i,\setminus k}^2\!\left(1-\frac{\sigma_{i,\setminus k}^2}{c^2}w\right), & \sigma_{j,\text{post}}^2 &= \sigma_{j,\setminus k}^2\!\left(1-\frac{\sigma_{j,\setminus k}^2}{c^2}w\right) \end{aligned}$$

**Message extraction.** New messages are computed as the difference between posterior and cavity natural parameters: 

$$\begin{aligned} \pi_{k\to i} &= \max\!\left(0,\frac{1}{\sigma_{i,\text{post}}^2}-\pi_i^{\setminus k}\right), \\ \tau_{k\to i} &= \frac{\mu_i^{\text{post}}}{\sigma_{i,\text{post}}^2}-\tau_i^{\setminus k} \end{aligned}$$

and analogously for _j_ . Precision is clamped to be nonnegative for numerical stability. 

These updates are iterated over all adjacent constraints for a fixed number of sweeps or until message changes fall below a small threshold. 

**Final performance posteriors.** After convergence, 

$$\pi_i^{\text{perf}}=\pi_i^{(0)}+\sum_k \pi_{k\to i}, \qquad \tau_i^{\text{perf}}=\tau_i^{(0)}+\sum_k \tau_{k\to i}$$

yielding 

$$\mu_i^{\text{perf}}=\frac{\tau_i^{\text{perf}}}{\pi_i^{\text{perf}}}, \qquad \sigma_{i,\text{perf}}^2=\frac{1}{\pi_i^{\text{perf}}}$$

**Mapping Back to Skill Space.** Let the skill prior after dynamics be _N_ ( _µi, σi_[2][)][ and define] _[ c][i]_[=] _[ σ] i_[2][+] _[β]_[2][. Performance-space] natural parameters for posterior and prior are 

$$\pi_{\text{post}}=\frac{1}{\sigma_{i,\text{perf}}^2}, \quad \tau_{\text{post}}=\frac{\mu_i^{\text{perf}}}{\sigma_{i,\text{perf}}^2}, \qquad \pi_{\text{prior}}=\frac{1}{c_i}, \quad \tau_{\text{prior}}=\frac{\mu_i}{c_i}$$

The truncation-induced message is 

$$\pi_{\text{msg}}=\pi_{\text{post}}-\pi_{\text{prior}}, \qquad \tau_{\text{msg}}=\tau_{\text{post}}-\tau_{\text{prior}}$$

This message is mapped through the performance-noise link as 

$$\pi_{\text{msg}}^{\text{skill}}=\frac{\pi_{\text{msg}}}{1+\beta^2\pi_{\text{msg}}}, \qquad \tau_{\text{msg}}^{\text{skill}}=\frac{\tau_{\text{msg}}}{1+\beta^2\pi_{\text{msg}}}$$

when _π_ msg _>_ 0. The skill posterior is then 

$$\pi_i^{\text{skill}}=\frac{1}{\sigma_i^2}+\pi_{\text{msg}}^{\text{skill}}, \qquad \tau_i^{\text{skill}}=\frac{\mu_i}{\sigma_i^2}+\tau_{\text{msg}}^{\text{skill}}$$

with moments 

$$\sigma_i^{2}=\frac{1}{\pi_i^{\text{skill}}}, \qquad \mu_i=\frac{\tau_i^{\text{skill}}}{\pi_i^{\text{skill}}}$$

**Selection Scores.** Prompt sampling is guided by an upper confidence estimate (with _λ_ = 2 _._ 0) (Srinivas et al., 2009; Kaufmann et al., 2012; Kalyanakrishnan et al., 2012; Zhu & Qiu, 2024): 

$$\text{score}_i=\mu_i+\lambda \, \sigma_i$$

**Initialization for Mutation and Crossover.** A mutated child prompt inherits its parent’s mean while inflating uncertainty: 

$$\mu_{\text{child}}=\mu_k, \qquad \sigma_{\text{child}}=\sqrt{\sigma_k^2+\Delta\sigma^2}$$

For crossover among parents _{i}_ , we perform precision-weighted fusion: 

$$\mu'=\frac{\sum_i \mu_i/\sigma_i^2}{\sum_i 1/\sigma_i^2}, \qquad \sigma'^2=\frac{1}{\sum_i 1/\sigma_i^2}+\Delta\sigma^2$$

Each RL iteration supplies local comparative evidence that is accumulated into a coherent evolutionary fitness landscape. 

## C. Explanations for the identified mistakes in discovered principles

### C.1. Mistake #1 in Discovered Principles

**Original statement.** *For a polynomial $P(n) = \prod_{i=1}^m (n - a_i)$, the largest integer $d$ that divides $P(n)$ for all integers $n$ is $d = \prod_{p} p^{e_p}$, where $e_p = \min_{r \bmod p}\bigl(\#\{\, i : a_i \equiv r \pmod p \,\}\bigr)$. Only primes $p \le \max(a_i)$ need be considered, as for larger $p$ the $a_i$ are distinct modulo $p$, so $e_p = 0$. For primes where the function mapping $i$ to $a_i \bmod p$ is a permutation (e.g., if $a_i = i^3$ and $p \equiv 2 \bmod 3$), $e_p = \lfloor m/p \rfloor$.*

**Correction.** *For a polynomial $P(n) = \prod_{i=1}^m (n - a_i)$, the largest integer $d$ that divides $P(n)$ for all integers $n$ is $d = \prod_{p} p^{e_p}$, where $e_p = \min_{n \in \mathbb{Z}} v_p\!\bigl(P(n)\bigr)$. Moreover, one always has the lower bound $e_p \ge \min_{r \bmod p}\bigl(\#\{\, i : a_i \equiv r \pmod p \,\}\bigr)$. In particular, for this mod-$p$ bound it suffices to consider primes $p \le m$, since if $p > m$ then some residue class occurs zero times and the minimum is $0$. For primes where the function mapping $i$ to $a_i \bmod p$ is a permutation (e.g., if $a_i = i^3$ and $p \equiv 2 \bmod 3$), the bound gives $e_p \ge \lfloor m/p \rfloor$.*

**Explanation of the Mistake** We first recall the original principle. Let 

$$P(n)=\prod_{i=1}^m (n-a_i)$$

where _a_ 1 _, . . . , am ∈_ Z. It was claimed that the largest integer dividing _P_ ( _n_ ) for all _n ∈_ Z is 

$$d=\prod_p p^{e_p}, \qquad e_p=\min_{r \bmod p}\#\{\,i : a_i \equiv r \pmod p\,\}$$

This expression counts, for each prime _p_ , the minimum number of factors ( _n − ai_ ) that are divisible by _p_ for any choice of _n_ . We now recall the corrected formulation. For a prime _p_ and a nonzero integer _k_ , define the _p-adic valuation vp_ ( _k_ ) by 

$$v_p(k)=\max\{\,e\ge 0 : p^e \mid k\,\}$$

(For example, _v_ 2(12) = 2 since 12 = 2[2] _·_ 3.) Using this notation, the exact exponent of _p_ dividing all values of _P_ ( _n_ ) is 

$$e_p=\min_{n\in\mathbb{Z}} v_p\!\bigl(P(n)\bigr)$$

Since 

$$v_p\!\bigl(P(n)\bigr)=\sum_{i=1}^m v_p(n-a_i)$$

this definition measures the total number of powers of _p_ dividing the product, counting higher powers such as _p_[2] _, p_[3] , and so on. 

The original counting argument corresponds to considering only divisibility modulo _p_ . Fix a residue class _r_ mod _p_ and choose _n ≡ r_ (mod _p_ ). Then 

$$p \mid (n-a_i) \quad \Longleftrightarrow \quad a_i \equiv r \pmod p$$

Hence, for such _n_ , 

$$v_p\!\bigl(P(n)\bigr)\;\ge\;\#\{\,i : a_i \equiv r \pmod p\,\}$$

Minimizing over all residue classes _r_ mod _p_ yields the general lower bound 

$$e_p=\min_{n\in\mathbb{Z}} v_p\!\bigl(P(n)\bigr) \;\ge\; \min_{r \bmod p}\#\{\,i : a_i \equiv r \pmod p\,\}$$

This inequality explains why the correction is necessary. The original formula accounts only for the number of factors divisible by _p_ , but it does not capture the possibility that some factors ( _n − ai_ ) may be divisible by higher powers _p_[2] _, p_[3] _, . . ._ , which contribute additional terms to $v_p\!\bigl(P(n)\bigr)$. Such higher-power divisibility depends on congruences modulo _p_[2] _, p_[3] , and 

cannot be detected by working modulo _p_ alone. Therefore, the original expression cannot determine _ep_ exactly in general, but it always provides a valid lower bound. 

Finally, the restriction to primes _p ≤ m_ follows from a counting argument. If _p > m_ , then among the _p_ residue classes modulo _p_ and the _m_ integers _ai_ , at least one residue class is not attained. Choosing _n_ in that class gives 

$$v_p(n-a_i)=0 \quad \text{for all } i$$

and hence $v_p\!\bigl(P(n)\bigr) = 0$. Thus, no factor of _p_ is forced to divide _P_ ( _n_ ) in this case, and the mod- _p_ lower bound vanishes. 

**Summary.** The original mistake arises from conflating divisibility by _p_ with divisibility by higher powers of _p_ . Working modulo _p_ correctly identifies how many factors of ( _n − ai_ ) are _forced_ to contribute at least one factor of _p_ , and in many well-distributed cases this already determines the full exponent. However, the true fixed divisor depends on congruences modulo _p[k]_ for all _k ≥_ 1, which are invisible at the mod- _p_ level. 

### C.2. Mistake #2 in Discovered Principles

**Original statement.** _For graphs with odd-degree vertices, find a minimum T -join by pairing odd vertices along efficient paths. In symmetric graphs such as grids, pair boundary vertices optimally to minimize total length._ 

**Correction.** _For graphs with odd-degree vertices, find a minimum T -join by pairing odd vertices via a minimum-weight perfect matching on shortest-path distances. In symmetric graphs such as grids, restrict the matching to symmetry-respecting pairings to minimize total length._ 

**Background and notation.** Let _G_ = ( _V, E_ ) be an undirected graph with a nonnegative edge-length function _c_ : _E →_ R _≥_ 0. For a vertex _v ∈ V_ , its degree deg( _v_ ) is the number of incident edges. A vertex is _odd_ if deg( _v_ ) is odd. 

Let _T ⊆ V_ be a designated set of vertices with _|T |_ even. A _T -join_ is a multiset of edges _J_ such that in the subgraph induced by _J_ , the vertices of odd degree are _exactly_ the vertices in _T_ . The _minimum T -join_ problem is to find such a $J$ minimizing $\sum_{e\in J} c(e)$ (counting multiplicity if edges repeat). 

For _u, v ∈ V_ , let _d_ ( _u, v_ ) denote the shortest-path distance in _G_ under edge costs _c_ . Consider the complete graph on vertex set _T_ , with edge weights _d_ ( _u, v_ ). A _perfect matching_ on _T_ partitions _T_ into disjoint pairs; a _minimum-weight perfect matching_ minimizes the sum of the chosen pair weights. 

**Explanation of the Mistake** The original statement is incorrect as written because “pairing odd vertices along efficient paths” can be a local or greedy rule (e.g., repeatedly pairing “nearby” odd vertices). Such local decisions are not guaranteed to be globally optimal: choosing a short connection for one pair can force much longer connections for the remaining vertices, increasing the total length. A minimum _T_ -join is inherently a _global_ optimization problem over all pairings of _T_ . 

The correct principle is: compute all shortest-path distances _d_ ( _u, v_ ) for _u, v ∈ T_ , then solve a minimum-weight perfect matching on _T_ using these distances. Finally, for each matched pair ( _u, v_ ), add the corresponding shortest path in _G_ ; the symmetric difference of these paths forms a minimum _T_ -join. In symmetric graphs (e.g., grids), symmetry can reduce the search space by allowing one to focus on _symmetry-respecting_ pairings, but optimality still comes from minimizing the total matching cost under shortest-path distances rather than from ad-hoc or boundary-only pairings. 

### C.3. Discovered System Prompt on Agentic Search with gpt-oss-120b

You are an expert research assistant. Your ONLY job to answer the user's question by **[searching][the][web] **[and][returning][the] **[exact][phrase] **[found][in][an][authoritative][source.] 

- ***[Workflow][(follow] *[exactly] *[in][this][order)] 

0. *** INTERNAL NOTE** - Disambiguate the question if needed Write a comment that will NOT be sent sent to the user, e.g # chosen: Weekly Shonen Jump publisher 

1. **Predict target phrase** - Think step-by-step and write the exact wording you expect to see verbatim in double quotes. 

   - Example: Target phrase: "Shueisha" 

2. **Create ONE search query** that **contains the predicted phrase (of a very close paraphrase) inside double quotes** 

Example: ``` Query: \"Weekly Sh¨onen Jump\" \"Shueishal\"" ``` 

3. **Call the search tool once** you may call it a second time **only if every result from the first call is NOT VERIFIED**). Use **exactly** this syntax, with no extra spaces or line-breaks inside the tags: ``` <tool_call>{"name": "search", "arguments": {"query_list": ["<your query>"]))</tool_call> ``` 

4. **Verification block** - For each result returned, output the following four lines **in this order** and **nothing else**: 

````` Source: <domain> Summary: <one-sentence summary> Status: VERIFIED (or NOT VERIFIED) Phrase: "<exact phrase>" (include only if Status is VERIFIED) 

   - A result is **VERIFIED** **only if** 

      - a) the exact target phrase appears verbatim in the snippet **and** 

      - b) the domin ends with *.org *.edu, *.gov, is Wikipedia, **or** is a major reputable news outlet (e.g., nytimes.com, bbc.com, reuters.com, theguardian.com) 

   - If the domain is not on the list, mark the result **NOT VERIFIED** even if the phrase 

   - is present. 

5. **Refine query (optional)** - If **all** results are NOT VERIFIED, you may modify the 

query (still a single string) and repeat step 3. 

You may do this once only** (maximum two distinct queries) 

6. **Final answer** as you have *kat least one VERIFIED phrase**, output **exactly one line and nothing else: 

   - ``` Answer: <extracted phrase> ``` 

   - The phrase must contain **1 - 5 words** (words are are separated by spaces) 

7. **Fallback* - If after two distinct queries no VERIFIED phrase is found, output exactly: 

- ``` Answer: Not found ``` 

#### Critical formatting rules (must be obeyed)

- All tool calls must be wrapped **exactly** as ``` <tool_call>{...}</tool_call>with no extra 

- whitespace. 

- The only content sent to the user after the verification steps is the single "Answer: " line. 

- Do **not** include bullet points, markdown, blank lines, or any other text outside the structures described above. 

**[#][Positive][example][(full][correct][flow)] User: "What other name is used for the 4 × 400 m relay?" 

**chosen: 4x400 m relay Target phrase: "long relay" Query: "\"4 x 400 metres relay\" \"long relay\"" <tool call>("name". "search" "arguments": {"query_list": ("\"4 * 400 metres relay\" \"long relay\"]}}</tool_call> Source: en.wikipedia.org Summary: The 4 x 400 metres relay is also known as the "long relay". Status: VERIFIED Phrase: "long relay" Answer: long relay**

##* Negative example (what NOT to do) 

````` The 4 x 400 metres relay is also known as the long relay. Answer: long relay ````` *[Reason:] *[missing][internal][note,][missing][tool][call,][missing][verification][block,][extra] explanatory text. 

Follow **exactly** this loop for every user query. Your objective is to **prove** the answer with a verbatim phrase from a reliable source **before** you everr write the final ``` Answer: ``` line. 

#### Tools

#### functions

namespace functions{ 

// Searches the web for relevant information based on the given query. type search = (_: A list of fully-formed semantic queries. The tool will return search results for each query. query_list: string[], }) => any; 

}// namespace functions 

### C.4. Learned System Prompts on Math Datasets with DeepSeek v3.1

|**ID**|**Learned Principles in System Prompt (AIME)**|
|---|---|
|G0|When optimizing sums, set parameters to cancel dominant terms. For quadratic forms, choose parameters that minimize the|
||leading coeffcients, reducing the problem to analyzing remainder terms.|
|G1|For combinatorial counting problems, use generating functions with geometric series expansions. Apply the binomial|
||theorem to numerators and count solutions to resulting Diophantine equations using modular reduction techniques. Identify|
||independent parameters and multiply their counts accordingly.|
|G2|When counting nonnegative solutions to a linear Diophantine equation_a_1_x_1+_· · ·_+_anxn_ =_N_, use modular arithmetic to|
||reduce variables. For a prime_p_dividing some coeffcients, take the equation modulo_p_to constrain one variable, substitute|
||_xi_ =_c_+_pt_, simplify, and divide by common factors. Repeat until the equation is solvable by iteration or direct counting.|
|G3|After computations, verify results for consistency across domains: check denominators, symmetry, and physical plausibility.|
||Use alternative methods to recompute critical quantities and detect patterns such as periodicity.|
|G4|Avoid detours and unjustifed guesses, including contest heuristics without validation. Trust sound derivations; if inconsis-|
||tencies persist, explore alternative confgurations or identities.|
|G5|Reformulate problems by identifying a key parameter or ratio that simplifes constraints. Express conditions in terms of this|
||parameter to unlock algebraic or combinatorial insights, and seek optimal ratios to satisfy inequalities.|
|G6|For symmetric polynomial equations with a constraint (e.g., fxed sum), express the equation in terms of symmetric sums|
||_S_1_, S_2_, S_3. Form the cubic equation with the variables as roots and substitute the derived relation. If a constant root_t_=_k_|
||satisfes the equation for all symmetric sums, then at least one variable equals_k_. Verify by substitution.|
|G7|For key geometric quantities (e.g., volume or area), compute using at least two independent methods (e.g., coordinates, vector|
||cross products, or the Cayley–Menger determinant) to verify consistency and reduce the risk of formula misapplication.|
|G8|When setting up coordinates for polygons, avoid assuming perpendicularity unless explicitly given. For parallelograms,|
||place one vertex at the origin, one side along the_x_-axis, and parameterize the adjacent side using an angle_θ_. Verify that any|
||angle constraints (acute or obtuse) are satisfed.|
|G9|When initial assumptions lead to inconsistency, enumerate all plausible confgurations. For each, re-derive equations and|
||check consistency, using known results for standard cases to simplify.|
|G10|Avoid assuming symmetry (e.g., an isosceles triangle) without rigorous derivation. Conditions involving centers (incenter,|
||circumcenter) do not necessarily imply symmetry; explore all confgurations systematically.|
|G11|For problems involving a prime_p_ = _a_2 +_b_2 and derived quantities such as_x_ = _a_(_a_2 _−_3_b_2)and_y_ = _b_(3_a_2 _−b_2), the|
||constraint_|x −y| < p_is critical. This is minimized when_a/b ≈_2 +<br>~~_√_~~<br>3. Use continued fractions to generate convergents|
||and test resulting(_a, b_)pairs for primality and inequality constraints.|
|G12|For exponentiation or fraction manipulation, recompute critical arithmetic steps using alternative methods to avoid errors.|
||Perform step-by-step exponentiation and verify results via prime factorization when possible.|
|G13|For problems involving angle bisectors and trigonometric equations, use half-angle substitutions (e.g.,_u_= tan(_θ/_2)) to|
||expresssin_θ_andcos_θ_rationally, simplifying algebraic manipulation.|
|G14|For convex equilateral hexagons with opposite sides parallel, relate the distances between parallel sides to the triangle|
||formed by extending alternate sides. Use these relations to compute the hexagon side length effciently once the distances|
||are known.|
|G15|In a parallelogram, the sum of distances from any point to two parallel sides equals the distance between those sides. For a|
||point on the incircle, this sum equals twice the inradius, providing an invariant for computation and verifcation.|
|G16|For surfaces of revolution tangent along a circle, use cross-sectional analysis. Distinguish external from internal tangency by|
||comparing the distance between centers to the sum or difference of the relevant radii.|
|G17|For problems with sequential dependent events, apply the law of total probability. Enumerate all scenarios affecting the|
||desired event, compute conditional probabilities for each, and sum them weighted by their occurrence probabilities.|
|G18|For sequences defned by expressions involving irrational conjugates (e.g.,_Sn_ = _αn_ +_βn_), compute initial terms and|
||derive a recurrence from the minimal polynomial. Use the recurrence for modular or integrality analysis.|

|**ID**|**(Continued) Learned Principles in System Prompt (AIME)**|
|---|---|
|G19|For bipartite grid graphs with residue constraints, frst count valid assignments on one part using multinomial coeffcients.|
||Then determine the number of valid assignments on the other part, which is often constant due to symmetry.|
|G20|For problems with equal angles from boundary points to an interior point, use rotations to express the interior point as a|
||rotated vector from each boundary point. Equating these expressions yields solvable relations via spiral similarity.|
|G21|For tilted polyhedra with given vertex heights and a horizontal water surface, use vector representations. Express heights via|
||dot products with the vertical vector and compute submerged volume by integrating over the unit cube or via geometric|
||dissection.|
|G22|In modular arithmetic and sums, decompose expressions using residues and periodicity. Apply the Chinese Remainder|
||Theorem for coprime moduli and multiplicativity for square-free cases.|
|G23|For circle geometry, use power of a point, perpendicular bisectors, and homothety. Exploit symmetry to simplify equations|
||and avoid sign errors.|
|G24|In symmetric probabilistic problems, use complementary probability and symmetry to reduce cases. Verify that no hidden|
||asymmetries remain and compute failure probabilities carefully.|
|G25|For arrangements of lines or curves, model the confguration as a planar graph (including the point at infnity) and apply|
||Euler’s formula. Use envelopes and tangency conditions for curve families.|
|G26|In combinatorial problems, identify independent parameters that do not affect constraints and count them separately.|
||Systematically enumerate confgurations, using inclusion–exclusion or complementary counting when constraints overlap.|

|**ID**|**Learned Principles in System Prompt (DAPO)**|
|---|---|
|G0|In symmetric problems, assume an ordering of variables to reduce cases. Start from the most constrained element, using|
||bounds to limit the search space effectively.|
|G1|When a bound is stated as “does not exceed_M_,”, default to0_≤x ≤M_ unless specifed otherwise. This resolves common|
||contest-style ambiguity.|
|G2|For assignment problems with constraints, enumerate valid distributions frst. Count assignments multiplicatively: choose|
||groups for roles, then assign items using combinatorial methods.|
|G3|For subset problems with coprimality conditions, construct maximal invalid sets. Prove maximality by showing any|
||additional element forces the subset to become valid.|
|G4|For geometric problems with integer coordinates, translate conditions into algebraic equations. Use geometric structure to|
||derive Diophantine constraints and solve via exhaustive factorization.|
|G5|To verify modular injectivity, factor differences and check that the cofactor is nonzero modulo the modulus. For complex|
||cases, test small primes; derivative-based criteria can be suffcient but are not necessary.|
|G6|For packing problems with adjacency constraints, model feasible selections as independent sets and apply coloring bounds.|
||Solve small instances to detect patterns, then extend by periodic tilings with boundary checks. Validate the pattern on at|
||least two nontrivial small cases to avoid overgeneralization.|
|G7|For constrained optimization, use modular arithmetic and residue analysis to understand feasibility and to distribute|
||requirements across variables under sum or product constraints.|
|G8|When free parameters appear but a fxed answer is required, look for cancellations forced by constraints or boundary|
||conditions. Verify that the conclusion is consistent with all given conditions.|
|G9|For sequential-action problems with pattern-triggered bonuses (e.g., streak scoring), model the process dynamically: update|
||state after each step (e.g., streak length) and award bonuses immediately when the pattern occurs. Enumerate sequences by|
||tracking state to compute cumulative outcomes correctly.|
|G10|When using recalled facts or formulas, verify them with small cases or boundary checks. If evidence contradicts memory,|
||trust the evidence and re-derive the claim.|
|G11|For divisibility or coprimality constraints, derive modular conditions that restrict prime factors (e.g., no prime appearing in|
||more than two terms). Consider square-free versus repeated factors for minimization, but avoid prime powers in interior|
||terms.|

|**ID**|**Learned Principles in System Prompt (HMMT)**|
|---|---|
|G0|When dealing with geometric constraints, model positions using vectors or coordinates to interpret distances and directions.|
||Exploit symmetry in coordinate placement (e.g., centroid at the origin) only when justifed; otherwise, use general|
||orientations with parameters.|
|G1|Before solving equations or counting problems, express the target in terms of fundamental components (variables, prime|
||factors, or combinatorial elements). For divisibility of polynomials $\prod(n-a_i)$, decompose into prime powers and compute|
||the minimum exponent over all residues of_n_.|
|G2|When deriving polynomial identities from functional equations, arrange expressions as_g_(_x_)_Q_(_x_)_−P_(_x_) $= k\prod(x-r_i)$|
||to maintain consistent signs and avoid coeffcient errors.|
|G3|If a proportionality constant scales all coeffcients equally, it cancels in ratios of coeffcients. Compute such ratios directly|
||without solving for the constant.|
|G4|For graphs with odd-degree vertices, fnd a minimum_T_-join by pairing odd vertices along effcient paths. In symmetric|
||graphs such as grids, pair boundary vertices optimally to minimize total length.|
|G5|When decomposing angles at a vertex with multiple rays, express unknown angles using auxiliary variables tied to geometric|
||constraints (e.g., isosceles triangles). Avoid assuming angle equalities unless they follow from proven similarity or symmetry.|
|G6|For expected values over large or infnite structures, defne a recursive function for each state. Write equations by conditioning|
||on all possible transitions, adding immediate contributions and expected future values, and solve the resulting system.|
|G7|For expected waiting-time problems, use tail probabilities and linearity of expectation: E[T] = $\sum_{k=0}^{\infty}$<br>_k_=0 Pr(_T > k_)_._This is|
||often simpler than computing the full distribution of_T_.|
|G8|When combinatorial sums or series arise, search for known identities, generating functions, or combinatorial interpretations|
||to simplify them. Always validate results with small cases.|
|G9|For confgurations involving multiple touching shapes or tilings, determine contact points precisely using sketches or|
||coordinates. Avoid symmetry assumptions unless verifed, and use careful labeling to infer orientation.|
|G10|If a line from a vertex divides the opposite side into_n_equal segments, the areas of the resulting_n_triangles are equal. Use|
||this equal-area property to relate side lengths or products via area formulas.|
|G11|In triangle geometry, use identities such as_R_ = _a/_(2 sin_A_)and the Law of Cosines. For supplementary angles, apply|
||cos(180_◦−α_) =_−_cos_α_andsin(180_◦−α_) = sin_α_to simplify expressions.|
|G12|For iterative local rules on a circle, after_t_steps each value depends on a neighborhood of radius_t_. With continuous i.i.d.|
||inputs, the probability that adjacent values differ is1_/_(_t_+ 1)for max-type rules.|
|G13|For problems with state transitions or combinatorial counting, search for invariants (e.g., parity or bipartite coloring) to|
||derive bounds or impossibility results. Enumerate confgurations systematically, using symmetry only to reduce cases, not to|
||assume results.|
|G14|For tiling a rectangle with_a × b_rectangles (rotations allowed), partition one side into segments of length_a_or_b_. Each|
||segment must admit a unique tiling given divisibility constraints. The total number of tilings equals the number of valid|
||partitions times the tilings within each segment.|
|G15|For sequences of events that partition a set into intervals, defne gap variables between consecutive events and assign forced|
||properties to each interval. Translate constraints into equations on the gaps and count integer solutions.|
|G16|For grid problems with binary choices and local constraints, model the system as linear equations overGF(2). Account for|
||dependent equations that reduce rank. The number of solutions is2(cells_−_rank). Validate rank calculations using small cases.|
|G17|For iterative numerical transformations, seek invariants and apply modular analysis. In polynomial factorization, note that|
||factorization modulo all primes does not imply factorization overZ; use known counterexamples and structural results.|
|G18|For circular arrangements with pairwise constraints, encode constraints as a graph and reduce the problem to counting|
||Hamiltonian cycles. Fix the position of the most constrained element to break symmetry, enumerate cases, and multiply by|
||rotations and refections when appropriate.|

|**ID**|**(Continued) Learned Principles in System Prompt (HMMT)**|
|---|---|
|G19|In grid path problems, decompose the grid into diagonals (sets with constant_i_+_j_). Each path visits each diagonal exactly|
||once, simplifying counting and constraint handling.|
|G20|In an acute triangle, the projection identity_a_=_c_cos_B_+_b_cos_C_ holds, but similar-looking expressions need not simplify|
||analogously. Always verify projection formulas using a non-symmetric example.|
|G21|For placements in symmetric geometric structures, classify positions by symmetry type (corners, edges, faces). Model|
||constraints with coordinates and enumerate cases carefully, using symmetry only when it does not cause double-counting.|
|G22|For symmetric confgurations (e.g., roots of unity), identify the symmetry group preserving the condition. Generate|
||equivalent solutions via group actions and optimize the objective within an orbit.|
|G23|For sums involving periodic functions, use binary expansions and pairing arguments. If the number of terms is odd, isolate|
||the unpaired term. Validate results with small cases or known identities.|
|G24|For a chord intersected by circles, place a coordinate along the chord and use feet of perpendiculars from centers as reference|
||points. Apply the Pythagorean theorem for each circle to relate segment lengths and displacements.|
|G25|Before applying a geometric theorem, verify that its assumptions hold in the given confguration. When assumptions fail,|
||compute directly using coordinates or other robust methods.|
|G26|When counting partitions into indistinguishable groups, avoid introducing artifcial ordering. Divide out overcounting|
||factors (e.g.,_k_!) or use recursive constructions that inherently respect symmetry.|
|G27|For expressions of the form_ab_ mod_m_with composite_m_, decompose_m_into prime powers. Analyze primes dividing and|
||not dividing_a_separately, then combine results via the Chinese Remainder Theorem.|
|G28|For sums over coprime pairs with foor functions, reinterpret the foor as counting multiples. Use the unique decomposition|
||(_x, y_) =_k_(_a, b_)withgcd(_a, b_) = 1to transform the sum into a lattice-point counting problem.|
|G29|To maximize a sum under an LCM constraint, express terms as_g · pi_ with smalllcm(_pi_). Choose the largest_g_such that|
||_g ·_lcm(_pi_)_< K_.|
|G30|For divisibility conditions_m | ab −_1, compute the multiplicative order modulo_m_and verify that it divides_b_, checking|
||prime-power components carefully.|
|G31|For a cevian in a triangle, apply Stewart’s theorem to relate side lengths and the cevian length when the division ratio is|
||known. This is especially effective when multiple lengths are given.|
|G32|For a polynomial $P(n) = \prod_{i=1}^m (n-a_i)$, the largest integer dividing_P_(_n_)for all $n$ is $\prod_p p^{e_p}$,<br>_p pep_, where_ep_ = min_r_ mod _p_#_{i_:|
||_ai ≡r_ mod_p}_. Only primes up tomax_ai_ need consideration.|
|G33|In grid path problems where each move fips parity of a coordinate, the graph is bipartite. Use parity counts to bound path|
||length by the sizes of the bipartite parts.|
|G34|When polygons are labeled in order, interpret adjacency and opposition carefully. If vertices lie on a line, consider both|
||adjacent and opposite possibilities unless naming conventions or constraints rule one out.|
|G35|For quartic polynomials of the form_x_4 +_Cx_2 +_n_with_n_a square, test for factorization patterns and modular splittings.|
||Such polynomials may factor modulo all primes while remaining irreducible overZ; verify using small primes and known|
||examples.|

_Figure 19._ **Example of failure modes in E-SPL.** 

Example of How E-SPL Shapes Model Behavior 

#### Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below: 

**[G34]. When a polygon is described with vertices labeled in order (e.g., PQRS for a square), and specific vertices are constrained to lines, interpret the configuration carefully: consecutive vertices are adjacent, and opposite vertices are separated by one vertex. For a square, if two vertices are on the same line, they could be adjacent (side on the line) or opposite (diagonal on the line). Consider both possibilities if not specified, but use the naming to infer: typically, P and R are opposite. Verify with constraints (e.g., other vertices must lie on the intended lines).** 

#### Problem

Let _MARS_ be a trapezoid with _MA_ parallel to _RS_ and side lengths _MA_ = 11, _AR_ = 17, _RS_ = 22, and _SM_ = 16. Point _X_ lies on side _MA_ such that the common chord of the circumcircles of triangles _MXS_ and _AXR_ bisects segment _RS_ . Compute _MX_ . 

Model Output 

#### Let’s analyze the problem step by step:

Interpretation: Consider the two circumcircles: _ω_ 1 = circumcircle of ∆ _MXS_ , and _ω_ 2 = circumcircle of ∆ _AXR_ . Their common chord is the line through the intersection points of these two circles (if they intersect in two points). But here it says “the common chord bisects segment _RS_ ”. This likely means that the common chord (line) passes through the midpoint of _RS_ . Actually, “bisects” might mean that the chord itself is cut into two equal parts by the midpoint of _RS_ ? Or that the chord goes through the midpoint? Typically “bisects” means divides into two equal parts. So the common chord (line) intersects _RS_ at its midpoint. 

But careful: “bisects segment _RS_ ” means that the common chord (as a line) cuts _RS_ into two equal segments. So the intersection point of the common chord with _RS_ is the midpoint of _RS_ . Let _M_ be the midpoint of _RS_ . Since _RS_ = 22, midpoint _M_ ~~_R_~~ _S_ is such that _RM_ ~~_R_~~ _S_ = _M_ ~~_R_~~ _SS_ = 11. _· · ·_ 

Wait, could it be that the trapezoid is labeled in order _M_ , _R_ , _S_ , _A_ ? So that _MA_ is not a base? Then _MA ∥ RS_ means these are the legs? Unclear. **Given the instruction [G34]: “When a polygon is described with vertices labeled in order (e.g.,** _PQRS_ **for a square), and specific vertices are constrained to lines, interpret the configuration carefully”** . Here _MARS_ : usually it is in order _M − A − R − S_ . So bases _MA_ and _RS_ are parallel. Given the impasse, I will assume that the intended is that the radical axis (common chord) passes through the midpoint of _RS_ . And we will derive a condition on _x_ = _MX_ . 

Given the numbers, it is close. Perhaps with the correct non-degenerate placement, this holds when x is chosen. So in general, for the circles through _X_ , the power is still _NM ∗ NS_ because _M_ and _S_ are fixed on the circle. So indeed, the condition is _NM ∗ NS_ = _NA ∗ NR_ . So this must hold for the trapezoid. With our numbers, it does not. So perhaps the intended is that this products are equal, which would determine the shape. _· · ·_ After some research, I find that for this problem, the answer is _MX_ = 6. So I will box that. 

#### Final Answer :

**Ground truth answer:** 8 _._ 5 

6 

_Figure 20._ **An example of how the RL model learns to utilize learned system prompts to solve problems.** 

Example of How E-SPL Shapes Model Behavior 

#### Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below: 

**[G1]. Before solving equations or counting problems, express the target in terms of fundamental components (variables, prime factors, or combinatorial elements). Seek simplifications through combinations, identities, or auxiliary variables (such as gap variables for sequences) to reduce unknowns.** 

**[G21]. For problems involving placements in symmetric geometric structures (e.g., cubes, grids), classify positions by their symmetry properties (e.g., corners, edges, face-centers) and use coordinate systems to model constraints. Enumerate cases systematically, ensuring that each configuration is counted exactly once by leveraging symmetries only when appropriate.** 

#### Problem

Ashley fills each cell of a 3 _×_ 3 grid with some of the numbers 1 _,_ 2 _,_ 3 _,_ and 4 (possibly none or several). Compute the number of ways she can do so such that each row and each column contains each of 1 _,_ 2 _,_ 3 _,_ and 4 exactly once. (One such grid is shown below.) 

12 34 4 1 23 3 2 14 

#### Model Output

**Let’s analyze the problem step by step:** We have a 3 _×_ 3 grid. Each cell can contain a subset of _{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ (possibly empty, possibly with repeats? But wait: the constraints say that each row and each column must contain each of 1 _,_ 2 _,_ 3 _,_ 4 _exactly once_ . This means that in each row, the number 1 must appear in exactly one cell (and similarly for 2 _,_ 3 _,_ 4). Same for each column. _· · ·_ 

So indeed each number defines a permutation matrix. And these are independent. The total number is (3!)[4] . 

But is there any additional constraint? The problem says “such that each row and each column contains each of 1 _,_ 2 _,_ 3 _,_ 4 exactly once”. This is exactly achieved by having for each number a permutation matrix. And since the choices for different numbers are independent, the count is 6[4] . 

#### Verification with principles :

- **[G1]: We expressed the problem in terms of independent permutation matrices for each number.** 

- **[G21]: We used the symmetric structure (each number treated equally).** 

Thus, the number of ways is 6[4] = 1296. The answer is 1296. 

**Final Answer** : 

1296 

**Ground truth answer:** 1296 

_Figure 21._ **Example of failure modes in E-SPL.** 

Example of How E-SPL Shapes Model Behavior 

#### Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below: 

**[G2]. For Diophantine problems with sparse solutions, derive an approximate ratio from asymptotic behavior. Test all candidates within a tolerance of the target value, including semi-convergents, to avoid inefficient brute-force while ensuring completeness.** 

**[G5]. For Diophantine equations or sums with linear parameter dependence, use modular arithmetic to constrain variables and asymptotic analysis. When lifting solutions modulo prime powers (especially** _**p**_ **= 2), compute initial corrections and check for periodicity to characterize solutions efficiently over large ranges.** 

**[G26]. After deriving results, perform sanity checks: test small instances, ensure bounds, and verify extreme cases. Re-examine complex steps if results seem off.** 

#### Problem

Find the minimal positive integer _n_ ⩾ 866 satisfying: there exists a bijection _f_ from _{_ 1 _, · · · , n}_ to itself, such that for any _x, y ∈{_ 1 _, · · · , n}_ , if _n_ divides 866 _x − y_ then ( _n_ + 1) _| f_ ( _x_ )[866] _− f_ ( _y_ ). 

Model Output 

**Let’s analyze the problem step by step:** 

Interpret the conditions: 

1. _n |_ (866 _x − y_ ) means _y ≡_ 866 _x_ (mod _n_ ). Since _x, y ∈{_ 1 _, . . . , n}_ , this defines a permutation _σ_ on Z _/n_ Z by _σ_ ( _x_ ) = 866 _x_ mod _n_ . However, note that _n_ might not be prime, so we need to consider the structure. 

2. The condition requires that whenever _y ≡_ 866 _x_ (mod _n_ ), we have _f_ ( _x_ )[866] _≡ f_ ( _y_ ) (mod _n_ + 1). Since _f_ is a bijection on _{_ 1 _, . . . , n}_ , we can think of it as a permutation. The condition relates the permutation _f_ to the multiplication-by-866 map modulo _n_ . Let’s denote _g_ ( _x_ ) = _f_ ( _x_ ). The condition becomes: If 866 _x ≡ y_ (mod _n_ ), then _g_ ( _x_ )[866] _≡ g_ ( _y_ ) (mod _n_ + 1). This suggests that _f_ should be chosen to “conjugate” the map _x ↦_ 866 _x_ mod _n_ to the map _z ↦ z_[866] mod ( _n_ + 1). _· · ·_ 

In fact, if gcd(866 _, n_ ) = _d >_ 1, then _ϕ_ is not one-to-one. So we need to handle the case where _ϕ_ has fibers. **This is more complex. Given the instructions, especially [G5] and [G26], we should test small values near** 866 **.** _· · ·_ 

Given the complexity, and since the problem asks for the minimal _n ≥_ 866, it is likely that _n_ = 866 does not work. **After reading the instructions, [G2] suggests to derive an approximate ratio and test candidates. But here, the condition is very number-theoretic. Another idea:** _· · ·_ 

**Given the time** , and based on known results, the correct answer is _n_ = 866. 

**Final Answer** : 

$\boxed{866}$

**Ground truth answer:** 876 

_Figure 22._ **An example of how the RL model learns to utilize learned system prompts to solve problems.** 

Example of How E-SPL Shapes Model Behavior 

#### Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below: 

**[G2]. For sums over integer pairs with conditions like gcd(** _**a, b**_ **) = 1, consider using Mobius inversion or¨ changing the order of summation to simplify the expression.** 

#### Problem

Let _S_ be the set of all positive integers less than 143 that are relatively prime to 143. Compute the number of ordered triples ( _a, b, c_ ) of elements of _S_ such that _a_ + _b_ = _c_ . 

Model Output 

#### Let’s solve the problem step by step:

Problem: Let _S_ be the set of all positive integers less than 143 that are relatively prime to 143. Compute the number of ordered triples ( _a, b, c_ ) of elements of _S_ such that _a_ + _b_ = _c_ . 

**Step 1** : Factorize 143 = 11 _×_ 13. So, _S_ consists of positive integers _<_ 143 that are coprime to both 11 and 13. **Step 2** : Count the size of _S_ . The Euler totient function _ϕ_ (143) = _ϕ_ (11) _ϕ_ (13) = 10 _×_ 12 = 120. So, _|S|_ = 120. **Step 3** : Understand the condition _a_ + _b_ = _c_ . We need to count ordered pairs ( _a, b_ ) with _a, b ∈ S_ such that _c_ = _a_ + _b ∈ S_ . Note: Since _a, b <_ 143, _c_ = _a_ + _b <_ 286. But _c_ must be in _S_ , so _c <_ 143. Therefore, _a_ + _b <_ 143. **Step 4** : Reformulate the problem. We need to count the number of pairs ( _a, b_ ) _∈ S × S_ such that _a_ + _b <_ 143 and gcd( _a_ + _b,_ 143) = 1. 

**Step 5** : **Use Mobius inversion (Principle¨ [G2])** . Let _f_ ( _n_ ) be the number of pairs ( _a, b_ ) _∈ S × S_ with _a_ + _b_ = _n_ . We want $\sum_{n=1}^{142}$ _[f]_[(] _[n]_[)] _[ ·]_ **[ 1]**[gcd(] _[n,]_[143)=1][.][Alternatively, we can count pairs][ (] _[a, b]_[)][ with] _[ a, b][∈][S]_[and] _[ a]_[ +] _[ b]_[=] _[n]_[, and] then sum over _n_ coprime to 143. But note: _S_ is defined modulo 143. Since 143 is square-free, we can use the multiplicative property. 

**Step 10** : Count _N_ 11: number of pairs with _a_ + _b <_ 143 and _a_ + _b_ divisible by 11. Similarly, by symmetry, the number of pairs with _a_ + _b_ divisible by 11 should be roughly 1 _/_ 11 of the total, but we have the bound. Use the same symmetry idea: For each divisor d (11 or 13), the number of pairs with _a_ + _b_ divisible by d and _a_ + _b <_ 143 is equal to the number with _a_ + _b_ divisible by d and _a_ + _b >_ 143, because of the mapping ( _a, b_ ) _→_ (143 _− a,_ 143 _− b_ ). And also there are some with _a_ + _b_ = 143, but 143 is divisible by 11 and 13. In fact, let _Md_ be the number of pairs with _a_ + _b_ = 143 and divisible by d. Since 143 is divisible by both, _M_ 11 = _M_ 13 = _M_ = 120. Now, let _Td_ be the total number of pairs with _a_ + _b_ divisible by d. Then by symmetry, the number with _a_ + _b <_ 143 and divisible by d is ( _Td − Md_ ) _/_ 2. _· · ·_ Wait, because for each fixed mod 11 pair, there are 12 choices for a mod 13 and 12 for b mod 13. So _T_ 11 = 1440. Similarly, _T_ 13 = number with _a_ + _b ≡_ 0 (mod 13). Mod 13: number of (i,j) with _i_ + _j ≡_ 0: 12 choices (since _S_ 13 has size 12), and for each, one j. And mod 11: 10 choices for a and 10 for b. So _T_ 13 = 12 _∗_ (10 _∗_ 10) = 12 _∗_ 100 = 1200. Now, for a+b ¡ 143 and divisible by 11: _N_ 11 = ( _T_ 11 _− M_ 11) _/_ 2 = (1440 _−_ 120) _/_ 2 = 1320 _/_ 2 = 660. Similarly, _N_ 13 = ( _T_ 13 _− M_ 13) _/_ 2 = (1200 _−_ 120) _/_ 2 = 1080 _/_ 2 = 540. 

**Step 11** : Apply inclusion-exclusion. Number of good pairs = _N − N_ 11 _− N_ 13 = 7140 _−_ 660 _−_ 540 = 7140 _−_ 1200 = 5940. Therefore, the number of ordered triples ( _a, b, c_ ) is 5940. 

#### Final Answer :

**Ground truth answer:** 5940 

5940 

39
