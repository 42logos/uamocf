# uamocf — Uncertainty-Aware Multi-Objective Counterfactuals 

**uamocf** is a PyTorch implementation of **multi-objective counterfactual explanation generation** using  **NSGA-II**.  
Unlike most existing counterfactual methods that combine objectives into a single scalar loss (often distance-relevant), uamocf **explicitly models prediction uncertainty as an additional objective**, producing a **Pareto front** of diverse, non-dominated counterfactuals.
