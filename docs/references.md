# Primary references

This list separates the sources that define the dataset, motivate the methods,
and constrain interpretation. A citation here does not imply that the project
implements every method in the cited work.

## Dataset and sample weight

- Becker, B., and Kohavi, R. (1996).
  [Adult](https://archive.ics.uci.edu/dataset/2/adult). UCI Machine Learning
  Repository. <https://doi.org/10.24432/C5XW20>

  Primary source for the benchmark task, row count, variables, missing values,
  and license information.

- UCI Machine Learning Repository.
  [Adult data description](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names).

  Original data dictionary. Its `fnlwgt` note describes the final weight as an
  estimate of how many people in the population one record represents. This is
  why the project excludes it from predictors and uses it only for labelled
  sensitivity analysis.

## Fair classification and subgroup auditing

- Hardt, M., Price, E., and Srebro, N. (2016).
  [Equality of Opportunity in Supervised Learning](https://proceedings.neurips.cc/paper_files/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0-Abstract.html).
  Advances in Neural Information Processing Systems 29.

  Defines equality of opportunity and studies post-processing of learned
  predictors. This project uses TPR gaps as one audit view, not as a complete
  definition of fairness.

- Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., and Wallach, H. (2018).
  [A Reductions Approach to Fair Classification](https://proceedings.mlr.press/v80/agarwal18a.html).
  Proceedings of Machine Learning Research 80:60-69.

  Demonstrates explicit error and fairness constraints and reinforces the need
  to treat policy choice as an optimization problem rather than a single metric
  edit. The implementation here is an exhaustive validation threshold search,
  not the reductions algorithm from the paper.

- Kearns, M., Neel, S., Roth, A., and Wu, Z. S. (2018).
  [Preventing Fairness Gerrymandering: Auditing and Learning for Subgroup Fairness](https://proceedings.mlr.press/v80/kearns18a.html).
  Proceedings of Machine Learning Research 80:2569-2577.

  Motivates looking beyond marginal groups. This project reports observed sex
  by original-race cells with support and evidence states; it does not implement
  the paper's rich subgroup learner.

## Statistical evidence

- Cherian, J. J., and Candes, E. J. (2024).
  [Statistical Inference for Fairness Auditing](https://jmlr.org/papers/v25/23-0739.html).
  Journal of Machine Learning Research 25(149):1-49.

  Shows why subgroup auditing requires statistical inference and multiple-group
  care. The project's bootstrap and Wilson intervals are narrower descriptive
  tools and do not implement the paper's simultaneous inference guarantees.

## Abstention and review

- Geifman, Y., and El-Yaniv, R. (2017).
  [Selective Classification for Deep Neural Networks](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
  Advances in Neural Information Processing Systems 30.

  Establishes the coverage and risk framing for selective classification. This
  project uses a validation-selected global review band and reports held-out
  coverage and automated error. It does not claim the paper's high-probability
  risk guarantee.

## Risk management and employment context

- Tabassi, E. (2023).
  [Artificial Intelligence Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1).
  NIST AI 100-1.

  Primary US risk-management framework. The project's evidence inventory and
  fail-closed gate align with a measure-and-manage mindset, but use of this
  repository does not establish conformity with the framework.

- US Equal Employment Opportunity Commission.
  [Employment Tests and Selection Procedures](https://www.eeoc.gov/laws/guidance/employment-tests-and-selection-procedures).

  Official technical assistance explaining that employment procedures must be
  considered in their actual job context and may require job-related validation.
  Adult income classification cannot provide that evidence.

- Electronic Code of Federal Regulations.
  [29 CFR Part 1607, Uniform Guidelines on Employee Selection Procedures](https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1607).

  Primary regulatory text for the US Uniform Guidelines. The project's DI gate
  is an engineering criterion and must not be treated as a legal safe harbor or
  substitute for the Guidelines' validation requirements.

## Citation rule for project results

When reporting an experiment, cite the run ID, Git commit, source SHA-256, data
SHA-256, resolved-config SHA-256, seed, model type, and gate verdict. Link or
attach the bound `audit.html` and `report.json`. Do not cite a metric without its
policy condition, interval when available, and relevant evidence limitations.
