---
title: Locally Weighted Ensembling: An Implementation and Analysis
author: Kennan LeJeune
date: December 7, 2019
header-includes:
  - \usepackage{times}
  - \usepackage{fullpage}
---

# Introduction

The Locally Weighted Ensembling algorithm is a Transfer Learning method used to learn a target task $\tau$ on a testing domain $T$ by training models $M_1, M_2, \ldots, M_k$ on training domains $D_1, D_2, \ldots, D_k$.  For any example $x$, we weight the model predictions according to their performance in the neighborhood of other examples clustered near $x$, making an overall prediction with a weighted average of the model outputs at $x$.

