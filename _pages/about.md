---
permalink: /
title: ""
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a master student at **ETH Zurich** majoring in **Robotics**. I am interested in all topics surrounding **Self-driving Vehicles**. More specific, I am interested in developing robots capable to interact with humans in an intelligent manner. I am also interested in the interplay between **Machine Learning** and **Optimization Based Algorithms** for robot motion planning, control and prediction/forecasting. Previously I was a bachelor student studying **Mechatronics** at **Tec de Monterrey**, during my bachelor I had the opportunity to intern at **Daimler AG** in the automated trucks department and at **Crabi SA** as a data scientist.

During my time at **ETH Zurich**, I had the pleasure to be part of **AMZ driverless**. An inter-disciplinary robotics team with the goal of building the fastest formula style autonomous car. We managed to win two international competitions, **Formula Student Germany** and **Formula Student East** with our vehicle, **pilatus driverless**.

After AMZ driverless, I spent time at **Embotech AG** where I designed optimal control and estimation algorithms for commercial vehicles as part of my master's degree internship. Recently, I have been interested in how to use **Machine Learning** to assist classical robotics algorithms.

![pilatus_driverless](/images/pilatus_wide.jpg)

## Publications

<table class="table table-hover">
<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/2003.04882' target='_blank'><img src="/images/control_diagram.jpg"/></a> </td>
<td>
    <strong>Optimization Based Hierarchical Motion Planning for Autonomous Racing</strong><br>
    <strong>J. Vazquez*</strong>, M. Brühlmeier*, A. Liniger*, A. Rupenyan, J. Lygeros<br>
    IROS 2020 <a style="color:red">Best Paper Award Finalist</a> <br>
    
[<a href='javascript:;'
    onclick='$("#abs_vazquez2020hierarchical").toggle()'>abstract</a>] [<a href='https://arxiv.org/abs/2003.04882' target='_blank'>arXiv</a>] <br>
    
<div id="abs_vazquez2020hierarchical" style="text-align: justify; display: none" markdown="1">
In this paper we propose a hierarchical controller for autonomous racing where the same vehicle model is used in a two level optimization framework for motion planning. The high-level controller computes a trajectory that minimizes the lap time, and the low-level nonlinear model predictive path following controller tracks the computed trajectory online. Following a computed optimal trajectory avoids online planning and enables fast computational times. The efficiency is further enhanced by the coupling of the two levels through a terminal constraint, computed in the high-level controller. Including this constraint in the real-time optimization level ensures that the prediction horizon can be shortened, while safety is guaranteed. This proves crucial for the experimental validation of the approach on a full size driverless race car. The vehicle in question won two international student racing competitions using the proposed framework; moreover, our hierarchical controller achieved an improvement of 20% in the lap time compared to the state of the art result achieved using a very similar car and track.
</div>
    
</td>
</tr>

</table>

## Projects

<table class="table table-hover">
<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/2003.04882' target='_blank'><img src="/images/projects/cartpole.png"/></a> </td>
<td>
    <strong>Assessing Generalization in Probabilistic Ensembles for Model-Based Deep Reinforcement Learning</strong><br>
    M. Bayerle, S. Panighetti, B. Tearle, <strong>J. Vazquez</strong><br>
    Deep Learning -- Final Project<br>
    
[<a href='javascript:;'
    onclick='$("#abs_dl_project").toggle()'>description</a>] [<a href='https://arxiv.org/abs/2003.04882' target='_blank'>report</a>] <br>
    
<div id="abs_dl_project" style="text-align: justify; display: none" markdown="1">
Generalization in reinforcement learning refers to an agent’s ability to perform outside of the environment it was trained in. Reinforcement learning (RL) algorithms are typically both trained and tested on fixed environments, which can result in over-fitting to the system under analysis. Since model-based RL (MBRL) methods attempt to learn the underlying system dynamics, they may perform better in generalization tasks compared to model-free methods, which directly learn a policy. Probabilistic Ensembles with Trajectory Sampling (PETS) [1] is a MBRL algorithm that is both sample efficient and high-performing on standard benchmark tasks. In this paper, we study how well PETS is able to generalize outside of the parameters it was trained on using a recently proposed generalization standard and modifiable gym environments from [2]. We find that PETS performs lower on these benchmarks compared to current state of the art model-free methods, and there is no apparent benefit of a standard model-based algorithm in terms of performance under model mismatch.
</div>

</td>
</tr>

</table>



