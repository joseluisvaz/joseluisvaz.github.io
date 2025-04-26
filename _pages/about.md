---
permalink: /
title: ""
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a roboticist interested in building algorithms and tools to enable autonomous robots at scale. I am currently a Senior Software Engineer in the Behavior team at Nuro. Previously, I have worked as a software/planning engineer at autonomy companies like Argo AI, Kodiak Robotics, and Ford.

Before working in industry, I completed a master's degree in robotics at ETH Zurich; and before that, I did my undergraduate studies in mechatronics at Tec de Monterrey, where I discovered my passion for robotics through engaging projects and internships abroad.

If I had to name a single experience that defined my robotics career, I would mention my time at AMZ driverless, an interdisciplinary robotics team with the goal of building the fastest formula-style autonomous car. We managed to win two international competitions, Formula Student Germany and Formula Student East, with our vehicle, pilatus driverless.

![pilatus_driverless](/images/pilatus_wide.jpg)

## Publications

<table class="table table-hover">
<tr>
<td class="col-sm-2" width="40%"><a href='https://arxiv.org/pdf/2204.02392.pdf' target='_blank'><img src="/images/l4dc_animation.gif"/></a> </td>
<td>
    <strong>Deep Interactive Motion Prediction and Planning: Playing Games with Motion Prediction Models</strong><br>
    <strong>J. Vazquez</strong>, A. Liniger, W. Schwarting, D. Rus, L. van Gool<br>
    L4DC 2022 <br>
    
[<a href='javascript:;'
    onclick='$("#abs_vazquez2022interactive").toggle()'>abstract</a>] [<a href='https://arxiv.org/pdf/2204.02392.pdf' target='_blank'>arXiv</a>] [<a href='https://sites.google.com/view/deep-interactive-predict-plan' target='_blank'>website</a>] <br>
    
<div id="abs_vazquez2022interactive" style="text-align: justify; display: none" markdown="1">
In most classical Autonomous Vehicle (AV) stacks, the prediction and planning layers are separated, limiting the planner to react to predictions that are not informed by the planned trajectory of the AV. This work presents a module that tightly couples these layers via a game-theoretic Model Predictive Controller (MPC) that uses a novel interactive multi-agent neural network policy as part of its predictive model. In our setting, the MPC planner considers all the surrounding agents by informing the multi-agent policy with the planned state sequence. Fundamental to the success of our method is the design of a novel multi-agent policy network that can steer a vehicle given the state of the surrounding agents and the map information. The policy network is trained implicitly with ground-truth observation data using backpropagation through time and a differentiable dynamics model to roll out the trajectory forward in time. Finally, we show that our multi-agent policy network learns to drive while interacting with the environment, and, when combined with the game-theoretic MPC planner, can successfully generate interactive behaviors.
</div>
    
</td>
</tr>

<table class="table table-hover">
<tr>
<td class="col-sm-2" width="40%"><a href='https://arxiv.org/abs/2003.04882' target='_blank'><img src="/images/iros_animation.gif"/></a> </td>
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
<td class="col-sm-2" width="40%"><a href='{{site.url}}/files/3dv_report.pdf' target='_blank'><img src="/images/projects/3dv_project.jpg"/></a> </td>
<td>
    <strong> Deep Dense Bundle Adjustment Networks</strong><br>
    3D Vision<br>
    
[<a href='javascript:;'
    onclick='$("#abs_3dv_project").toggle()'>description</a>] [<a href='{{site.url}}/files/3dv_report.pdf' target='_blank'>report</a>] <br>
    
<div id="abs_3dv_project" style="text-align: justify; display: none" markdown="1">
This project focuses on developing a neural network architecture, which has embedded a non-linear least squares (NL-LS) optimization problem in its core. This means that gradients are being back-propagated from the output of this NL-LS problem to its input. The architecture is based on the work from Tang, et al. and it is extended using ideas from Lv et al. The main focus of this work is to provide a working, trainable implementation of a differentiable bundle adjustment layer. Extensions to Tang, et al. are suggested to improve pose estimation accuracy via modifications in the CNN based feature network, the damping factor estimation layer, and an additional subnetwork for the camera pose initialization
</div>

</td>
</tr>

<tr class="row">
<td class="col-sm-2" width="40%"><a href='{{site.url}}/files/plr_report.pdf' target='_blank'><img src="/images/projects/learning_sampling_half.png"/></a> </td>
<td>
    <strong>Generative Models for Sampling Based Motion Planning on Distance Fields</strong><br>
    Perception and Learning for Robotics<br>
    
[<a href='javascript:;'
    onclick='$("#abs_plr_project").toggle()'>description</a>] [<a href='{{site.url}}/files/plr_report.pdf' target='_blank'>report</a>] <br>
    
<div id="abs_plr_project" style="text-align: justify; display: none" markdown="1">
Sampling based motion planners (SMBP’s) suffer from the use of uninformative sampling distributions that rely on heuristics designed by roboticists to be effective. Now, with recent work in deep generative models, complex and high-dimensional sampling distributions can be learned in an unsupervised fashion from data generated by the target distribution. We demonstrate the marriage of the two concepts: using a generative model as a sampling distribution for an SBMP. We implement an open source pipeline for training and inference based on standard open source motion planning tools that is capable of learning from previous plans to improve the results on new planning problems. Furthermore we extended the approach to incorporated learning distributions from unstructured conditional data.
</div>

</td>
</tr>

<tr class="row">
<td class="col-sm-2" width="40%"><a href='{{site.url}}/files/deep_learning_report.pdf' target='_blank'><img src="/images/projects/cartpole.png"/></a> </td>
<td>
    <strong>Assessing Generalization in Probabilistic Ensembles for Model-Based Deep Reinforcement Learning</strong><br>
    Deep Learning<br>
    
[<a href='javascript:;'
    onclick='$("#abs_dl_project").toggle()'>description</a>] [<a href='{{site.url}}/files/deep_learning_report.pdf' target='_blank'>report</a>] <br>
    
<div id="abs_dl_project" style="text-align: justify; display: none" markdown="1">
Generalization in reinforcement learning refers to an agent’s ability to perform outside of the environment it was trained in. Reinforcement learning (RL) algorithms are typically both trained and tested on fixed environments, which can result in over-fitting to the system under analysis. Since model-based RL (MBRL) methods attempt to learn the underlying system dynamics, they may perform better in generalization tasks compared to model-free methods, which directly learn a policy. Probabilistic Ensembles with Trajectory Sampling (PETS) [1] is a MBRL algorithm that is both sample efficient and high-performing on standard benchmark tasks. In this paper, we study how well PETS is able to generalize outside of the parameters it was trained on using a recently proposed generalization standard and modifiable gym environments from [2]. We find that PETS performs lower on these benchmarks compared to current state of the art model-free methods, and there is no apparent benefit of a standard model-based algorithm in terms of performance under model mismatch.
</div>

</td>
</tr>

<tr>
<td class="col-sm-2"><a href='https://www.youtube.com/watch?v=dyNT3g425sU&feature=youtu.be' target='_blank'><img src="/images/projects/vision.jpg"/></a> </td>
<td>
    <strong> Sparse Monocular Visual Odometry Pipeline </strong><br>
    Vision Algorithm for Mobile Robotics<br>
    
[<a href='javascript:;'
    onclick='$("#abs_vision").toggle()'>description</a>] [<a href='{{site.url}}/files/vision_report.pdf' target='_blank'>report</a>] [<a href='https://www.youtube.com/watch?v=dyNT3g425sU&feature=youtu.be' target='_blank'>video</a>]<br>
    
<div id="abs_vision" style="text-align: justify; display: none" markdown="1">
Final project for the Vision Algorithms for Mobile Robotics at ETHZ, in this project we developed traditional sparese visual odometry pipeline. Traditional computer vision techniques were used for the essential matrix estimation and the estimation of the posses between frames with the tracked keypoints.
</div>

</td>
</tr>

</table>



