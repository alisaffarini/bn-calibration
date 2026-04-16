### Paper 1: Beyond the Final Layer: Intermediate Representations for Better Multilingual Calibration in Large Language Models
**Authors:** Ej Zhou, Caiqi Zhang, Tiancheng Hu et al. (2025)
**URL:** https://arxiv.org/abs/2510.03136v1
**Abstract:** Confidence calibration, the alignment of a model's predicted confidence with its actual accuracy, is crucial for the reliable deployment of Large Language Models (LLMs). However, this critical property remains largely under-explored in multilingual contexts. In this work, we conduct the first large-scale, systematic studies of multilingual calibration across six model families and over 100 languages, revealing that non-English languages suffer from systematically worse calibration. To diagnose t

### Paper 2: Calibration Across Layers: Understanding Calibration Evolution in LLMs
**Authors:** Abhinav Joshi, Areeb Ahmad, Ashutosh Modi (2025)
**URL:** https://arxiv.org/abs/2511.00280v1
**Abstract:** Large Language Models (LLMs) have demonstrated inherent calibration capabilities, where predicted probabilities align well with correctness, despite prior findings that deep neural networks are often overconfident. Recent studies have linked this behavior to specific components in the final layer, such as entropy neurons and the unembedding matrix null space. In this work, we provide a complementary perspective by investigating how calibration evolves throughout the network depth. Analyzing mult

### Paper 3: Transformer^-1: Input-Adaptive Computation for Resource-Constrained Deployment
**Authors:** Lumen AI, Tengzhou No. 1 Middle School, Shihao Ji et al. (2025)
**URL:** https://arxiv.org/abs/2501.16394v1
**Abstract:** Addressing the resource waste caused by fixed computation paradigms in deep learning models under dynamic scenarios, this paper proposes a Transformer$^{-1}$ architecture based on the principle of deep adaptivity. This architecture achieves dynamic matching between input features and computational resources by establishing a joint optimization model for complexity and computation. Our core contributions include: (1) designing a two-layer control mechanism, composed of a complexity predictor and 

### Paper 4: Selecting for Less Discriminatory Algorithms: A Relational Search Framework for Navigating Fairness-Accuracy Trade-offs in Practice
**Authors:** Hana Samad, Michael Akinwumi, Jameel Khan et al. (2025)
**URL:** https://arxiv.org/abs/2506.01594v2
**Abstract:** As machine learning models are increasingly embedded into society through high-stakes decision-making, selecting the right algorithm for a given task, audience, and sector presents a critical challenge, particularly in the context of fairness. Traditional assessments of model fairness have often framed fairness as an objective mathematical property, treating model selection as an optimization problem under idealized informational conditions. This overlooks model multiplicity as a consideration--

### Paper 5: Group Equivariance Meets Mechanistic Interpretability: Equivariant Sparse Autoencoders
**Authors:** Ege Erdogan, Ana Lucic (2025)
**URL:** https://arxiv.org/abs/2511.09432v1
**Abstract:** Sparse autoencoders (SAEs) have proven useful in disentangling the opaque activations of neural networks, primarily large language models, into sets of interpretable features. However, adapting them to domains beyond language, such as scientific data with group symmetries, introduces challenges that can hinder their effectiveness. We show that incorporating such group symmetries into the SAEs yields features more useful in downstream tasks. More specifically, we train autoencoders on synthetic i

### Paper 6: nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers
**Authors:** Clément Dumas (2025)
**URL:** https://arxiv.org/abs/2511.14465v2
**Abstract:** Mechanistic interpretability research requires reliable tools for analyzing transformer internals across diverse architectures. Current approaches face a fundamental tradeoff: custom implementations like TransformerLens ensure consistent interfaces but require coding a manual adaptation for each architecture, introducing numerical mismatch with the original models, while direct HuggingFace access through NNsight preserves exact behavior but lacks standardization across models. To bridge this gap

### Paper 7: Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video
**Authors:** Sonia Joseph, Praneet Suresh, Lorenz Hufe et al. (2025)
**URL:** https://arxiv.org/abs/2504.19475v3
**Abstract:** Robust tooling and publicly available pre-trained models have helped drive recent advances in mechanistic interpretability for language models. However, similar progress in vision mechanistic interpretability has been hindered by the lack of accessible frameworks and pre-trained weights. We present Prisma (Access the codebase here: https://github.com/Prisma-Multimodal/ViT-Prisma), an open-source framework designed to accelerate vision mechanistic interpretability research, providing a unified to

### Paper 8: Predicting concentration levels of air pollutants by transfer learning and recurrent neural network
**Authors:** Iat Hang Fong, Tengyue Li, Simon Fong et al. (2025)
**URL:** https://arxiv.org/abs/2502.01654v1
**Abstract:** Air pollution (AP) poses a great threat to human health, and people are paying more attention than ever to its prediction. Accurate prediction of AP helps people to plan for their outdoor activities and aids protecting human health. In this paper, long-short term memory (LSTM) recurrent neural networks (RNNs) have been used to predict the future concentration of air pollutants (APS) in Macau. Additionally, meteorological data and data on the concentration of APS have been utilized. Moreover, in 

### Paper 9: Towards Understanding Dual BN In Hybrid Adversarial Training
**Authors:** Chenshuang Zhang, Chaoning Zhang, Kang Zhang et al. (2024)
**URL:** https://arxiv.org/abs/2403.19150v1
**Abstract:** There is a growing concern about applying batch normalization (BN) in adversarial training (AT), especially when the model is trained on both adversarial samples and clean samples (termed Hybrid-AT). With the assumption that adversarial and clean samples are from two different domains, a common practice in prior works is to adopt Dual BN, where BN and BN are used for adversarial and clean branches, respectively. A popular belief for motivating Dual BN is that estimating normalization statistics 

### Paper 10: Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees
**Authors:** Xin Yu, Zelin He, Ying Sun et al. (2024)
**URL:** https://arxiv.org/abs/2410.08934v4
**Abstract:** Personalized federated learning (PFL) offers a flexible framework for aggregating information across distributed clients with heterogeneous data. This work considers a personalized federated learning setting that simultaneously learns global and local models. While purely local training has no communication cost, collaborative learning among the clients can leverage shared knowledge to improve statistical accuracy, presenting an accuracy-communication trade-off in personalized federated learning

### Paper 11: Challenges in Mechanistically Interpreting Model Representations
**Authors:** Satvik Golechha, James Dao (2024)
**URL:** https://arxiv.org/abs/2402.03855v2
**Abstract:** Mechanistic interpretability (MI) aims to understand AI models by reverse-engineering the exact algorithms neural networks learn. Most works in MI so far have studied behaviors and capabilities that are trivial and token-aligned. However, most capabilities important for safety and trust are not that trivial, which advocates for the study of hidden representations inside these networks as the unit of analysis. We formalize representations for features and behaviors, highlight their importance and

### Paper 12: Mechanistic Neural Networks for Scientific Machine Learning
**Authors:** Adeel Pervez, Francesco Locatello, Efstratios Gavves (2024)
**URL:** https://arxiv.org/abs/2402.13077v1
**Abstract:** This paper presents Mechanistic Neural Networks, a neural network design for machine learning applications in the sciences. It incorporates a new Mechanistic Block in standard architectures to explicitly learn governing differential equations as representations, revealing the underlying dynamics of data and enhancing interpretability and efficiency in data modeling. Central to our approach is a novel Relaxed Linear Programming Solver (NeuRLP) inspired by a technique that reduces solving linear O

### Paper 13: NTU-NPU System for Voice Privacy 2024 Challenge
**Authors:** Nikita Kuzmin, Hieu-Thi Luong, Jixun Yao et al. (2024)
**URL:** https://arxiv.org/abs/2410.02371v1
**Abstract:** In this work, we describe our submissions for the Voice Privacy Challenge 2024. Rather than proposing a novel speech anonymization system, we enhance the provided baselines to meet all required conditions and improve evaluated metrics. Specifically, we implement emotion embedding and experiment with WavLM and ECAPA2 speaker embedders for the B3 baseline. Additionally, we compare different speaker and prosody anonymization techniques. Furthermore, we introduce Mean Reversion F0 for B5, which help

### Paper 14: Atmospheric entry and fragmentation of small asteroid 2024 BX1: Bolide trajectory, orbit, dynamics, light curve, and spectrum
**Authors:** P. Spurny, J. Borovicka, L. Shrbeny et al. (2024)
**URL:** https://arxiv.org/abs/2403.00634v2
**Abstract:** Asteroid 2024 BX1 was the eighth asteroid that was discovered shortly before colliding with the Earth. The associated bolide was recorded by dedicated instruments of the European Fireball Network and the AllSky7 network on 2024 January 21 at 0:32:38-44 UT. We report a comprehensive analysis of this instrumentally observed meteorite fall, which occurred as predicted west of Berlin, Germany. The atmospheric trajectory was quite steep, with an average slope to the Earth's surface of 75.6 deg. The e

### Paper 15: Discovery Opportunities with Gravitational Waves -- TASI 2024 Lecture Notes
**Authors:** Valerie Domcke (2024)
**URL:** https://arxiv.org/abs/2409.08956v1
**Abstract:** Recent advancements in gravitational wave astronomy hold the promise of a completely new way to explore our Universe. These lecture notes aim to provide a concise but self-contained introduction to key concepts of gravitational wave physics, with a focus on the opportunities to explore fundamental physics in transient gravitational wave signals and stochastic gravitational wave background searches.CERN-TH-2024-152

### Paper 16: Double Multi-Head Attention Multimodal System for Odyssey 2024 Speech Emotion Recognition Challenge
**Authors:** Federico Costa, Miquel India, Javier Hernando (2024)
**URL:** https://arxiv.org/abs/2406.10598v1
**Abstract:** As computer-based applications are becoming more integrated into our daily lives, the importance of Speech Emotion Recognition (SER) has increased significantly. Promoting research with innovative approaches in SER, the Odyssey 2024 Speech Emotion Recognition Challenge was organized as part of the Odyssey 2024 Speaker and Language Recognition Workshop. In this paper we describe the Double Multi-Head Attention Multimodal System developed for this challenge. Pre-trained self-supervised models were

### Paper 17: ICAGC 2024: Inspirational and Convincing Audio Generation Challenge 2024
**Authors:** Ruibo Fu, Rui Liu, Chunyu Qiang et al. (2024)
**URL:** https://arxiv.org/abs/2407.12038v2
**Abstract:** The Inspirational and Convincing Audio Generation Challenge 2024 (ICAGC 2024) is part of the ISCSLP 2024 Competitions and Challenges track. While current text-to-speech (TTS) technology can generate high-quality audio, its ability to convey complex emotions and controlled detail content remains limited. This constraint leads to a discrepancy between the generated audio and human subjective perception in practical applications like companion robots for children and marketing bots. The core issue 

### Paper 18: Uncovering Coordinated Cross-Platform Information Operations Threatening the Integrity of the 2024 U.S. Presidential Election Online Discussion
**Authors:** Marco Minici, Luca Luceri, Federico Cinus et al. (2024)
**URL:** https://arxiv.org/abs/2409.15402v2
**Abstract:** Information Operations (IOs) pose a significant threat to the integrity of democratic processes, with the potential to influence election-related online discourse. In anticipation of the 2024 U.S. presidential election, we present a study aimed at uncovering the digital traces of coordinated IOs on $\mathbb{X}$ (formerly Twitter). Using our machine learning framework for detecting online coordination, we analyze a dataset comprising election-related conversations on $\mathbb{X}$ from May 2024. T

### Paper 19: Overview of the 2024 ALTA Shared Task: Detect Automatic AI-Generated Sentences for Human-AI Hybrid Articles
**Authors:** Diego Mollá, Qiongkai Xu, Zijie Zeng et al. (2024)
**URL:** https://arxiv.org/abs/2412.17848v1
**Abstract:** The ALTA shared tasks have been running annually since 2010. In 2024, the purpose of the task is to detect machine-generated text in a hybrid setting where the text may contain portions of human text and portions machine-generated. In this paper, we present the task, the evaluation criteria, and the results of the systems participating in the shared task.

### Paper 20: OpenFact at CheckThat! 2024: Combining Multiple Attack Methods for Effective Adversarial Text Generation
**Authors:** Włodzimierz Lewoniewski, Piotr Stolarski, Milena Stróżyna et al. (2024)
**URL:** https://arxiv.org/abs/2409.02649v2
**Abstract:** This paper presents the experiments and results for the CheckThat! Lab at CLEF 2024 Task 6: Robustness of Credibility Assessment with Adversarial Examples (InCrediblAE). The primary objective of this task was to generate adversarial examples in five problem domains in order to evaluate the robustness of widely used text classification methods (fine-tuned BERT, BiLSTM, and RoBERTa) when applied to credibility assessment issues.   This study explores the application of ensemble learning to enhance

### Paper 21: Proceedings of 6th International Conference AsiaHaptics 2024
**Authors:** Yasutoshi Makino, Hsin-Ni Ho, Seokhee Jeon (2024)
**URL:** https://arxiv.org/abs/2411.08318v1
**Abstract:** The sixth international conference AsiaHaptics 2024 took place at Sunway University, Malaysia on 28-30 October 2024. AsiaHaptics is an exhibition type of international conference dedicated to the haptics domain, engaging presentations accompanied by hands-on demonstrations. It presents the state-of-the-art of the diverse haptics (touch)-related research, including perception and illusion, development of haptics devices, and applications to a wide variety of fields such as education, medicine, te

### Paper 22: On The Lunar Origin of Near-Earth Asteroid 2024 PT5
**Authors:** Theodore Kareta, Oscar Fuentes-Muñoz, Nicholas Moskovitz et al. (2024)
**URL:** https://arxiv.org/abs/2412.10264v1
**Abstract:** The Near-Earth Asteroid (NEA) 2024 PT5 is on an Earth-like orbit which remained in Earth's immediate vicinity for several months at the end of 2024. PT5's orbit is challenging to populate with asteroids originating from the Main Belt and is more commonly associated with rocket bodies mistakenly identified as natural objects or with debris ejected from impacts on the Moon. We obtained visible and near-infrared reflectance spectra of PT5 with the Lowell Discovery Telescope and NASA Infrared Telesc

### Paper 23: Team HYU ASML ROBOVOX SP Cup 2024 System Description
**Authors:** Jeong-Hwan Choi, Gaeun Kim, Hee-Jae Lee et al. (2024)
**URL:** https://arxiv.org/abs/2407.11365v1
**Abstract:** This report describes the submission of HYU ASML team to the IEEE Signal Processing Cup 2024 (SP Cup 2024). This challenge, titled "ROBOVOX: Far-Field Speaker Recognition by a Mobile Robot," focuses on speaker recognition using a mobile robot in noisy and reverberant conditions. Our solution combines the result of deep residual neural networks and time-delay neural network-based speaker embedding models. These models were trained on a diverse dataset that includes French speech. To account for t

### Paper 24: NPU-NTU System for Voice Privacy 2024 Challenge
**Authors:** Jixun Yao, Nikita Kuzmin, Qing Wang et al. (2024)
**URL:** https://arxiv.org/abs/2409.04173v2
**Abstract:** Speaker anonymization is an effective privacy protection solution that conceals the speaker's identity while preserving the linguistic content and paralinguistic information of the original speech. To establish a fair benchmark and facilitate comparison of speaker anonymization systems, the VoicePrivacy Challenge (VPC) was held in 2020 and 2022, with a new edition planned for 2024. In this paper, we describe our proposed speaker anonymization system for VPC 2024. Our system employs a disentangle

### Paper 25: The First Place Solution of WSDM Cup 2024: Leveraging Large Language Models for Conversational Multi-Doc QA
**Authors:** Yiming Li, Zhao Zhang (2024)
**URL:** https://arxiv.org/abs/2402.18385v1
**Abstract:** Conversational multi-doc question answering aims to answer specific questions based on the retrieved documents as well as the contextual conversations. In this paper, we introduce our winning approach for the "Conversational Multi-Doc QA" challenge in WSDM Cup 2024, which exploits the superior natural language understanding and generation capability of Large Language Models (LLMs). We first adapt LLMs to the task, then devise a hybrid training strategy to make the most of in-domain unlabeled dat

### Paper 26: Lisbon Computational Linguists at SemEval-2024 Task 2: Using A Mistral 7B Model and Data Augmentation
**Authors:** Artur Guimarães, Bruno Martins, João Magalhães (2024)
**URL:** https://arxiv.org/abs/2408.03127v1
**Abstract:** This paper describes our approach to the SemEval-2024 safe biomedical Natural Language Inference for Clinical Trials (NLI4CT) task, which concerns classifying statements about Clinical Trial Reports (CTRs). We explored the capabilities of Mistral-7B, a generalist open-source Large Language Model (LLM). We developed a prompt for the NLI4CT task, and fine-tuned a quantized version of the model using an augmented version of the training dataset. The experimental results show that this approach can 

### Paper 27: LLMs4OL 2024 Overview: The 1st Large Language Models for Ontology Learning Challenge
**Authors:** Hamed Babaei Giglou, Jennifer D'Souza, Sören Auer (2024)
**URL:** https://arxiv.org/abs/2409.10146v1
**Abstract:** This paper outlines the LLMs4OL 2024, the first edition of the Large Language Models for Ontology Learning Challenge. LLMs4OL is a community development initiative collocated with the 23rd International Semantic Web Conference (ISWC) to explore the potential of Large Language Models (LLMs) in Ontology Learning (OL), a vital process for enhancing the web with structured knowledge to improve interoperability. By leveraging LLMs, the challenge aims to advance understanding and innovation in OL, ali

### Paper 28: Multichannel Orthogonal Transform-Based Perceptron Layers for Efficient ResNets
**Authors:** Hongyi Pan, Emadeldeen Hamdan, Xin Zhu et al. (2023)
**URL:** https://arxiv.org/abs/2303.06797v3
**Abstract:** In this paper, we propose a set of transform-based neural network layers as an alternative to the $3\times3$ Conv2D layers in Convolutional Neural Networks (CNNs). The proposed layers can be implemented based on orthogonal transforms such as the Discrete Cosine Transform (DCT), Hadamard transform (HT), and biorthogonal Block Wavelet Transform (BWT). Furthermore, by taking advantage of the convolution theorems, convolutional filtering operations are performed in the transform domain using element

### Paper 29: Comparison between layer-to-layer network training and conventional network training using Deep Convolutional Neural Networks
**Authors:** Kiran Kumar Ashish Bhyravabhottla, WonSook Lee (2023)
**URL:** https://arxiv.org/abs/2303.15245v2
**Abstract:** Title: Comparison between layer-to-layer network training and conventional network training using Deep Convolutional Neural Networks   Abstract: Convolutional neural networks (CNNs) are widely used in various applications due to their effectiveness in extracting features from data. However, the performance of a CNN heavily depends on its architecture and training process. In this study, we propose a layer-to-layer training method and compare its performance with the conventional training method.

### Paper 30: The Deep Arbitrary Polynomial Chaos Neural Network or how Deep Artificial Neural Networks could benefit from Data-Driven Homogeneous Chaos Theory
**Authors:** Sergey Oladyshkin, Timothy Praditia, Ilja Kröker et al. (2023)
**URL:** https://arxiv.org/abs/2306.14753v1
**Abstract:** Artificial Intelligence and Machine learning have been widely used in various fields of mathematical computing, physical modeling, computational science, communication science, and stochastic analysis. Approaches based on Deep Artificial Neural Networks (DANN) are very popular in our days. Depending on the learning task, the exact form of DANNs is determined via their multi-layer architecture, activation functions and the so-called loss function. However, for a majority of deep learning approach
