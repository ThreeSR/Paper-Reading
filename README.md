# Paper-Reading
This Repo is used to collect and record papers. The topic includes vision and language multimodal learning, natural language processing, and computer vision. There are a large number of papers published on ArXiv every day. Let's keep a record of papers that may inspire some new ideas and closely follow AI development.

# Time-Order
Time order is based on the release date on ArXiv.
[Notion Page - Paper to be read](https://rui-sun.notion.site/458332784d254a1d9b1990c911befa24?pvs=4)

# Topic-Order
## Language Agent

Some scenarios: Online Shopping, OS, API, VLN, Travel Planning, Tool Use

Online Shopping

Mind2Web: Towards a Generalist Agent for the Web

SeeAct GPT-4V(ision) is a Generalist Web Agent, if Grounded

WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents

WebArena

VisualWebArena

## LLM
Prerequisite: T5 ...

Teaching Language Models to Self-Improve through Interactive Demonstrations

Large Language Models as Tool Makers

Toolformer: Language Models Can Teach Themselves to Use Tools

IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models

ViCor: Bridging Visual Understanding and Commonsense Reasoning with Large Language Models

CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets

Avalon's Game of Thoughts: Battle Against Deception through Recursive Contemplation (interesting)

An Emulator for Fine-Tuning Large Language Models using Small Language Models

Why Can Large Language Models Generate Correct Chain-of-Thoughts?

### Code generation
CodeFusion: A Pre-trained Diffusion Model for Code Generation

Symbolic Planning and Code Generation for Grounded Dialogue

Large Language Model-Aware In-Context Learning for Code Generation

Code-Style In-Context Learning for Knowledge-Based Question Answering

## VLM
Prerequisite: ViLBERT VisualBERT VL-BERT Oscar UNITER CLIP Flamingo ...

### Foundation Models
Kosmos-G: Generating Images in Context with Multimodal Large Language Models

**The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)**

A Survey on Image-text Multimodal Models



NExT-GPT: Any-to-Any Multimodal LLM (MM-Dialog system, _interleaved_ multimodal information)

NExT-Chat: An LMM for Chat, Detection and Segmentation

MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens

Grounding Language Models to Images for Multimodal Inputs and Outputs (ICML 2023)

Generating Images with Multimodal Language Models (NIPS 2023)

DreamLLM

NExT-GPT: Any-to-Any Multimodal LLM

SEED: Planting a SEED of Vision in Large Language Model

OPENLEAF: OPEN-DOMAIN INTERLEAVED IMAGETEXT GENERATION AND EVALUATION

Generative Pretraining in Multimodality (Emu)

Mini-DALL•E 3: Interactive Text to Image by Prompting Large Language Models



CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding

Otter

OtterHD: A High-Resolution Multi-modality Model

mPLUG-Owl

mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration



### Text-to-Image
DiagrammerGPT: Generating Open-Domain, Open-Platform Diagrams via LLM Planning (diagram generation, text-to-image)

### Visual Grounding
Ferret: Refer and Ground Anything Anywhere at Any Granularity

### Image Editing
Guiding Instruction-based Image Editing via Multimodal Large Language Models

### Visual Prompting
**Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V**

What does CLIP know about a red circle? Visual prompt engineering for VLMs

Visual Cropping Improves Zero-Shot Question Answering of Multimodal Large Language Models

MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound

### Visual Clasification
Prompting Scientific Names for Zero-Shot Species Recognition (iNat)

## Efficiency
FP8-LM: Training FP8 Large Language Models

LLM-FP4: 4-Bit Floating-Point Quantized Transformers

BitNet: Scaling 1-bit Transformers for Large Language Models

Small Language Models Fine-tuned to Coordinate Larger Language Models improve Complex Reasoning

## Robustness
### Pre-training Dataset Investigation
Detecting Pretraining Data from Large Language Models

Demystifying CLIP Data

Holistic Analysis of Hallucination in GPT-4V(ision): Bias and Interference Challenges

## Bias Mitigation
Dataset Bias Mitigation in Multiple-Choice Visual Question Answering and Beyond

## Hallucination
Woodpecker: Hallucination Correction for Multimodal Large Language Models

A Survey of Hallucination in Large Foundation Models

Towards Mitigating Hallucination in Large Language Models via Self-Reflection



## Evaluation and Benchmark
GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks

Don't Make Your LLM an Evaluation Benchmark Cheater

ROME: Evaluating Pre-trained Vision-Language Models on Reasoning beyond Visual Common Sense

TIGERScore: Towards Building Explainable Metric for All Text Generation Tasks

Improving Automatic VQA Evaluation Using Large Language Models

ALCUNA: Large Language Models Meet New Knowledge （benchmark work）

HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models

Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation (text-to-image)

## Continual Learning
A Study of Continual Learning Under Language Shift

TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models

Understanding the Effects of RLHF on LLM Generalisation and Diversity

Continual Named Entity Recognition without Catastrophic Forgetting

Tree Prompting: Efficient Task Adaptation without Fine-Tuning

An Emulator for Fine-Tuning Large Language Models using Small Language Models

Online Continual Knowledge Learning for Language Models

### Model Editing
Revisiting the Knowledge Injection Frameworks, revisit some previous works and discuss the question that random (unaligned) knowledge can obtain comparable performance as aligned knowledge

BadLlama: cheaply removing safety fine-tuning from Llama 2-Chat 13B, showcase if ckpt is released, user can utilize ckpt to convert the model into previous state. Thus, even though Llama has been fine-tuned to prevent harmful content, user can also restore it in Llama. this paper investigates the conversion of safety fine-tuning.

Unlearn What You Want to Forget: Efficient Unlearning for LLMs, Introduce a new efficient unlearning approach for LLM, code has not been released yet

Who’s Harry Potter? Approximate Unlearning in LLMs

TiC-CLIP: Continual Training of CLIP Models

Knowledge Editing for Large Language Models: A Survey

Can We Edit Multimodal Large Language Models?

Resolving Knowledge Conflicts in Large Language Models (pending)

Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory

Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models

Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks (Black-box and white-box attack)

Do Language Models Have Beliefs? Methods for Detecting, Updating, and Visualizing Model Beliefs

Massive Editing for Large Language Models via Meta Learning

Locating and Editing Factual Associations in GPT [[Realated Work](https://rome.baulab.info/)]

Evaluating the Ripple Effects of Knowledge Editing in Language Models

Finding and Editing Multi-Modal Neurons in Pre-Trained Transformer

Untying the Reversal Curse via Bidirectional Language Model Editing

Model Editing Can Hurt General Abilities of Large Language Models



#### ICL Model Editing

Can We Edit Factual Knowledge by In-Context Learning?

Mquake: Assessing knowledge editing in language models via multi-hop questions

Memory-assisted prompt editing to improve GPT-3 after deployment


## Dataset
### Visual Classification
iNat (Fine-grained Visual Classification)
### Image and Text
Prerequisite: VQA VCR GQA RefCOCO SNLI-VE VQG...


### Video and Text
Prerequisite: TVQA VLEP ...

ACQUIRED: A Dataset for Answering Counterfactual Questions In Real-Life Videos (Video Commonsense Reasoning, Counterfactual Reasoning)

ComPhy (Video Commonsense Reasoning, Physical Reasoning)

Next-QA

STAR (Video Commonsense Reasoning)

## Misc
EmojiLM: Modeling the New Emoji Language (text-to-emoji, interesting)
