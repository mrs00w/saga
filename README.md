# 🧠 SAGA: Situation-Aware Generative Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SAGA (Situation-Aware Generative Agent)** is an agentic decision-support system designed to arbitrate and prioritize conflicting social calendar events. Built upon social psychology foundations and modern agentic architectures, SAGA translates unstructured natural language descriptions into structured psychological dimensions to provide explainable priorities and adaptive negotiation.

---

## 📄 Project Paper & Presentation

This project was developed and evaluated as an extension of research in Social Situation Comprehension (SSC). Below, you can read the paper I wrote for this project, as well as a presentation with slides :
* **Research Paper**: [`A_Situation_Aware_Generative_Agent_Based_on_Psychological_Characteristics_of_Situation.pdf`](./A_Situation_Aware_Generative_Agent_Based_on_Psychological_Characteristics_of_Situation.pdf)
* **Presentation Slides**: [`Presentation_LEDDA_Romain_Using-Psychological-Characteristics-of-Situations-for-Social-Situation-Comprehension-in-Support-Agent.pdf`](./Presentation_LEDDA_Romain_Using-Psychological-Characteristics-of-Situations-for-Social-Situation-Comprehension-in-Support-Agent.pdf)

---

## 📌 Theoretical Framework

SAGA implements a three-level situation awareness framework:

1. **Level 1 : Perception**: Objective situation cues are extracted from natural language descriptions.
2. **Level 2 : Comprehension**: Situations are mapped into the **DIAMONDS** taxonomy (scored 1–7):
   * **Duty** (obligations), **Intellect** (cognitive demand), **Adversity** (conflict), **Mating** (romantic potential), **Positivity** (enjoyment), **Negativity** (stress), **Deception** (distrust), **Sociality** (relationships).
3. **Level 3 : Projection**: A pre-trained **Random Forest Regressor** estimates the social priority (1–7) based on the extracted profile.

---

## 🏗️ Architecture

The agent pipeline is orchestrated using **LangGraph** and deployed on a **Streamlit** interface.

* **Perception Node**: LLMs handle unstructured text parsing into strict JSON schemas (DIAMONDS scores).
* **Reasoning Node**: A classical ML Random Forest Regressor calculates deterministic priority modeling based on the JSON output.
* **Explanation Node**: Generates transparent, human-readable rationales based on dominant psychological dimensions.
* **Negotiation Node**: A state-machine-driven loop that dynamically counter-argues if the user contests the agent's decision.

## 🚀 Installation & Quickstart

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed locally for local LLM inference.
* *(Optional)* Groq or Google Gemini API key for cloud inference.

### 2. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/mrs00w/saga.git
cd saga
pip install -r requirements.txt
```

Pull the required local models via Ollama:

```bash
ollama pull llama3.2:3b
ollama pull gemma2:2b
```

### 3. Run SAGA

Launch the Streamlit interactive dashboard:

```bash
streamlit run app.py
```

---

## 📚 References

* Kola, I., Jonker, C. M., & van Riemsdijk, M. B. (2023). *Using psychological characteristics of situations for social situation comprehension in support agents*. Autonomous Agents and Multi-Agent Systems (AAMAS).
* Rauthmann, J. F., et al. (2014). *The Situational Eight DIAMONDS: A taxonomy of major dimensions of situation characteristics*. Journal of Personality and Social Psychology.

---

## 📄 License

Distributed under the Apache 2.0. See `LICENSE` for details.
