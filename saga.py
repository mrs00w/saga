import streamlit as st
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import ollama
import json
from saga_brain import pcs_extraction, predict_priority

class AgentState(TypedDict):
    description: str
    model_name: str
    provider: str
    pcs_scores: Dict[str, int]
    priority: float
    decision: str
    explanation: str
    user_feedback: str
    negotiation_history: List[str]

def perception_node(state: AgentState):
    scores = pcs_extraction(state['description'], model_name=state['model_name'], provider=state["provider"])
    return {"pcs_scores": scores}

def reasoning_node(state: AgentState):
    priority = predict_priority(state['pcs_scores'], mode="ML", model_name=state["model_name"], provider=state["provider"])
    
    decision = "Should attempt this meeting" if priority >= 4.5 else "Should skip this meeting"
    return {"priority": priority, "decision": decision}

def explanation_node(state: AgentState):
    scores = state['pcs_scores']
    max_dim = max(scores, key=scores.get)
    min_dim = min(scores, key=scores.get)
    
    if state['priority'] >= 5.5:
        verdict = "Critical Priority"
    elif state['priority'] >= 4.0:
        verdict = "Moderate Priority"
    else:
        verdict = "Low Priority"

    text = f"{state['decision']} ({verdict})\n"
    text += f"The analysis shows a dominant level of **{max_dim}** ({scores[max_dim]}/7), "
    
    if state['priority'] >= 4.5:
        text += f"which significantly outweighs the lower **{min_dim}** ({scores[min_dim]}/7). "
        text += "This profile suggests that the social or functional cost of missing this meeting is high."
    else:
        text += f"but since the overall urgency is only {state['priority']:.1f}/7, "
        text += "it may not require your immediate attendance."
        
    return {"explanation": text}

def negotiation_node(state: AgentState):    
    last_user_msg = state['user_feedback']
    scores = state['pcs_scores']
    prio = state['priority']
    history = state.get('negotiation_history', [])
    
    prompt = f"""
    You are SAGA, a situation aware generative agent. Your task is to convince the user to go to the meeting you have chosen. 
    
    Context:
    - User refuses a meeting with Priority {prio:.1f}/7.
    - Situational Scores: {json.dumps(scores)}.
    - User's specific objection: "{last_user_msg}".
    - Previous arguments you used: {history}.

    Instructions:
    1. Acknowledge the user's feelings (especially if Negativity is high).
    2. Counter-argue using the highest score (e.g., Duty or Sociality).
    3. If the user mentioned a specific reason, address it directly.
    4. Keep it to 2 sentences max. Be firm but supportive.
    5. Do NOT repeat previous arguments.
    """
    
    try:
        if state.get('provider') == "groq":
            from langchain_groq import ChatGroq
            llm = ChatGroq(model_name=state['model_name'], groq_api_key=state.get('api_key'))
            response = llm.invoke(prompt).content
        else:
            res = ollama.generate(model=state['model_name'], prompt=prompt, stream=False)
            response = res['response']
    except Exception as e:
        response = f"I hear your concern about '{last_user_msg}'. However, the situational pressure of {max(scores, key=scores.get)} is still the primary factor here."

    history.append(response)
    
    return {"negotiation_history": history, "explanation": response}

builder = StateGraph(AgentState)

builder.add_node("perception", perception_node)
builder.add_node("reasoning", reasoning_node)
builder.add_node("explanation", explanation_node)
builder.add_node("negotiation", negotiation_node)

builder.set_entry_point("perception")

builder.add_edge("perception", "reasoning")
builder.add_edge("reasoning", "explanation")
builder.add_edge("explanation", END)

saga_agent = builder.compile()