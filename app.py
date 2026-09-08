import streamlit as st

from saga import saga_agent
from saga import negotiation_node

st.set_page_config(page_title="SAGA", page_icon="🧠")

if "history" not in st.session_state: 
    st.session_state.history = [{"role": "assistant", "content": "Hello! Not sure which meeting to attend ? Tell me about the two meetings you would like to attend and I will advise you which one to prioritise."}]
    
if "step" not in st.session_state: st.session_state.step = "GET_SITUATION_1"
if "states" not in st.session_state: st.session_state.states = []
if "winner_index" not in st.session_state: st.session_state.winner_index = None

is_finished = st.session_state.step == "DONE"

with st.sidebar:
    st.title("🧠 SAGA")
    selected_model = st.selectbox("Modèle", ["llama3.2:3b", "mistral:7b", "gemma2:2b", "gemma2:9b", "phi3:3.8b", "llama-3.3-70b-versatile"])
    selected_provider = st.selectbox("Provider", ["ollama", "groq"])

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "details" in msg:
            with st.expander("LangGraph Analysis"):
                st.json(msg["details"])

if user_input := st.chat_input("Your message...", disabled=is_finished):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if st.session_state.step == "GET_SITUATION_1":
        
        initial_state = {
            "description": user_input,
            "model_name": selected_model,
            "provider": selected_provider,
            "negotiation_history": []
        }
        
        st.session_state.states.append(initial_state)
        
        resp = "Understood. What is the second appointment you are hesitating to attend ?"
        st.session_state.history.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"): st.write(resp)
        st.session_state.step = "GET_SITUATION_2"
        st.rerun()

    elif st.session_state.step == "GET_SITUATION_2":
        initial_state_2 = {
            "description": user_input,
            "model_name": selected_model,
            "provider": selected_provider,
            "negotiation_history": []
        }
        
        st.session_state.states.append(initial_state_2)
        
        with st.chat_message("assistant"):
            with st.status("LangGraph pipeline execution...", expanded=True):
                st.write("🔄 SAGA is thinking about the first meeting...")

                res_s1 = saga_agent.invoke(st.session_state.states[0])
                st.session_state.states[0] = res_s1

                st.write("🔄 SAGA is thinking about the second meeting")
                res_s2 = saga_agent.invoke(st.session_state.states[1])
                st.session_state.states[1] = res_s2

                prio1 = res_s1['priority']
                prio2 = res_s2['priority']

                if prio1 >= prio2:
                    winner_idx = 0
                    loser_idx = 1
                else:
                    winner_idx = 1
                    loser_idx = 0
                
                st.session_state.winner_index = winner_idx
                winner_state = st.session_state.states[winner_idx]
                
                explanation = f"I recommend you go here : **{winner_state['description']}**.\n\n"
                explanation += f"SAGA explication for the first meeting : {res_s1['explanation']}\n"
                explanation += f"SAGA explication for the second meeting : {res_s2['explanation']}"
            
            st.write(explanation)
            st.session_state.history.append({
                "role": "assistant", 
                "content": explanation, 
                "details": {"S1_PCS": res_s1['pcs_scores'], "S2_PCS": res_s2['pcs_scores']}
            })
            
            st.session_state.step = "DECISION"
            st.rerun()
            
    elif st.session_state.step in ["WAITING_FOR_OBJECTION", "NEGOTIATING"]:
        winner_idx = st.session_state.winner_index
        current_state = st.session_state.states[winner_idx]
        current_state['user_feedback'] = user_input
        current_state['model_name'] = selected_model
        
        with st.spinner("SAGA is analysing your objection..."):
            neg_result = negotiation_node(current_state)
            current_state['negotiation_history'] = neg_result['negotiation_history']
            response = neg_result['explanation']
        
        st.session_state.history.append({"role": "assistant", "content": response})
        st.session_state.step = "NEGOTIATING"
        st.rerun()
        
if st.session_state.step == "DECISION":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Accept", use_container_width=True):
            st.session_state.history.append({"role": "assistant", "content": "Excellent choice. I'm delighted to have been of assistance."})
            st.session_state.step = "DONE"
            st.rerun()
    with col2:
        if st.button("❌ Contest", use_container_width=True):
            st.session_state.step = "WAITING_FOR_OBJECTION"
            st.session_state.history.append({"role": "assistant", "content": "I understand. Explain to me why you disagree, and I will try to refine my arguments."})
            st.rerun()

elif st.session_state.step == "NEGOTIATING":
    st.write("**Does this argument change your mind ?**")
    c1, c2 = st.columns(2)
    if c1.button("✅ Finally, I accept", use_container_width=True):
        st.session_state.history.append({"role": "assistant", "content": "Glad to have been able to clarify the situation. See you soon !"})
        st.session_state.step = "DONE"
        st.rerun()
    if c2.button("❌ Still no", use_container_width=True):
        st.session_state.step = "WAITING_FOR_OBJECTION"
        st.info("Continue explaining your point of view in the chat.")
        st.rerun()