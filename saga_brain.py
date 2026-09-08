import pandas as pd
import json
from langchain_groq import ChatGroq
import joblib
import ollama
import time

try:
    rf_priority_model = joblib.load('rf_priority_model.pkl')
except:
    rf_priority_model = None
    print("Modèle rf_priority_model.pkl non trouvé. Le mode ML de priorité sera indisponible.")
    
PROMPT_PCS_EXTRACTION = """Role : 
You are a social psychology expert specialized in Psychological Characteristics of Situations. Your task is to analyze social situations using the DIAMONDS taxonomy.

Task : Analyze the provided situation and rate it on 8 dimensions using a Likert scale from 1 (Very Uncharacteristic) to 7 (Very Characteristic) :
- Duty : situations where a job has to be done, minor details are important, and rational thinking is called for
- Intellect : situations that afford an opportunity to demonstrate intellectual capacity
- Adversity : situations where you or someone else are (potentially) being criticized, blamed, or under threat
- Mating : situations where potential romantic partners are present, and physical attractiveness is relevant
- Positivity : playful and enjoyable situations, which are simple and clear‐cut
- Negativity : stressful, frustrating, and anxiety‐inducing situations
- Deception : situations where someone might be deceitful. These situations may cause feelings of hostility
- Sociality : situations where social interaction is possible, and close personal relationships are present or have the potential to develop.

Output rules : 
1. Provide ONLY a valid JSON object.
2. No conversational text, no explanations.
3. Use exactly these keys: "Duty", "Intellect", "Adversity", "Mating", "Positivity", "Negativity", "Deception", "Sociality".
4. Ensure scores are integers."""

GROQ_API = r"YOUR API"

def call_ollama(model_name, prompt):
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        format='json',
        stream=False
    )
    return json.loads(response['response'])

def pcs_extraction(description, api_key=GROQ_API, model_name="llama-3.3-70b-versatile", provider="groq", row_data=None, system_prompt=PROMPT_PCS_EXTRACTION, mode="LLM"):
    diamonds = ['Adversity', 'Deception', 'Duty', 'Intellect', 'Mating', 'Negativity', 'Positivity', 'sociality']

    if mode == "ML" and row_data is not None:
        return {col: row_data[col] for col in diamonds if col in row_data}
    else:
        prompt = f"{system_prompt}\n\nSituation: {description}"
        
        if provider == "groq":
            max_retries = 3
            for i in range(max_retries):
                try:
                    llm = ChatGroq(model_name=model_name, groq_api_key=api_key)
                    response = llm.invoke(prompt)
                    return json.loads(response.content)
                except Exception as e:
                    if "429" in str(e): # Too Many Requests
                        wait_time = (i + 1) * 10
                        print(f"Rate limit reached. Waiting for {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise e
        else:
            return call_ollama(model_name, prompt)

def predict_priority(pcs_scores, mode="ML", model_name="llama3.2", provider="ollama", api_key=GROQ_API):
    diamonds = ['Adversity', 'Deception', 'Duty', 'Intellect', 'Mating', 'Negativity', 'Positivity', 'sociality']
    
    if mode == "ML" and rf_priority_model is not None:
        if 'Sociality' in pcs_scores:
            pcs_scores['sociality'] = pcs_scores.pop('Sociality')
            
        df_input = pd.DataFrame([pcs_scores])
        df_input = df_input[diamonds]
        
        return float(rf_priority_model.predict(df_input)[0])
    
    else:        
        priority_prompt=f"""Role :
        You are a strategic personal assistant. Your task is to determine the urgency and importance (Priority) of a social situation based on its psychological profile.
        
        Input data : 
        The situation has been rated from 1 to 7 on the following dimensions:
        {json.dumps(pcs_scores)}
        
        Decision Logic :
        When calculating the priority (1-7), consider the typical social impact of these dimensions:
        - High Duty or Adversity usually indicates High Priority.
        - High Negativity combined with Deception often requires urgent attention.
        - High Positivity or Sociality might be important for well-being but can be lower priority if Duty is low.
        - Mating and Intellect are situational and depend on personal goals.
        - Human usually prefer to meet their relatives unless Duty is high.
        
        Output rules :
        1. Provide ONLY a valid JSON object.
        2. The "priority" value must be a float between 1.0 and 7.0.
        3. No conversational text or reasoning explanation.
        4. Format: {{"priority": float}}
        
        Final task :
        Based on the scores provided, what is the priority of this meeting?"""
        
        if provider == "groq":
            llm = ChatGroq(temperature=0, model_name=model_name, groq_api_key=api_key)
            res = llm.invoke(priority_prompt)
            return float(json.loads(res.content)['priority'])
        else:
            res = call_ollama(model_name, priority_prompt)
            return float(res['priority']) if res else 4.0