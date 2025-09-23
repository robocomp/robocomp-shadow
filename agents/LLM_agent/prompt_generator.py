import streamlit as st
import json
from openai import OpenAI

st.title("Generador de Prompt para el Robot Seguidor con API")

# Inputs
name = st.text_input("Nombre de la persona", "Alice")
distance = st.slider("Distancia (m)", 0.0, 5.0, 1.0, 0.1)
angle = st.slider("Ángulo (rad)", -3.14, 3.14, 0.0, 0.1)
orientation = st.slider("Orientación persona (rad)", -3.14, 3.14, 3.0, 0.1)
linear_speed = st.slider("Velocidad lineal (m/s)", 0.0, 2.0, 0.5, 0.05)
angular_speed = st.slider("Velocidad angular (rad/s)", -1.0, 1.0, 0.0, 0.05)
time_since_last = st.slider("Tiempo desde último mensaje (s)", 0.0, 10.0, 1.0, 0.1)
time_since_start = st.slider("Tiempo desde inicio de misión (s)", 0.0, 3600.0, 60.0, 1.0)

# Build STATE block
state_block = f"""[STATE]
- name: {name}
- distance: {distance:.2f}
- angle: {angle:.2f}
- person_orientation: {orientation:.2f}
- robot_speed: linear: {linear_speed:.2f}, angular: {angular_speed:.2f}
- time_since_last_message: {time_since_last:.2f}
- time_since_start: {time_since_start:.2f}
[/STATE]"""

# Build full messages list for OpenAI API
developer_prompt_ = """
        You are a robot following a person. You will get updates in this format:
        [STATE]
        - name: string // the name of the person
        - distance: float (m) // over 2.5 moreorless means too far
        - angle: float (rad) // from -pi to pi, over ±1.5 moreorless means beign out of FOV
        - person_orientation: float (rad) // from -pi to pi, close to 3 or -3 means the person is facing you. You could ask for interacting in this case.
        - robot_speed: linear: float (m/s), angular: float (rad/s) // close to 0 means not moving
        - time_since_last_message: float (s)
        - time_since_start: float (s)
        [/STATE]
        Your job:
        1. Analyze the states.
        2. Decide if you need to say something. If yes, pick one short, clear, empathetic phrase (no questions unless the person is facing you). Otherwise leave it blank.
        3. Choose how many seconds until your next message.
        Very important!! Your output must be **exactly** one JSON dict, parseable by `json.loads`, with exactly these keys:
        - `"response"`: string (the phrase, or `""` if nothing to say)
        - `"new_sending_frequency"`: float (seconds)
        Examples (you can use intermediate values and doesn't necessarily have to be these phrases):
        - If they’re too far: `"response":"Please slow down so I can keep up.","new_sending_frequency":5.0`
        - If all good: `"response":"","new_sending_frequency":1.0`
        Don’t repeat past messages. Speak only when needed.  
        If a JSON dict is not returned, an advice will be given and you need to correct it at the next message.
        Very important!! Your output must be **exactly** one JSON dict, parseable by `json.loads`, no more text than JSON dict, no markdown, no code blocks, no explanations, no extra text.
        """
developer_prompt = [{"role": "system", "content": developer_prompt_}]
model = "google/gemma-3n-e4b"
client = OpenAI(base_url="http://192.168.50.19:3000/v1", api_key="lm-studio")
user_message = [{
    "role": "user",
    "content": state_block + "\n\nPrevious messages: []"
}]

messages = developer_prompt+ user_message

# Display
st.subheader("Bloque [STATE]")
st.code(state_block)

st.subheader("Ejemplo de llamada API (Python)")
api_snippet =  completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=75,
                top_p=0.9


            )
print("response", completion.choices[0].message.content.strip())
