from openai import OpenAI
import streamlit as st

# OpenAI API 키 설정
api_key = 'none'
client = OpenAI(api_key=api_key, base_url="https://f08a-14-35-173-251.ngrok-free.app/v1")
# Streamlit UI 구성
st.title("Real-time Text Generation with OpenAI API")

prompt = st.text_input("Enter your prompt:", "")

if st.button("Generate Text"):
    if prompt:
        # 텍스트 생성 중 표시
        st.write("Generating text...")

        # OpenAI API 호출
        response = client.chat.completions.create(
#            model='gpt-4o-mini',
            model='mistral:7b',
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            temperature=0,
            stream=True  # this time, we set stream=True
        )

        generated_text = ""
        # 실시간으로 텍스트 표시
        # for chunk in response:
        #     chunk_text = chunk.choices[0].delta.content
        #     generated_text += chunk_text
        #     st.text_area("Generated Text:", value=generated_text, height=300)
        st.write_stream(response)
    else:
        st.warning("Please enter a prompt to generate text.")
