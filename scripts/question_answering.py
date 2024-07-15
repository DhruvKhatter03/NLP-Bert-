import openai

openai.api_key = 'sk-proj-6gem5qtG1ytYnnmrHbJoT3BlbkFJBo0vJfcXvgiD8Z3Q0nmA'

def answer_question(query, context, engine="text-davinci-003"):
    response = openai.Completion.create(
        engine=engine,
        prompt=f"Answer the following question based on the given context:\n\nContext:\n{context}\n\nQuestion:\n{query}",
        max_tokens=150
    )
    answer = response.choices[0].text.strip()
    return answer

if __name__ == "__main__":
    context = "Machine learning is a method of data analysis that automates analytical model building."
    query = "What is machine learning?"
    answer = answer_question(query, context)
    print("Answer:", answer)
