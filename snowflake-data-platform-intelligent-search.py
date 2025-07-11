import os
from dotenv import load_dotenv
import streamlit as st
from snowflake.snowpark import Session
from snowflake.core import Root

load_dotenv()

# ------------------------ MODEL CHOICES ------------------------ #
MODELS = [
    "mistral-large",
    "llama3-70b",
    "llama3-8b",
]

# ------------------------ STRUCTURED KEYWORDS ------------------------ #
STRUCTURED_KEYWORDS = [
    "nasdaq traded", "symbol", "security name", "listing exchange",
    "market category", "etf", "round lot size", "test issue",
    "financial status", "cqs symbol", "nasdaq symbol", "nextshares"
]

STRUCTURED_SERVICE = "stock_market_service"
UNSTRUCTURED_SERVICE = "prospectus_service"

# ------------------------ SESSION CREATION ------------------------ #
@st.cache_resource
def create_session():
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
    }
    return Session.builder.configs(connection_parameters).create()

session = create_session()
root = Root(session)

# ------------------------ INIT METHODS ------------------------ #
def init_messages():
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def init_service_metadata():
    if "service_metadata" not in st.session_state:
        services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
        service_metadata = []
        if services:
            for s in services:
                svc_name = s["name"]
                svc_search_col = session.sql(
                    f"DESC CORTEX SEARCH SERVICE {svc_name};"
                ).collect()[0]["search_column"]
                service_metadata.append(
                    {"name": svc_name, "search_column": svc_search_col}
                )
        st.session_state.service_metadata = service_metadata

# ------------------------ CLASSIFIER ------------------------ #
def classify_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for keyword in STRUCTURED_KEYWORDS:
        if keyword in prompt_lower:
            return "structured"
    return "unstructured"

# ------------------------ CONFIG OPTIONS ------------------------ #
def init_config_options():
    st.sidebar.button("Clear conversation", key="clear_conversation")
    st.sidebar.toggle("Debug", key="debug", value=False)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)

    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name")
        st.number_input("Select number of context chunks", value=5, key="num_retrieved_chunks", min_value=1, max_value=10)
        st.number_input("Select number of messages to use in chat history", value=5, key="num_chat_messages", min_value=1, max_value=10)

    if st.session_state.debug:
        st.sidebar.expander("Session State").write(st.session_state)

# ------------------------ MAIN UTILITY METHODS ------------------------ #
def query_cortex_search_service(query):
    db, schema = session.get_current_database(), session.get_current_schema()

    # Classify prompt
    service_type = classify_prompt(query)
    service_name = STRUCTURED_SERVICE if service_type == "structured" else UNSTRUCTURED_SERVICE
    st.session_state.selected_cortex_search_service = service_name

    cortex_search_service = root.databases[db].schemas[schema].cortex_search_services[service_name]

    context_documents = cortex_search_service.search(
        query, columns=[], limit=st.session_state.num_retrieved_chunks
    )
    results = context_documents.results

    service_metadata = st.session_state.service_metadata
    search_col_list = [
        s["search_column"]
        for s in service_metadata
        if s["name"].lower() == service_name.lower()
    ]

    if not search_col_list:
        raise ValueError(f"Search service '{service_name}' not found in service metadata.")

    search_col = search_col_list[0]

    context_str = ""
    for i, r in enumerate(results):
        context_str += f"Context document {i+1}: {r[search_col]}\n\n"

    if st.session_state.debug:
        st.sidebar.text_area("Context documents", context_str, height=500)

    return context_str

def get_chat_history():
    start_index = max(0, len(st.session_state.messages) - st.session_state.num_chat_messages)
    return st.session_state.messages[start_index : len(st.session_state.messages) - 1]

def complete(model, prompt):
    return session.sql("SELECT snowflake.cortex.complete(?, ?)", (model, prompt)).collect()[0][0]

def make_chat_history_summary(chat_history, question):
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """
    summary = complete(st.session_state.model_name, prompt)

    if st.session_state.debug:
        st.sidebar.text_area("Chat history summary", summary.replace("$", "\$"), height=150)

    return summary

def create_prompt(user_question):
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history != []:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context = query_cortex_search_service(question_summary)
        else:
            prompt_context = query_cortex_search_service(user_question)
    else:
        prompt_context = query_cortex_search_service(user_question)
        chat_history = ""

    prompt = f"""
        [INST]
        You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question,
        you will also be given context provided between <context> and </context> tags. Use that context
        with the user's chat history provided in the between <chat_history> and </chat_history> tags
        to provide a summary that addresses the user's question. Ensure the answer is coherent, concise,
        and directly relevant to the user's question.

        If the user asks a generic question which cannot be answered with the given context or chat_history,
        just say "I don't know the answer to that question."

        Don't say things like "according to the provided context".

        <chat_history>
        {chat_history}
        </chat_history>
        <context>
        {prompt_context}
        </context>
        <question>
        {user_question}
        </question>
        [/INST]
        Answer:
    """
    return prompt

# ------------------------ MAIN ENTRY POINT ------------------------ #
def main():
    st.title(":speech_balloon: Chatbot with Snowflake Cortex")

    init_service_metadata()
    init_config_options()
    init_messages()

    icons = {"assistant": "❄️", "user": "👤"}

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.markdown(message["content"])

    disable_chat = (
        "service_metadata" not in st.session_state
        or len(st.session_state.service_metadata) == 0
    )

    if question := st.chat_input("Ask a question...", disabled=disable_chat):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=icons["user"]):
            st.markdown(question.replace("$", "\$"))

        with st.chat_message("assistant", avatar=icons["assistant"]):
            message_placeholder = st.empty()
            question = question.replace("'", "")
            with st.spinner("Thinking..."):
                generated_response = complete(
                    st.session_state.model_name, create_prompt(question)
                )
                message_placeholder.markdown(generated_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": generated_response}
        )

if __name__ == "__main__":
    main()