import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import tempfile
import re

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# --- SECRETS LOADING FOR STREAMLIT CLOUD (CORRECTED) ---
# We populate os.environ from Streamlit's secrets for LangChain compatibility.

# Use st.secrets to check for keys and expose them as environment variables
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# NOTE: Colab-specific 'userdata.get()' calls for other keys have been REMOVED.

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="InsightForge: AI Business Intelligence Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use OpenAI by default, fall back to Groq if key is missing
MODEL_NAME = "gpt-4-turbo-preview"  # or "llama3-70b-8192" for Groq
LLM_PROVIDER = "OpenAI"

if os.environ.get('GROQ_API_KEY') and not os.environ.get('OPENAI_API_KEY'):
    LLM_PROVIDER = "Groq"
    MODEL_NAME = "llama3-70b-8192"
elif not os.environ.get('OPENAI_API_KEY') and not os.environ.get('GROQ_API_KEY'):
    # *** MODIFIED ERROR MESSAGE ***
    st.error("FATAL: Neither OPENAI_API_KEY nor GROQ_API_KEY are set. Please set keys in Streamlit Cloud Secrets.")
    st.stop()

# --- 2. LLM & EMBEDDING SETUP ---

# Initialize LLM based on available keys
@st.cache_resource
def load_llm():
    if LLM_PROVIDER == "OpenAI":
        return ChatOpenAI(temperature=0, model=MODEL_NAME)
    elif LLM_PROVIDER == "Groq":
        return ChatGroq(temperature=0, model_name=MODEL_NAME)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = load_llm()
embeddings = load_embeddings()

# --- 3. STATE MANAGEMENT & DATA PROCESSING (UNIFIED) ---

# New State for Multi-File/Mixed-Media Support
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}  # {filename: df} for structured data
if 'pdf_names' not in st.session_state:
    st.session_state.pdf_names = set()  # PDF filenames
if 'pdf_chunks' not in st.session_state:
    st.session_state.pdf_chunks = []  # all text chunks from all PDFs
if 'retriever' not in st.session_state:
    st.session_state.retriever = None  # single retriever for PDF/unstructured data
if 'master_agent_executor' not in st.session_state:
    st.session_state.master_agent_executor = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_uploaded_name' not in st.session_state:
    st.session_state.last_uploaded_name = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None

def process_structured_data(uploaded_file):
    """Loads CSV/Excel data and adds it to the session state."""
    filename = uploaded_file.name
    try:
        # Load the file
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return

        # ---------------------------
        # NORMALIZE COLUMN NAMES
        # ---------------------------

        # Base cleanup (remove spaces/dashes → underscores)
        df.columns = (
            df.columns
                .str.strip()
                .str.replace(r'[\\s\\-]+', '_', regex=True)
        )

        # Fix double underscores
        df.columns = df.columns.str.replace('__', '_', regex=False)

        # Standardize names for visualizations
        rename_map = {
            'date': 'Date',
            'Date': 'Date',

            'product': 'Product',
            'Product': 'Product',

            'region': 'Region',
            'Region': 'Region',
            'region_sales': 'Region',
            'Region_Sales': 'Region',

            'sales_revenue': 'Sales_Revenue',
            'sale_revenue': 'Sales_Revenue',
            'Sale_Revenue': 'Sales_Revenue',
            'Sale__Revenue': 'Sales_Revenue',
            'Sales_Revenue': 'Sales_Revenue'
        }

        df.rename(columns=rename_map, inplace=True)

        # ---------------------------
        # SAVE DF
        # ---------------------------
        st.session_state.dfs[filename] = df
        st.success(f"Structured Data loaded: {filename}")

    except Exception as e:
        st.error(f"Error processing structured data {filename}: {e}")

def create_retriever_from_chunks():
    """Builds the FAISS index from all stored chunks and updates the retriever."""
    if st.session_state.pdf_chunks:
        vectorstore = FAISS.from_documents(st.session_state.pdf_chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.info(
            f"Unified PDF RAG index rebuilt with {len(st.session_state.pdf_chunks)} chunks "
            f"from {len(st.session_state.pdf_names)} document(s)."
        )
    else:
        st.session_state.retriever = None

def process_pdf_data(uploaded_file):
    """Loads PDF data, splits it into chunks, and adds it to the global chunk list."""
    filename = uploaded_file.name
    try:
        # Write file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load PDF pages
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)

        # Split into text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)

        # Store globally
        st.session_state.pdf_chunks.extend(texts)
        st.session_state.pdf_names.add(filename)

        st.success(f"PDF loaded: {filename}. Added {len(texts)} chunks.")

    except Exception as e:
        st.error(f"Error processing PDF document {filename}: {e}")

def handle_file_upload(uploaded_file):
    """Router for file processing based on extension."""
    if uploaded_file.name.endswith(('.csv', '.xls', '.xlsx')):
        process_structured_data(uploaded_file)

    elif uploaded_file.name.endswith('.pdf'):
        process_pdf_data(uploaded_file)
        # Rebuild the RAG system to include the new PDF's chunks
        create_retriever_from_chunks()

    else:
        st.error("Unsupported file type. Please upload a CSV, Excel, or PDF file.")

    # Re-create the master agent after any file change
    create_master_agent()
    st.session_state.chat_history = []  # Reset chat history on file change

# --- MASTER AGENT CREATION (UNIFIED SEARCH) ---

# Helper function to create the structured data tool execution function
def create_pandas_executor_func(agent_executor, filename):
    """Creates a function that runs the Pandas agent and prepends the filename to the output."""
    def pandas_executor_func(query: str) -> str:
        result = agent_executor.invoke({"input": query})
        return f"ANSWER FROM **{filename}**: " + result.get("output", "Could not run structured data analysis.")
    return pandas_executor_func

# Helper function for PDF RAG tool
def pdf_qa_executor_func(query: str) -> str:
    """Executes the RetrievalQA chain and prepends the source label."""
    if st.session_state.retriever is None:
        return "No PDF document is currently loaded for RAG analysis."

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.retriever
    )
    response = qa_chain.invoke({"query": query})

    source_tag = f"ANSWER FROM **UNSTRUCTURED PDF DATA** (Documents: {', '.join(st.session_state.pdf_names)})"
    return source_tag + ": " + response.get("result", "Could not retrieve an answer from the PDF content.")

def create_master_agent():
    """Creates a master agent that can use all loaded DataFrames and the PDF RAG system."""
    all_tools = []

    # ------------------------------------------------------------
    # 1. Structured Data (CSV / Excel) Tools
    # ------------------------------------------------------------
    for filename, df in st.session_state.dfs.items():
        # Create a Pandas agent using LangChain experimental toolkit
        # NOTE: allow_dangerous_code=True is REQUIRED by newer LangChain versions
        pandas_agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True
        )

        tool_func = create_pandas_executor_func(pandas_agent_executor, filename)

        sanitized_name = filename.split('.')[0]
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', sanitized_name)

        pandas_tool = Tool(
            name=f"structured_data_on_{sanitized_name}",
            description=(
                f"Use this tool to answer business questions (statistics, queries, "
                f"calculations, patterns) ONLY about the structured data in: {filename}. "
                f"The input must be the full question."
            ),
            func=tool_func
        )
        all_tools.append(pandas_tool)

    # ------------------------------------------------------------
    # 2. PDF RAG Tool (if any PDFs processed)
    # ------------------------------------------------------------
    if st.session_state.retriever is not None:
        pdf_rag_tool = Tool(
            name="unstructured_pdf_document_qa",
            description=(
                "Use this tool to answer questions about the content of ALL uploaded "
                "PDF documents (unstructured data). Input must be a full question."
            ),
            func=pdf_qa_executor_func
        )
        all_tools.append(pdf_rag_tool)

    if not all_tools:
        st.session_state.master_agent_executor = None
        return

    # ------------------------------------------------------------
    # 3. Build Master Agent
    # ------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an intelligent assistant that can use multiple tools to analyze "
                "structured CSV/Excel data and unstructured PDF documents. Always reference "
                "the source filename in your final answers."
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    master_agent = create_tool_calling_agent(llm, all_tools, prompt)

    st.session_state.master_agent_executor = AgentExecutor(
        agent=master_agent,
        tools=all_tools,
        verbose=False,
        max_iterations=15,
        handle_parsing_errors=True
    )

    st.success(f"Master AI Agent initialized with {len(all_tools)} tool(s) for unified search.")

# --- 4. DATA VISUALIZATION FUNCTIONS ---

def generate_visual_insights(df):
    """Generate and display simple, insightful visualizations."""
    st.subheader("Visual Data Insights")
    st.markdown("Here are some automatically generated visualizations based on the current dataset.")

    if not isinstance(df, pd.DataFrame):
        st.warning("No structured data available for visualization.")
        return

    # Helper for safe plotting
    def safe_plot(title, fig_creator, warning_message):
        try:
            fig, ax = fig_creator()
            ax.set_title(title, fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"{warning_message} Error: {e}")

    # 1. Sales Trends Over Time (if Date column exists)
    def plot_sales_trend():
        if 'Date' in df.columns and 'Sales_Revenue' in df.columns:
            df_plot = df.copy()
            df_plot['Date'] = pd.to_datetime(df_plot['Date'], errors='coerce')
            df_plot['Sales_Revenue'] = pd.to_numeric(df_plot['Sales_Revenue'], errors='coerce')
            df_plot.dropna(subset=['Date', 'Sales_Revenue'], inplace=True)

            if not df_plot['Date'].empty:
                df_monthly = df_plot.set_index('Date')['Sales_Revenue'].resample('M').sum().reset_index()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_monthly['Date'], df_monthly['Sales_Revenue'], marker='o', color='teal')
                ax.set_xlabel("Date")
                ax.set_ylabel("Total Sales Revenue")
                plt.xticks(rotation=45)
                return fig, ax
        raise ValueError("Required columns for Sales Trend not found or data is invalid.")

    safe_plot("Monthly Sales Revenue Trend", plot_sales_trend, "Could not generate Sales Trend visualization.")

    # 2. Product Performance Comparison
    def plot_product_performance():
        if 'Product' in df.columns and 'Sales_Revenue' in df.columns:
            product_sales = df.groupby('Product')['Sales_Revenue'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            product_sales.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_xlabel("Product")
            ax.set_ylabel("Total Sales Revenue")
            plt.xticks(rotation=0)
            return fig, ax
        raise ValueError("Required columns for Product Performance not found.")

    safe_plot("Sales Revenue by Product", plot_product_performance, "Could not generate Product Performance visualization.")

    # 3. Regional Analysis
    def plot_regional_analysis():
        if 'Region' in df.columns and 'Sales_Revenue' in df.columns:
            region_sales = df.groupby('Region')['Sales_Revenue'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
            ax.axis('equal')
            return fig, ax
        raise ValueError("Required columns for Regional Analysis not found.")

    safe_plot("Sales Distribution by Region", plot_regional_analysis, "Could not generate Regional Analysis visualization.")

# --- 5. STREAMLIT UI LAYOUT ---

st.title(" InsightForge: AI Business Intelligence Assistant (Multi-File)")

st.markdown(f"**LLM Backend:** `{LLM_PROVIDER} / {MODEL_NAME}`")
st.markdown("Upload your business data (CSV, Excel) for structured analysis, and **PDF documents** for combined, mixed-media RAG analysis.")

# --- SIDEBAR: DATA UPLOAD & STATUS ---
with st.sidebar:
    st.header("1. Data Preparation")
    uploaded_files = st.file_uploader(
        "Upload a CSV, Excel (.xlsx/.xls), or PDF file",
        type=['csv', 'xlsx', 'xls', 'pdf'],
        accept_multiple_files=True
    )

    # Handle multiple file uploads
    if uploaded_files:
        for file in uploaded_files:
            is_structured = file.name.endswith(('.csv', '.xls', '.xlsx'))
            is_pdf = file.name.endswith('.pdf')

            if is_structured and file.name not in st.session_state.dfs:
                handle_file_upload(file)
            elif is_pdf and file.name not in st.session_state.pdf_names:
                handle_file_upload(file)

    # Logic to handle initial demo load if no files are uploaded
    if not st.session_state.dfs and not st.session_state.master_agent_executor and not st.session_state.pdf_names:
        st.info("No files uploaded. Please upload files.")

    st.subheader("Loaded Data Assets")
    if st.session_state.dfs:
        st.success(f"Structured Files: {', '.join(st.session_state.dfs.keys())}")
        first_df_name = next(iter(st.session_state.dfs))
        st.dataframe(st.session_state.dfs[first_df_name].head(), use_container_width=True)

    if st.session_state.pdf_names:
        st.success(f"Unstructured PDF Data RAG is Active. Documents: {', '.join(st.session_state.pdf_names)}")

    st.header("Master Agent Status")
    if st.session_state.master_agent_executor:
        st.success(f"Master Agent is Ready for Unified Search. Tools: {len(st.session_state.master_agent_executor.tools)}")
    else:
        st.warning("Upload data to initialize the Master Agent.")

# --- MAIN CONTENT TABS ---
tab1, tab2 = st.tabs(["AI Analysis & Chat", "Visual Insights"])

with tab1:
    st.header("Ask for Unified Insights & Recommendations")

    if st.session_state.master_agent_executor is None:
        st.warning("Please upload a CSV, Excel, or PDF file to start the analysis.")
    else:
        st.info("The Master Agent can query **multiple structured data files** and **combined PDF documents** simultaneously, and **it will note the source file** in the answer.")

        # Chat Input
        prompt_hint = "E.g., What is the average Sales Revenue from sales_data.csv, and what are the key risks mentioned in the market_report.pdf?"

        prompt = st.chat_input(prompt_hint)

        # Display chat history
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        if prompt:
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner(f"Analyzing data with {MODEL_NAME} using multiple tools..."):
                with st.chat_message("assistant"):
                    try:
                        response = st.session_state.master_agent_executor.invoke({"input": prompt})
                        answer = response.get("output", str(response))
                        st.markdown(answer)
                        st.session_state.chat_history.append(("assistant", answer))
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
                        st.markdown("I encountered an error during unified analysis. Please try rephrasing your question.")

with tab2:
    st.header("Automated Visualizations")
    if st.session_state.dfs:
        df_names = list(st.session_state.dfs.keys())
        selected_df_name = st.selectbox(
            "Select the Structured Data File to Visualize:",
            df_names,
            key='viz_file_selector'
        )
        df_to_visualize = st.session_state.dfs[selected_df_name]
        st.markdown(f"**Visualizations for:** `{selected_df_name}`")
        generate_visual_insights(df_to_visualize)
    else:
        st.warning("Visualizations are only generated for structured data (CSV/Excel) files.")