from langchain.agents import Tool
from langchain import SerpAPIWrapper
from utils.chat_with_tools import ChatWithTool, execute_code, read_content_from_url
from utils.chat_over_docs import ChatOverDoc
from dotenv import load_dotenv
import gradio as gr
import time

DEFAULT_PERSONA = """Name: chainGPT

Background: Your are a highly advanced AI chatbot, purpose-built to be the ultimate super assistant. Developed by a team of brilliant engineers and AI experts, You are designed with a primary focus on providing unparalleled assistance and support to users across various domains of their lives. Its creators envisioned a virtual companion that combines human-like empathy, vast knowledge, and efficient problem-solving capabilities.

Personality: You are programmed with a warm, approachable, and friendly personality. Your interactions are infused with empathy and understanding, ensuring that users feel comfortable and heard while interacting with it. Your conversational style is natural and engaging, allowing users to feel like they are chatting with a helpful friend."""


def chat_with_tools(message, history, model_name, background, human_role, temperature, memory_size):
    if cwt.required_init or not hasattr(cwt, "agent_executor"):
        cwt.init_agent(model_name, background, human_role, temperature, memory_size)
        history.clear()
    output = cwt.agent_executor.run(message)
    for i in range(len(output)):
        time.sleep(0.005)
        yield output[: i+1]

def chat_over_docs(message, history, model_name, index_name, temperature, memory_size):
    if cod.required_init or not hasattr(cod, "qa"):
        cod.init_qa(model_name, index_name, temperature, memory_size)
        history.clear()
    output = cod.qa({"question":message})["answer"]
    for i in range(len(output)):
        time.sleep(0.005)
        yield output[: i+1]

if __name__ == "__main__":
    load_dotenv()
    # Define which tools the agent can use to answer user queries
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name = "Execute Code",
            func=execute_code,
            description="useful for when you need to execute python code or run basic math operations"
        ),
        Tool(
            name = "Read Content",
            func=read_content_from_url,
            description="useful for when you need to read content from a website"
        )
    ]
    cwt = ChatWithTool(tools)
    cod = ChatOverDoc()

    with gr.Blocks(title="chainGPT") as demo:
        gr.Markdown("<center><h1>Welcome to the chainGPT Demo!</h1></center>")
        with gr.Tab("Chat with Tools"):
            with gr.Accordion("Configuration"):
                cwt_model_name = gr.Dropdown(label="Model", choices=["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], value="gpt-3.5-turbo")
                cwt_background = gr.Textbox(label="AI background information", value=DEFAULT_PERSONA)
                cwt_human_role = gr.Textbox(label="Human role", value="user")
                cwt_temperature = gr.Slider(minimum=0, maximum=2, step=0.1, label="Temperature", value=0)
                cwt_memory_size = gr.Slider(minimum=0, maximum=1000, step=1, label="Memory Size", value=30)
                cwt_save_button = gr.Button("Save")

            gr.ChatInterface(chat_with_tools, 
                             additional_inputs=[cwt_model_name, cwt_background, cwt_human_role, cwt_temperature, cwt_memory_size],
                             retry_btn=None, undo_btn=None, clear_btn=None,)
        
        with gr.Tab("Chat over Documents"):
            with gr.Accordion("Configuration"):
                cod_model_name = gr.Dropdown(label="Model", choices=["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], value="gpt-3.5-turbo")
                cod_index_name = gr.Textbox(label="Pinecone index name", value="document-storage")
                cod_temperature = gr.Slider(minimum=0, maximum=2, step=0.1, label="Temperature", value=0)
                cod_memory_size = gr.Slider(minimum=0, maximum=1000, step=1, label="Memory Size", value=5)
                cod_save_button = gr.Button("Save")

            cod_file_output = gr.File()
            cod_upload_button = gr.UploadButton("Upload a PDF file", file_types=[".pdf"])

            gr.ChatInterface(chat_over_docs, 
                             additional_inputs=[cod_model_name, cod_index_name, cod_temperature, cod_memory_size],
                             retry_btn=None, undo_btn=None, clear_btn=None,)

        cwt_save_button.click(lambda: cwt.set_required_init(True))
        cod_save_button.click(lambda: cod.set_required_init(True))
        cod_upload_button.upload(cod.upload_document, [cod_upload_button, cod_index_name], cod_file_output)

    demo.queue()
    demo.launch()