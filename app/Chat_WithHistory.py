import gradio as gr
import os
import logging
import httpx
# from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGUF/GGML models
import datetime
from langserve import RemoteRunnable
from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chain = RemoteRunnable("http://localhost:8080/chat/")
ingest = RemoteRunnable("http://localhost:8080/ingest/")

# logfile = './app/chat_history.txt'
# Define the path to the chat history file using absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script resides
log_directory = os.path.join(script_dir, 'app')
logfile = os.path.join(log_directory, 'chat_history.txt')

print("loading model...")
stt = datetime.datetime.now()
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")
#MODEL SETTINGS also for DISPLAY

# def writehistory(text):
#     with open(logfile, 'a', encoding='utf-8') as f:
#         f.write(text)
#         f.write('\n')
#     f.close()

def writehistory(text: str):
    try:
        # Ensure the log directory exists
        os.makedirs(log_directory, exist_ok=True)
        logger.info(f"Log directory ensured at: {log_directory}")

        # Write to the log file
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        logger.info(f"Written to log: {text}")
    except Exception as e:
        logger.error(f"Failed to write to {logfile}: {e}")

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]

# uploaded_files = []


# def process_file(files):
#     global uploaded_files
#     uploaded_files.extend(files)
#     names = ""
#     if files is not None:
#         # Reading the content of the file (for text files)
#         for file in files:
#             names += file.name + "\n"

#         print(f"File uploaded: {names}")    
#     return f"Tổng số file đã tải lên: {len(uploaded_files)}"

# def clear_files():
#     global uploaded_files
#     uploaded_files.clear()
#     print("Danh sách file đã được làm trống.")
#     return "Danh sách file đã được làm trống."

with gr.Blocks(theme='ParityError/Interstellar') as demo: 
    #TITLE SECTION
    # with gr.Row():
    #     with gr.Column(scale=12):
    #         gr.HTML("<center>"
    #         + "<h1>🤖 Chat bot retrieves based on your data 🤖</h1></center>")  
    #         gr.Markdown("""
    #         **Currently Running**:  [Ollama - mistral](https://ollama.com/library/mistral) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Chat History Log File**: *chat_history.txt*  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
    #         - Vector Store: [Chroma](https://docs.trychroma.com/deployment) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
    #         - Embedding: [OllamaEmbeddings - nomic-embed-text](https://ollama.com/library/nomic-embed-text) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    #         """)         
    #     gr.Image(value='./app/langchain3.webp', height="100%", width='100%')
   # chat and parameters settings
    with gr.Row():
        # with gr.Column(scale=2):
        #     gr.Interface(
        #         fn=process_file,                       # The function to process the file
        #         inputs=gr.Files(label="Upload your file", file_count="multiple"),  # Drag and drop file upload
        #         outputs="text",                        # Output text message
        #         title="Drag and Drop File Upload",     # Title for the interface
        #         description="Drag and drop a file or click to upload.", # Description
        #         allow_flagging = False,
        #     )
        
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height = 500, show_copy_button=False,
                                 avatar_images = ["./app/profile.png","./app/chatbot.jpg"])
            with gr.Row():
                with gr.Column(scale=14):
                    msg = gr.Textbox(show_label=False, 
                                     placeholder="Enter text",
                                     lines=2)
                submitBtn = gr.Button("\n💬 Send\n", size="lg", variant="primary", min_width=120)

        with gr.Column(min_width=50,scale=2):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.30,
                        step=0.01,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=360,
                        step=4,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    rep_pen = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1.2,
                        step=0.05,
                        interactive=True,
                        label="Repetition Penalty",
                    )
                    gr.Markdown("""
                    ### History
                    Insert num of chat rounds for conversation context
                    """)
                    mem_limit = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=8,
                        step=1,
                        interactive=True,
                        label="Chat History",
                    )

                clear = gr.Button("🗑️ Clear All Messages", variant='secondary')
    def user(user_message, history):

        writehistory(f"USER: {user_message}")
        return "", history + [[user_message, ""]]

    async def bot(history,t,m,r,limit):
        chat_history = []
        # always keep len(history) <= memory_limit
        if len(history) > limit:
            chat_history = history[-limit:]   
            print("History above set limit")
        else:
            chat_history = history
        # First prompt different because does not contain any context    
        chat_history = [{'human': sublist[0], 'ai': sublist[1]} for sublist in chat_history]
        if len(history) == 1:
            logger.info("Initial message without context")
            chat_request = ChatRequest(
            question=history[-1][0],
            chat_history=[]  # No context for the first message
        )
        else:
            logger.info(f"Processing history: {history}")
            chat_request = ChatRequest(
                question=history[-1][0],
                chat_history=chat_history[:-1]  # Exclude the last user message
            )
        # Preparing the CHATBOT reply
        history[-1][1] = ""
        
        try:
            async for chunk in chain.astream(
                chat_request, 
                config=RunnableConfig(
                    temperature=t, 
                    max_new_tokens=m, 
                    repetition_penalty=r
                )
            ):
                history[-1][1] += chunk
                yield history
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Status Error: {e}")
            history[-1][1] = "I'm sorry, something went wrong while processing your request."
            yield history
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            history[-1][1] = "I'm sorry, an unexpected error occurred."
            yield history
    
        # Log the interaction
        writehistory(f"temperature: {t}, maxNewTokens: {m}, repetitionPenalty: {r}\n---\nBOT: {history}\n\n")
        logger.info(f"USER: {history[-1][0]}\n---\ntemperature: {t}, maxNewTokens: {m}, repetitionPenalty: {r}\n---\nBOT: {history[-1][1]}\n\n")

    # Clicking the submitBtn will call the generation with Parameters in the slides
    submitBtn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot,temperature,max_length_tokens,rep_pen,mem_limit], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()  #required to yield the streams from the text generation
# demo.launch(inbrowser=True)
demo.launch(share=True)