import os

from autogen_agentchat.agents import AssistantAgent

config_list = {"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]},

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

# ragproxyagent = RetrieveUserProxyAgent(
#     name="ragproxyagent",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=3,
#     retrieve_config={
#         "task": "code",
#         "docs_path": ["https://github.com/langchain-ai/langchain",],
#         "custom_text_types": ["mdx"],
#         "chunk_token_size": 2000,
#         # "model": config_list[0]["model"],
#         # "client": chromadb.PersistentClient(path="/tmp/chromadb"),
#         "embedding_model": "all-mpnet-base-v2",
#         "get_or_create":
#             True,  # set to False if you don't want to reuse an existing collection, but you'll need to remove the collection manually
#     },
#     code_execution_config=False,  # set to False if you don't want to execute the code
# )

# code_problem = "langhainについて教えて"
# ragproxyagent.initiate_chat(assistant,
#                             message=ragproxyagent.message_generator,
#                             problem=code_problem,
#                             search_string="spark")

# response = ragproxyagent.run()
# print(response)
