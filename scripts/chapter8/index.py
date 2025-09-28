import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# async def main() -> None:
#     model_client = OpenAIChatCompletionClient(model="gpt-4.1")
#     agent = AssistantAgent("assistant", model_client=model_client)
#     print(await agent.run(task="Say 'Hello World!'"))
#     await model_client.close()

# asyncio.run(main())
async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4.1")
    agent = AssistantAgent("assistant", model_client=model_client)
    result = await agent.run(task="Say 'Hello World!'")
    output = result.messages[-1].content
    print(output)
    await model_client.close()


asyncio.run(main())
