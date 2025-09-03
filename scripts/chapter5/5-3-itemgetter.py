import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from operator import itemgetter

# topic_getter = itemgetter('topic')
# topic = topic_getter({"topic": "生成AIの進化について"})
# print(topic)