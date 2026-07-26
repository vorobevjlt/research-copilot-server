from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config.index import appConfig

openAI = {
    "embeddings_llm": ChatOpenAI(
        model="gpt-5.6-terra", api_key=appConfig["openai_api_key"], base_url="https://api.proxyapi.ru/openai/v1"
    ),
    "embeddings": OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=appConfig["openai_api_key"],
        dimensions=1536,
        base_url="https://api.proxyapi.ru/openai/v1"
    ),
    "chat_llm": ChatOpenAI(
        model="gpt-5.6-terra", api_key=appConfig["openai_api_key"], base_url="https://api.proxyapi.ru/openai/v1"
    ),
}
