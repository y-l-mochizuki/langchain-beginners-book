# from langchain_anthropic import ChatAnthropic  # コメントアウト: APIキー未設定のため
from langchain_openai import ChatOpenAI

from scripts.chapter12.common.reflection_manager import (ReflectionManager, TaskReflector)
from scripts.chapter12.self_reflection.main import ReflectiveAgent


def main():
    import argparse

    from scripts.chapter12.settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(description="ReflectiveAgentを使用してタスクを実行します（Cross-reflection）")
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    # OpenAIのLLMを初期化
    openai_llm = ChatOpenAI(
        model=settings.openai_smart_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )

    # AnthropicのLLMを初期化（コメントアウト: APIキー未設定のため）
    # anthropic_llm = ChatAnthropic(model=settings.anthropic_smart_model,
    #                               temperature=settings.temperature)

    # ReflectionManagerを初期化
    reflection_manager = ReflectionManager(file_path="tmp/cross_reflection_db.json")

    # OpenAIのLLMを使用するTaskReflectorを初期化（Anthropic代わりにOpenAIを使用）
    openai_task_reflector = TaskReflector(llm=openai_llm,
                                          reflection_manager=reflection_manager)

    # ReflectiveAgentを初期化
    agent = ReflectiveAgent(
        llm=openai_llm,
        reflection_manager=reflection_manager,
        task_reflector=openai_task_reflector,
    )

    # タスクを実行し、結果を取得
    result = agent.run(args.task)

    # 結果を出力
    print(result)


if __name__ == "__main__":
    main()
