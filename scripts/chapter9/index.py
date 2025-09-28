from langgraph.graph import END, StateGraph

from scripts.chapter9.node import answering_node, check_node, selection_node
from scripts.chapter9.state import State

# ステートグラフの初期化
workflow = StateGraph(State)

# ノードの追加：各処理ステップを定義
workflow.add_node("selection", selection_node)  # ロール選択ノード
workflow.add_node("answering", answering_node)  # 回答生成ノード
workflow.add_node("check", check_node)  # 品質チェックノード

# エントリーポイントの設定（最初に実行されるノード）
workflow.set_entry_point("selection")

# エッジの追加：処理の流れを定義
workflow.add_edge("selection", "answering")  # 選択 → 回答
workflow.add_edge("answering", "check")  # 回答 → チェック

# 条件分岐エッジ：チェック結果に基づいて次のステップを決定
workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,  # 判定結果を評価
    {
        True: END,  # 品質OK → 終了
        False: "selection"  # 品質NG → ロール再選択
    },
)

# ワークフローのコンパイル
compiled = workflow.compile()

# 初期状態の設定と実行
initial_state = State(query="生成AIについて教えてください。")
result = compiled.invoke(initial_state)

# 最終的な回答を出力
print(result["messages"][-1])

# ↓出力結果
"""
生成AI（生成的人工知能）とは、人工知能の一分野であり、新しいデータやコンテンツを生成する能力を持つモデルやシステムを指します。これには、テキスト、画像、音声、動画など、さまざまな形式のコンテンツが含まれます。生成AIは、ディープラーニング技術を活用して、大量のデータを学習し、そのパターンを理解することで、新しいコンテンツを生成します。

代表的な生成AIの技術には、以下のようなものがあります：

1. **GPT（Generative Pre-trained Transformer）**: テキスト生成に特化したモデルで、自然言語処理の分野で広く利用されています。GPTは、与えられたプロンプトに基づいて、自然な文章を生成することができます。

2. **GAN（Generative Adversarial Networks）**: 画像生成において特に有名な技術で、2つのニューラルネットワーク（生成者と識別者）が競い合うことで、リアルな画像を生成します。

3. **VAE（Variational Autoencoders）**: データの潜在的な特徴を学習し、新しいデータを生成するために使用されるモデルです。

生成AIは、クリエイティブなコンテンツの制作、自動化されたデザイン、データの補完、シミュレーションなど、さまざまな分野で応用されています。しかし、生成AIの利用には倫理的な課題も伴い、フェイクニュースの生成や著作権の問題などが議論されています。生成AIを活用する際には、これらの課題に対する理解と責任ある利用が求められます。
"""
