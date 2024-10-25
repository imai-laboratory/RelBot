from openai import OpenAI
from conversation_class import Conversation
from relation_class import Relation
from sigmoid import sigmoid
from balance_or_not import balance_or_not
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

relation_instance = Relation()

# pepperくんの態度の初期値の設定。これで現在の関係性の数値をxに変換。
attitude_parameters = []
attitude_parameters.append(round(relation_instance.get()[0]['康太と太郎の関係'] * 0.6 - 3, 1))
attitude_parameters.append(round(relation_instance.get()[2]['太郎と花子の関係'] * 0.6 - 3, 1))
# x　　　　　　　　　　　　　　：-3   , -2   , -1   , 0, 1   , 2   , 3
# gptに与えられる態度パラメータ：-0.91, -0.76, -0.46, 0, 0.46, 0.76, 0.91


# chatGPTに会話履歴を入力し，Pepperくんの発言を生成。Pepperくんの態度の制御も行う。
def chat1(topic):
    # シングルトンパターンによるインスタンス生成により、クラスに保存された情報を受け取れる
    history = Conversation()
    relation_instance = Relation()

    global attitude_parameters

    # Pepperくんの態度の制御
    # 現在の関係性からフィードバックを受け取る（初期化）
    attitude_parameters[0] = round(relation_instance.get()[0]['康太と太郎の関係'] * 0.6 - 3, 1)
    attitude_parameters[1] = round(relation_instance.get()[2]['太郎と花子の関係'] * 0.6 - 3, 1)
    # 人間の望む関係をもとに、態度を変更
    if relation_instance.get_future():  # 人間が喋った後なら態度を制御する
        human_intent_relation = relation_instance.get_future()
        balance = balance_or_not()
        # '人間とPepperくんの関係'に関して、人間の意図が'+'か'-'か'?'か
        if human_intent_relation['康太と太郎の関係'] == '+':
            # バランス理論に従っているかどうか
            if balance == 'yes':
                attitude_parameters[0] += 1.5
            else:
                # 人間の意図と現在の態度の符号が同じかどうか
                if attitude_parameters[0] >= 0:
                    attitude_parameters[0] += 2
                else:
                    attitude_parameters[0] += 3
        elif human_intent_relation['康太と太郎の関係'] == '-':
            if balance == 'yes':
                attitude_parameters[0] -= 1.5
            else:
                if attitude_parameters[0] < 0:
                    attitude_parameters[0] -= 2
                else:
                    attitude_parameters[0] -= 3
        # 'PepperくんとPepperちゃんの関係'に関して、人間の意図が'+'か'-'か'?'か
        if human_intent_relation['太郎と花子の関係'] == '+':
            # バランス理論に従っているかどうか
            if balance == 'yes':
                attitude_parameters[1] += 1.5
            else:
                # 人間の意図と現在の態度の符号が同じかどうか
                if attitude_parameters[1] >= 0:
                    attitude_parameters[1] += 2
                else:
                    attitude_parameters[1] += 3
        elif human_intent_relation['太郎と花子の関係'] == '-':
            if balance == 'yes':
                attitude_parameters[1] -= 1.5
            else:
                if attitude_parameters[1] < 0:
                    attitude_parameters[1] -= 2
                else:
                    attitude_parameters[1] -= 3
        attitude_parameters = [max(min(x, 3.0), -3.0) for x in attitude_parameters]

    print(attitude_parameters)

    if topic == '早慶戦':
        topic_sentence = '\n早慶戦について、\n慶応大学を応援する側として会話してください。'
    elif topic == 'きのこの山派かたけのこの里派か':
        topic_sentence = '\nきのこたけのこ戦争について、\nきのこの山派として会話してください。'
    elif topic == '都会に住みたいか田舎に住みたいか':
        topic_sentence = '\n都会と田舎どっちで住みたいかについて、\n都会派として会話してください。'
    elif topic == '結婚の必要性':
        topic_sentence = '\n結婚の必要性について、\n結婚は必要派として会話してください。'
    elif topic == '朝食の必要性':
        topic_sentence = '\n朝食の必要性について、\n朝食は必要派として会話してください。'

    default_messages = [{"role": "system",
                        "content": f"あなたは太郎です。太郎、花子、康太が会話します。過去の会話を考慮して太郎の会話を50文字以内で生成してください。\
    {topic_sentence}\
    \n太郎の会話部分のみを生成してください。\
    \n1人称は「僕」です。2人称と3人称の人代名詞は使わず、もしどちらかに話しかける場合は名前を呼んでください。\n康太の会話は生成しないでください。\n花子の会話は生成しないでください。\
    \n-1〜1の実数で関係を表します。1に近いほど関係は良好で、-1に近いほど関係は険悪です。\n太郎と康太の関係は{round(sigmoid(attitude_parameters[0]), 2)}にするように会話生成してください。\
    \n太郎と花子の関係は{round(sigmoid(attitude_parameters[1]), 2)}にするように会話生成してください。"}]  # プロンプトを入力
    next_messages = default_messages

    # 態度が変わったときに、systemを変更。最初Pepper同時がマイナスであることは前提。
    t = next_messages[0]["content"].split("\n")
    if attitude_parameters[1] < 0:
        t[2] = topic_sentence.split("、")[1].replace("\n", "")
    elif attitude_parameters[1] >= 0:
        t[2] = "会話してください。"
    next_messages[0]["content"] = "\n".join(t)

    # 会話履歴をプロンプトに追加
    combined_content = "\n".join(history.get())
    next_messages.append({"role": "user",
                         "content": combined_content}
                         )

    print(next_messages[0])  # systemを表示

    res = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        # model="gpt-4",
        model="gpt-4o",
        messages=next_messages)

    # "太郎："の部分を除いて発言部分だけに
    res_only = res.choices[0].message.content.replace("太郎:", "").replace("太郎: ", "").replace("太郎：", "")\
        .replace("「", "").replace("」", "")

    # GPTの返答
    return res_only


# キーボード入力での動作確認用
if __name__ == '__main__':
    history = Conversation()
    history.add("康太", "こんにちは")
    history.add("太郎", "こんにちは")
    history.add("花子", "こんにちは")
    history.add("康太", "今日はいい天気ですね")
    res = chat1()
    print('Pepperくん: ' + res)
