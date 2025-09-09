# Learny Chatbot

**Learny Chatbot** သည် အသုံးပြုသူ၏ အချက်အလက်များကို မှတ်မိနိုင်ပြီး၊ စကားဝိုင်းများအပေါ်မူတည်၍ Fine-Tune ပြုလုပ်နိုင်သည့်အပြင် GPT-2 ကို အသုံးပြု၍ တုံ့ပြန်မှုများကို ထုတ်ပေးနိုင်သော chatbot တစ်ခုဖြစ်သည်။

## လိုအပ်ချက်များ

Chatbot ကို run မလုပ်မီ အောက်ပါအရာများကို သေချာစွာ install လုပ်ထားရန်လိုအပ်ပါသည်-

- Python 3.8 သို့မဟုတ် အထက်
- Python package များ (requirement.txt တွင် ဖော်ပြထားသည်)

## Installation

1. Repository ကို Clone လုပ်ပါ သို့မဟုတ် Project File များကို Download လုပ်ပါ။
2. လိုအပ်သော Python package များကို Install လုပ်ပါ-

    ```bash
    pip install -r requirement.txt
    ```

3. Speech Synthesis အတွက် လိုအပ်သော System Dependency များကို Install လုပ်ပါ-

    ```bash
    sudo apt install espeak ffmpeg libespeak1
    ```

## Configuration

Chatbot ၏ အပြုအမူများကို `config.json` ဖိုင်ကို အသုံးပြု၍ ပြင်ဆင်နိုင်ပါသည်။ အဓိက Setting များမှာ-

- `train_interval`: Fine-Tuning ပြုလုပ်ရန်မတိုင်မီ အသုံးပြုသူ၏ input အရေအတွက်။
- `max_train_steps`: Fine-Tuning ပြုလုပ်သည့်အချိန်အတွင်း အများဆုံး Training Step အရေအတွက်။
- `speech_rate`: Text-to-Speech output ၏ အမြန်နှုန်း။
- `model_dir`: Fine-Tuned Model ကို သိမ်းဆည်း/ဖွင့်ရန် Directory။
- `base_model`: Text Generation အတွက် အသုံးပြုမည့် Base Model (Default: GPT-2)။
