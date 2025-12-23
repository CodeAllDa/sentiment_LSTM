import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from PIL import Image
import io
from collections import Counter
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# 指定nltk文件路径
nltk.data.path.append('./nltk_data')

# 生成词云
def generate_word_cloud(text_data, title, max_words=200):
    """
    生成并显示词云
    """
    # 将所有评论合并为一个字符串
    all_text = ' '.join(text_data)

    # 创建词云对象
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        contour_width=3,
        contour_color='steelblue',
        collocations=False  # 避免显示搭配词（如"New York"）
    ).generate(all_text)

    # 绘制词云
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 将图像保存到内存中的二进制流
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')  # 保存为 PNG 格式
    img_buffer.seek(0)  # 将指针移到流的开头

    # 将二进制流保存到img_data中
    img_data = img_buffer.getvalue()

    return img_data

# 生成可视化训练结果
def generate_epoch_metrics():
    # 从CSV加载
    df = pd.read_csv('./data/training_history.csv')

    # 重新绘制图表
    epochs = list(range(1, len(df) + 1))
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(epochs, df['val_accuracy'], label='Validation Accuracy', marker='x')
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, df['loss'], label='Training Loss', marker='o')
    plt.plot(epochs, df['val_loss'], label='Validation Loss', marker='x')
    plt.title('Loss Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 将图像保存到内存中的二进制流
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')  # 保存为 PNG 格式
    img_buffer.seek(0)  # 将指针移到流的开头

    # 将二进制流保存到img_data中
    img_data = img_buffer.getvalue()

    return img_data

# 预测文本
def quick_predict(text, model_path='./model/sentiment_model.keras',
                  vocab_path='./data/vocabulary/word_index.pkl'):
    # 1. 加载模型和词汇表
    model = load_model(model_path)

    with open(vocab_path, 'rb') as f:
        word_index = pickle.load(f)

    # 2. 设置参数
    MAX_LENGTH = 100
    pad_id = word_index.get('<PAD>', 0)
    unk_id = word_index.get('<UNK>', 0)

    # 3. 预处理函数（与训练时相同）
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        tokens = word_tokenize(str(text).lower())
        return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]

    def text_to_sequence(tokens):
        seq = [word_index.get(t, unk_id) for t in tokens]
        if len(seq) > MAX_LENGTH:
            seq = seq[:MAX_LENGTH]
        else:
            seq = seq + [pad_id] * (MAX_LENGTH - len(seq))
        return seq

    # 4. 预处理和预测
    tokens = preprocess(text)
    sequence = text_to_sequence(tokens)
    sequence_array = np.array([sequence])

    # 5. 预测
    prediction = model.predict(sequence_array, verbose=0)[0][0]

    # 6. 返回结果
    sentiment = "正面" if prediction > 0.5 else "负面"

    return sentiment, float(prediction)

# streamlit run test.py


# 设置页面的标题、图标和布局
st.set_page_config(
    page_title="亚马逊食物评论情感分析",  # 页面标题
    layout='wide',
)
# 使用侧边栏实现多页面效果
with st.sidebar:
    import os

    # jpg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images\logo.png')
    from PIL import Image

    # img = Image.open(jpg_dir)
    st.image('images\logo.png', width=100)
    # st.image('images\logo.jpg', width=100)
    st.title('请选择页面')
    page = st.selectbox("请选择页面", ["项目简介", "词云", "模型训练结果图", "情感分析"],
                        label_visibility='collapsed')

if page == "项目简介":
    st.title("评论情感分析")
    st.header('技术介绍')
    st.markdown("""基于nltk生成的词云、基于LSTM模型预测评论的积极情感或消极情感。
    数据集来源于天池的公开数据集Amazon Fine Food Reviews_Reviews_datasets.csv，
    通过56万条数据进行分层抽样11万条数据""")


elif page == "词云":
    st.header("词云")
    st.markdown("""一键生成词云图。""")
    title = st.text_input("请输入您想要的词云标题（仅限英文）:")

    if st.button('点击绘图'):
        df = pd.read_csv('./data/sample_11w.csv')

        # 获取Text列的数据
        reviews = df['Text'].dropna().astype(str).tolist()

        # 生成词云
        wordcloud = generate_word_cloud(reviews, title, max_words=150)
        img = Image.open(io.BytesIO(wordcloud))
        st.image(img, width=800)

elif page == "模型训练结果图":
    st.header('模型训练结果图')
    st.markdown("""一键查看模型训练的可视化结果。""")
    if st.button('点击绘图'):
        # 生成结果图
        result = generate_epoch_metrics()
        img = Image.open(io.BytesIO(result))
        st.image(img, width=800)

elif page == "情感分析":
    st.header('情感分析')
    st.markdown("""输入一段话，即可利用基于LSTM的情感分析模型实现情感分析，""")
    text = st.text_input("请输入一段话（仅限英文）:")

    if st.button('点击预测'):
        if text:
            sentiment, probability = quick_predict(text)  # 调用你的预测函数
            result = f"""
            文本: {text}
            情感: {sentiment}
            概率: {probability:.4f}
            """
            st.text_area("预测结果：", result, height=250)
        else:
            st.warning("请输入文本")
