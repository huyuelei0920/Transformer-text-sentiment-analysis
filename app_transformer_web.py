u"""
基于纯 Transformer 的中文情感分析系统
支持网页推理、数据集选择、网页训练与数据预处理展示
"""

import html
import os
import time
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from train_transformer_web_fixed import run_training, validate_dataset
from tokenizer import ChineseTokenizer


# 页面配置
st.set_page_config(
    page_title="Transformer 中文情感分析系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面样式
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }

    .header-section {
        background: rgba(255, 255, 255, 0.10);
        padding: 2rem;
        border-radius: 22px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.20);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .tech-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00d9ff 0%, #00ff88 100%);
        color: #1a1a2e !important;
        padding: 0.32rem 0.85rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0.2rem;
    }

    .card {
        background: rgba(255,255,255,0.97);
        padding: 1.15rem 1.35rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    .metric-card {
        background: rgba(255,255,255,0.97);
        padding: 1.8rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1rem;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    .small-gap { height: 10px; }
    .mid-gap { height: 16px; }
    .big-gap { height: 24px; }

    .header-section h1,
    .header-section h2,
    .header-section h3,
    .header-section h4,
    .header-section p,
    .header-section span {
        color: #ffffff !important;
    }

    .card,
    .card p,
    .card span,
    .card div,
    .card label,
    .card li,
    .card h1,
    .card h2,
    .card h3,
    .card h4,
    .metric-card,
    .metric-card p,
    .metric-card span,
    .metric-card div,
    .metric-card label,
    .metric-card h1,
    .metric-card h2,
    .metric-card h3,
    .metric-card h4 {
        color: #111827 !important;
    }

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown label,
    .stMarkdown div {
        color: #ffffff;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] button {
        color: rgba(255,255,255,0.78) !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #ff6b6b !important;
    }

    .stTabs [aria-selected="false"] {
        color: rgba(255,255,255,0.72) !important;
    }

    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }

    section[data-testid="stSidebar"] .card,
    section[data-testid="stSidebar"] .card p,
    section[data-testid="stSidebar"] .card span,
    section[data-testid="stSidebar"] .card div,
    section[data-testid="stSidebar"] .card h1,
    section[data-testid="stSidebar"] .card h2,
    section[data-testid="stSidebar"] .card h3,
    section[data-testid="stSidebar"] .card h4,
    section[data-testid="stSidebar"] .card label {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stCaption p,
    section[data-testid="stSidebar"] [data-baseweb="slider"] span,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSlider p,
    section[data-testid="stSidebar"] .stSlider div {
        color: #ffffff !important;
    }

    .stRadio label,
    .stCheckbox label,
    .stSelectbox label,
    .stSlider label,
    .stNumberInput label,
    .stTextInput label,
    .stTextArea label,
    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .stRadio label,
    .stRadio span,
    .stRadio p,
    .stRadio div {
        color: #ffffff !important;
    }

    [data-baseweb="radio"] label,
    [data-baseweb="radio"] span,
    [data-baseweb="radio"] div {
        color: #ffffff !important;
    }

    .stSlider label,
    .stSlider span,
    .stSlider p,
    .stSlider div {
        color: #ffffff !important;
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] input {
        color: #111827 !important;
        background: rgba(255,255,255,0.98) !important;
        border-radius: 12px !important;
    }

    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea {
        color: #111827 !important;
        background: rgba(255,255,255,0.98) !important;
        border-radius: 12px !important;
    }

    .stCheckbox span,
    .stCheckbox p,
    .stCheckbox div {
        color: #ffffff !important;
    }

    .stDataFrame {
        background: rgba(255,255,255,0.98);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    }

    .stDataFrame * {
        color: #111827 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00d9ff 0%, #00ff88 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 999px;
        font-weight: 700;
        padding: 0.75rem 1.3rem;
        box-shadow: 0 6px 16px rgba(0,217,255,0.22);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    [data-testid="stPlotlyChart"] {
        background: rgba(255,255,255,0.97);
        border-radius: 18px;
        padding: 0.7rem;
        overflow: hidden;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    }

    div[data-testid="column"] {
        padding-top: 0.2rem;
    }

    [data-testid="stAlert"] * {
        color: #ffffff !important;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div,
    [data-testid="stMetric"] span {
        color: #ffffff !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.88) !important;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("""
<div class="header-section">
    <h1 style="text-align:center;">🤖 Transformer 中文情感分析系统</h1>
    <p style="text-align:center;font-size:1.1rem;">
        支持网页推理、数据集选择、网页训练与数据预处理展示
    </p>
    <p style="text-align:center;">
        <span class="tech-badge">Transformer Encoder</span>
        <span class="tech-badge">Attention</span>
        <span class="tech-badge">Jieba 分词</span>
    </p>
</div>
""", unsafe_allow_html=True)


# 模型加载
@st.cache_resource
def load_transformer_model(model_dir='transformer_checkpoints'):
    from inference import create_predictor
    return create_predictor(
        model_dir=model_dir,
        model_file='best_model.pt'
    )

def load_transformer_model_with_dir(_hash=None, model_dir='transformer_checkpoints'):
    """带目录参数的模型加载，确保缓存区分不同目录"""
    return load_transformer_model(model_dir)


# 标签映射
label_map = {
    0: {'name': '负面', 'emoji': '😞', 'color': '#FF6B6B'},
    1: {'name': '正面', 'emoji': '😊', 'color': '#4ECDC4'},
    2: {'name': '中性', 'emoji': '😐', 'color': '#95E1D3'}
}


# 基础函数
def model_exists(model_dir='transformer_checkpoints'):
    return os.path.exists(os.path.join(model_dir, 'best_model.pt'))


def section_title(title):
    st.markdown(f'<div class="card"><h3>{title}</h3></div>', unsafe_allow_html=True)


@st.cache_data
@st.cache_data
def discover_csv_files(search_dir='.'):
    csv_files = []
    for path in Path(search_dir).glob('*.csv'):
        csv_files.append(str(path))
    return sorted(csv_files)


@st.cache_data
def load_csv_data(file_path):
    """缓存的CSV数据加载"""
    return pd.read_csv(file_path)


def clean_text_preview(text):
    import re
    text = str(text)
    text = re.sub(r'^\s*\d+[\.、,，]\s*', '', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""\'\'（）【】]', '', text)
    return text


@st.cache_data
def build_preprocess_preview(df, text_column, label_column, use_word=False, max_vocab_size=30000, min_freq=2):
    texts_raw = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()

    texts_clean = [clean_text_preview(x) for x in texts_raw]

    tokenizer = ChineseTokenizer(
        max_vocab_size=max_vocab_size,
        use_word=use_word
    )
    tokenizer.build_vocab(texts_clean, min_freq=min_freq)

    sample_pairs = []
    for raw, clean in list(zip(texts_raw, texts_clean))[:5]:
        tokens = tokenizer._tokenize_text(clean)
        sample_pairs.append({
            "raw": raw,
            "clean": clean,
            "tokens": tokens[:30]
        })

    label_counter = Counter(labels)
    text_lengths = [len(tokenizer._tokenize_text(x)) for x in texts_clean]

    special_tokens = {
        tokenizer.PAD_TOKEN,
        tokenizer.UNK_TOKEN,
        tokenizer.CLS_TOKEN,
        tokenizer.SEP_TOKEN
    }

    vocab_items = []
    for token, idx in tokenizer.vocab.items():
        if token not in special_tokens:
            vocab_items.append(token)

    top_tokens = vocab_items[:20]

    stats = {
        "sample_pairs": sample_pairs,
        "label_counter": label_counter,
        "text_lengths": text_lengths,
        "vocab_size": tokenizer.vocab_size,
        "top_tokens": top_tokens,
    }
    return stats


@st.cache_data
def plot_label_distribution(label_counter):
    labels = list(label_counter.keys())
    counts = list(label_counter.values())

    display_names = []
    colors = []
    mapping = {
        0: ("😞 负面", "#FF6B6B"),
        1: ("😊 正面", "#4ECDC4"),
        2: ("😐 中性", "#95E1D3"),
    }

    for label in labels:
        name, color = mapping.get(label, (str(label), "#888888"))
        display_names.append(name)
        colors.append(color)

    fig = go.Figure(data=[
        go.Bar(
            x=display_names,
            y=counts,
            marker=dict(color=colors),
            text=counts,
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='标签分布',
        xaxis_title='类别',
        yaxis_title='数量',
        height=360,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#111827', size=14),
        margin=dict(l=40, r=30, t=60, b=40),
        showlegend=False
    )
    return fig


@st.cache_data
def plot_length_distribution(lengths):
    fig = go.Figure(data=[
        go.Histogram(
            x=lengths,
            nbinsx=30
        )
    ])

    fig.update_layout(
        title='文本长度分布',
        xaxis_title='分词后长度',
        yaxis_title='样本数',
        height=360,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#111827', size=14),
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig


def analyze_text(model, text, return_attention=False):
    result = model.predict(text, return_all_scores=True, return_attention=return_attention)

    scores = result['all_scores']
    results = []

    for label_id, info in label_map.items():
        results.append({
            'label': label_id,
            'label_name': info['name'],
            'emoji': info['emoji'],
            'color': info['color'],
            'score': scores[info['name']]
        })

    if return_attention:
        return results, result.get('tokens'), result.get('attention_weights')
    return results


# 注意力可视化函数
@st.cache_data
def plot_attention_heatmap(tokens, attention_weights, layer_idx=0):
    import numpy as np

    if not attention_weights or layer_idx >= len(attention_weights):
        return None

    layer_attention = attention_weights[layer_idx]['attention']
    attention_array = np.array(layer_attention)

    n_tokens = len(tokens)
    if attention_array.shape[0] != n_tokens:
        attention_array = attention_array[:n_tokens, :n_tokens]

    fig = go.Figure(data=go.Heatmap(
        z=attention_array,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(
            title='注意力权重',
            tickformat='.2f'
        ),
        hovertemplate='从 "%{y}" 到 "%{x}"<br>权重: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'第 {layer_idx + 1} 层注意力权重热力图',
        xaxis_title='目标位置',
        yaxis_title='源位置',
        height=600,
        xaxis=dict(side='bottom', tickangle=-45),
        yaxis=dict(autorange='reversed'),
        font=dict(size=12, color='#111827'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=30, t=60, b=40)
    )

    return fig


def plot_attention_bar(tokens, attention_weights, layer_idx=0, head_idx=None):
    import numpy as np

    if not attention_weights or layer_idx >= len(attention_weights):
        return None

    layer_attention = np.array(attention_weights[layer_idx]['attention'])
    n_tokens = len(tokens)

    if layer_attention.shape[0] != n_tokens:
        layer_attention = layer_attention[:n_tokens, :n_tokens]

    avg_attention = layer_attention.mean(axis=0)

    indexed_tokens = [f"[{i}] {tok}" for i, tok in enumerate(tokens, start=1)]

    fig = go.Figure(data=[
        go.Bar(
            x=indexed_tokens,
            y=avg_attention,
            marker_color=['#FF6B6B' if a == max(avg_attention) else '#4ECDC4' for a in avg_attention],
            text=[f'{a:.3f}' for a in avg_attention],
            textposition='auto',
            textfont=dict(size=10),
        )
    ])

    fig.update_layout(
        title=f'各Token平均被关注程度（第 {layer_idx + 1} 层）',
        xaxis_title='Token',
        yaxis_title='平均注意力权重',
        height=420,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#111827', size=13),
        margin=dict(l=40, r=30, t=60, b=80)
    )

    return fig


def plot_attention_connection(tokens, attention_weights, layer_idx=0, threshold_value=0.1):
    import numpy as np

    if not attention_weights or layer_idx >= len(attention_weights):
        return None

    layer_attention = np.array(attention_weights[layer_idx]['attention'])
    n_tokens = len(tokens)

    if layer_attention.shape[0] != n_tokens:
        layer_attention = layer_attention[:n_tokens, :n_tokens]

    angles = np.linspace(0, 2 * np.pi, n_tokens, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    fig = go.Figure()

    connection_count = 0
    max_attention = 0

    for i in range(n_tokens):
        for j in range(n_tokens):
            if layer_attention[i, j] > threshold_value:
                connection_count += 1
                max_attention = max(max_attention, layer_attention[i, j])

    for i in range(n_tokens):
        for j in range(n_tokens):
            if layer_attention[i, j] > threshold_value:
                attention_value = layer_attention[i, j]

                if attention_value > 0.5:
                    r = 255
                    g = int(255 - (attention_value - 0.5) * 200)
                    b = 0
                else:
                    r = int(attention_value * 200)
                    g = int(200 + attention_value * 55)
                    b = int(255 - attention_value * 100)

                color = f'rgb({r}, {g}, {b})'
                line_width = max(1, min(8, attention_value * 10))

                fig.add_trace(go.Scatter(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    hoverinfo='text',
                    hovertext=f'{tokens[i]} → {tokens[j]}<br>权重: {attention_value:.3f}',
                    showlegend=False
                ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(
            size=25,
            color=['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(n_tokens)],
            line=dict(color='white', width=3)
        ),
        text=tokens,
        textposition='top center',
        textfont=dict(size=13, color='white', family='Arial Black'),
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))

    fig.update_layout(
        title=dict(
            text=f'第 {layer_idx + 1} 层注意力连接图<br><sub style="color:white">阈值: {threshold_value:.2f} | 连接数: {connection_count} | 最大权重: {max_attention:.3f}</sub>',
            font=dict(color='white', size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=550,
        showlegend=False,
        plot_bgcolor='rgba(26,26,46,0.8)',
        paper_bgcolor='rgba(26,26,46,0.8)',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def plot_multi_layer_attention(tokens, attention_weights):
    import numpy as np

    if not attention_weights:
        return None

    num_layers = len(attention_weights)
    n_tokens = len(tokens)

    layer_avg_attention = []
    for layer_idx in range(num_layers):
        layer_attention = np.array(attention_weights[layer_idx]['attention'])
        if layer_attention.shape[0] != n_tokens:
            layer_attention = layer_attention[:n_tokens, :n_tokens]
        avg_attention = layer_attention.mean(axis=0)
        layer_avg_attention.append(avg_attention)

    fig = go.Figure()

    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F9CA24', '#6C5CE7', '#00D9FF', '#00FF88', '#FF9FF3']

    for layer_idx, avg_attention in enumerate(layer_avg_attention):
        fig.add_trace(go.Bar(
            name=f'第 {layer_idx + 1} 层',
            x=tokens,
            y=avg_attention,
            marker_color=colors[layer_idx % len(colors)],
            opacity=0.7
        ))

    fig.update_layout(
        title={
            'text': '多层注意力分布对比',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=24, color='#111827')
        },
        xaxis=dict(
            title=dict(
                text='Token',
                font=dict(size=18, color='#111827')
            ),
            tickfont=dict(size=13, color='#111827'),
            title_standoff=18
        ),
        yaxis=dict(
            title=dict(
                text='平均注意力权重',
                font=dict(size=18, color='#111827')
            ),
            tickfont=dict(size=13, color='#111827'),
            gridcolor='rgba(0,0,0,0.12)',
            zerolinecolor='rgba(0,0,0,0.12)'
        ),
        barmode='group',
        height=420,
        font=dict(
            family='Arial, Microsoft YaHei, sans-serif',
            size=14,
            color='#111827'
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color='#111827')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=30, r=30, t=70, b=130)
    )

    return fig


@st.cache_data
def plot_training_history(history):
    if not history or len(history.get('train_loss', [])) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['train_loss'], mode='lines+markers', name='训练损失'))
    fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines+markers', name='验证损失'))
    fig.add_trace(go.Scatter(y=history['train_acc'], mode='lines+markers', name='训练准确率'))
    fig.add_trace(go.Scatter(y=history['val_acc'], mode='lines+markers', name='验证准确率'))
    fig.add_trace(go.Scatter(y=history['val_f1'], mode='lines+markers', name='验证F1'))

    fig.update_layout(
        title='训练过程曲线',
        height=460,
        xaxis_title='Epoch',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#111827', size=14),
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig


# 侧边栏
with st.sidebar:
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top:0;">📊 系统状态</h3>
        <p><strong>运行设备：</strong>{device_name}</p>
        <p><strong>自研 Transformer：</strong>{'已找到' if model_exists() else '未找到'}</p>
        <p><strong>任务：</strong>中文情感三分类</p>
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "注意力阈值（当前连接图会用到）",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.005,
        key="attention_threshold"
    )
    st.caption(f"当前值: {threshold:.3f}")


# 标签页
tab_predict, tab_batch, tab_train = st.tabs(["🔍 模型推理", "📊 批量预测", "🏋️ 模型训练"])


# 推理页
with tab_predict:
    model_dir = st.text_input(
        '模型目录',
        value='transformer_checkpoints_jieba',
        help='指定训练好的模型目录'
    )

    model = None
    model_loaded = False

    try:
        with st.spinner('🔄 正在加载模型，请稍候...'):
            if model_exists(model_dir):
                model = load_transformer_model_with_dir(model_dir=model_dir)
                model_loaded = True
            else:
                st.warning(f'⚠️ 模型目录「{model_dir}」中未找到模型')

        if model_loaded:
            st.success(f'✅ 模型加载成功（{model_dir}）')
    except Exception as e:
        st.error(f'❌ 模型加载失败: {str(e)}')
        model_loaded = False

    col1, col2 = st.columns([3, 2])

    def set_example_text(example_text):
        st.session_state['pending_text'] = example_text

    if 'pending_text' in st.session_state:
        st.session_state['text_input'] = st.session_state['pending_text']
        del st.session_state['pending_text']

    with col1:
        section_title('📝 输入文本')
        user_input = st.text_area(
            "",
            height=200,
            placeholder="请输入要分析的中文文本...",
            label_visibility="collapsed",
            key="text_input"
        )

    with col2:
        section_title('🎯 快捷操作')

        analyze_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

        if analyze_button and model_loaded:
            if user_input.strip():
                with st.spinner('🔍 正在分析...'):
                    time.sleep(0.3)
                    results, tokens, attention_weights = analyze_text(
                        model, user_input, return_attention=True
                    )
                    st.session_state['results'] = results
                    st.session_state['tokens'] = tokens
                    st.session_state['attention_weights'] = attention_weights
                    st.session_state['analyzed'] = True
            else:
                st.warning("⚠️ 请输入文本后再进行分析")
        elif analyze_button and not model_loaded:
            st.error("❌ 模型未加载，无法分析")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h4 style="margin-bottom: 1rem;">💡 示例文本</h4>
        </div>
        """, unsafe_allow_html=True)

        sample_texts = [
            "今天天气真好，心情很愉快",
            "这个产品质量太差了，我很失望",
            "这个产品还可以，没什么特别的"
        ]

        for i, text in enumerate(sample_texts):
            if st.button(f"示例 {i+1}: {text[:15]}...", key=f"sample_{i}", use_container_width=True):
                set_example_text(text)
                st.rerun()

    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        results = st.session_state['results']

        st.markdown("---")
        section_title('📊 分析结果')

        best_result = max(results, key=lambda x: x['score'])

        st.markdown(f"""
        <div class="metric-card" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 5rem; margin: 0;">{best_result['emoji']}</h1>
            <h2 style="margin: 1rem 0; color: #333;">{best_result['label_name']}</h2>
            <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <p style="font-size: 2.5rem; font-weight: bold; margin: 0; color: {best_result['color']};">
                    {best_result['score']:.2%}
                </p>
                <p style="font-size: 1rem; color: #666; margin: 0.5rem 0 0 0;">置信度</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        colv1, colv2 = st.columns(2)

        with colv1:
            section_title('📈 置信度对比')

            labels = [f"{r['emoji']} {r['label_name']}" for r in results]
            scores = [r['score'] for r in results]
            colors = [r['color'] for r in results]

            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=scores,
                    marker=dict(
                        color=colors,
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{s:.2%}' for s in scores],
                    textposition='auto',
                    textfont=dict(size=16, color='white', family='Arial'),
                )
            ])

            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=60),
                xaxis=dict(
                    title=dict(text='', font=dict(size=14, color='#666')),
                    tickfont=dict(size=12, color='#666')
                ),
                yaxis=dict(
                    title=dict(text='置信度', font=dict(size=14, color='#666')),
                    range=[0, 1],
                    tickfont=dict(size=12, color='#666'),
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                font=dict(color='#111827')
            )

            st.plotly_chart(fig, use_container_width=True)

        with colv2:
            section_title('🥧 情感分布')

            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=scores,
                marker=dict(colors=colors, line=dict(color='white', width=3)),
                textinfo='label+percent',
                textfont=dict(size=14, color='white'),
                hoverinfo='label+percent+value',
                pull=[0.05 if s == max(scores) else 0 for s in scores]
            )])

            fig_pie.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color='#666')
                ),
                font=dict(color='#111827')
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        section_title('📋 详细数据')

        df = pd.DataFrame([
            {
                '情感类别': f"{r['emoji']} {r['label_name']}",
                '标签': r['label'],
                '置信度': f"{r['score']:.4f}",
                '百分比': f"{r['score']:.2%}"
            }
            for r in results
        ])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '情感类别': st.column_config.TextColumn('情感类别', width='large'),
                '标签': st.column_config.TextColumn('标签', width='medium'),
                '置信度': st.column_config.TextColumn('置信度', width='medium'),
                '百分比': st.column_config.TextColumn('百分比', width='medium')
            }
        )

        tokens = st.session_state.get('tokens')
        attention_weights = st.session_state.get('attention_weights')

        if tokens and attention_weights:
            st.markdown("---")
            section_title('🧠 注意力可视化')

            st.markdown("""
            <div class="card">
                <p style="color: #666; margin-bottom: 1rem;">
                    Transformer 模型通过多头注意力机制关注输入文本的不同部分。下面的可视化展示了模型在处理文本时各 token 之间的注意力关系。
                </p>
            </div>
            """, unsafe_allow_html=True)

            section_title('📝 分词结果')

            token_html = '<div style="display:flex; flex-wrap:wrap; gap:10px; padding:1rem; background:white; border-radius:10px; margin-bottom:1.5rem;">'
            for i, token in enumerate(tokens, start=1):
                safe_token = html.escape(str(token))
                token_html += (
                    f'<div style="background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%); '
                    f'color: #1a1a2e; padding: 8px 12px; border-radius: 8px; font-weight: bold; '
                    f'box-shadow: 0 2px 8px rgba(0,217,255,0.3); display: inline-block;">'
                    f'<span style="font-size: 0.7rem; color: #666;">[{i}]</span> {safe_token}'
                    f'</div>'
                )
            token_html += '</div>'
            st.markdown(token_html, unsafe_allow_html=True)

            num_layers = len(attention_weights)
            col_layer, col_viz_type = st.columns(2)

            with col_layer:
                section_title('🔢 选择编码器层')
                layer_idx = st.select_slider(
                    f"选择要查看的层 (共 {num_layers} 层)",
                    options=list(range(num_layers)),
                    value=0,
                    key="layer_selector"
                )

            with col_viz_type:
                section_title('📊 可视化类型')
                viz_type = st.radio(
                    "选择可视化方式",
                    ["热力图", "条形图", "连接图"],
                    horizontal=True,
                    key="viz_type_selector"
                )

            if viz_type == "热力图":
                section_title('🔥 注意力热力图')
                fig_heatmap = plot_attention_heatmap(tokens, attention_weights, layer_idx)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)

            elif viz_type == "条形图":
                section_title('📊 Token 关注度分布')
                fig_bar = plot_attention_bar(tokens, attention_weights, layer_idx)
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)

            elif viz_type == "连接图":
                section_title('🕸️ 注意力连接图')
                fig_connection = plot_attention_connection(tokens, attention_weights, layer_idx, threshold_value=threshold)
                if fig_connection:
                    st.plotly_chart(fig_connection, use_container_width=True)

            section_title('📈 多层注意力对比')
            fig_layers = plot_multi_layer_attention(tokens, attention_weights)
            if fig_layers:
                st.plotly_chart(fig_layers, use_container_width=True)


# 批量预测页
with tab_batch:
    section_title('📁 模型加载')

    batch_model_dir = st.text_input(
        '模型目录',
        value='transformer_checkpoints_jieba',
        key='batch_model_dir',
        help='指定训练好的模型目录'
    )

    batch_model = None
    batch_model_loaded = False

    try:
        with st.spinner('🔄 正在加载模型，请稍候...'):
            if model_exists(batch_model_dir):
                batch_model = load_transformer_model_with_dir(model_dir=batch_model_dir)
                batch_model_loaded = True
            else:
                st.warning(f'⚠️ 模型目录「{batch_model_dir}」中未找到模型')

        if batch_model_loaded:
            st.success(f'✅ 模型加载成功（{batch_model_dir}）')
    except Exception as e:
        st.error(f'❌ 模型加载失败: {str(e)}')
        batch_model_loaded = False

    if batch_model_loaded:
        st.markdown("---")
        section_title('📤 批量输入')

        input_mode = st.radio(
            '选择输入方式',
            ['粘贴文本', '上传文件'],
            horizontal=True
        )

        batch_texts = []
        df_input = None

        if input_mode == '粘贴文本':
            raw_text = st.text_area(
                '',
                height=200,
                placeholder='请粘贴多条文本，每行一条...',
                label_visibility="collapsed",
                help='每行一条文本，将自动跳过空行'
            )

            if raw_text.strip():
                batch_texts = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]

        else:
            uploaded_file = st.file_uploader(
                '上传文件 (CSV/TXT)',
                type=['csv', 'txt'],
                help='CSV文件需要包含文本列；TXT文件每行一条文本'
            )

            if uploaded_file is not None:
                file_type = uploaded_file.name.split('.')[-1].lower()

                if file_type == 'csv':
                    df_input = pd.read_csv(uploaded_file)
                    columns = df_input.columns.tolist()

                    default_col_idx = 0
                    if 'sentence' in columns:
                        default_col_idx = columns.index('sentence')
                    elif 'text' in columns:
                        default_col_idx = columns.index('text')

                    text_col = st.selectbox('选择文本列', columns, index=default_col_idx)
                    batch_texts = df_input[text_col].astype(str).tolist()

                    st.markdown('**文件预览（前5行）**')
                    st.dataframe(df_input.head(), use_container_width=True, hide_index=True)
                else:  # txt
                    raw_text = uploaded_file.read().decode('utf-8')
                    batch_texts = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        if batch_texts:
            col_batch1, col_batch2, col_batch3 = st.columns(3)
            with col_batch1:
                st.metric("待预测文本数", len(batch_texts))
            with col_batch2:
                total_chars = sum(len(text) for text in batch_texts)
                st.metric("总字符数", f"{total_chars:,}")
            with col_batch3:
                avg_len = total_chars // len(batch_texts) if batch_texts else 0
                st.metric("平均长度", f"{avg_len} 字")

            st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

            section_title('📋 待预测文本预览')
            preview_count = min(10, len(batch_texts))
            for i in range(preview_count):
                text_display = batch_texts[i][:100] + '...' if len(batch_texts[i]) > 100 else batch_texts[i]
                st.markdown(
                    f"""
<div class="card" style="padding:0.6rem 1rem;">
    <p style="margin:0; font-size:0.9rem;"><strong>{i+1}.</strong> {html.escape(text_display)}</p>
</div>
""",
                    unsafe_allow_html=True
                )
            if len(batch_texts) > preview_count:
                st.caption(f'... 还有 {len(batch_texts) - preview_count} 条文本')

            st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

            section_title('🚀 开始批量预测')
            batch_predict_btn = st.button('🚀 开始批量预测', type='primary', use_container_width=True)

            if batch_predict_btn:
                with st.spinner('🔍 正在批量预测中，请稍候...'):
                    batch_results = []
                    for text in batch_texts:
                        result = batch_model.predict(text, return_all_scores=True)
                        scores = result['all_scores']
                        best_label = max(label_map.keys(), key=lambda x: scores[label_map[x]['name']])
                        batch_results.append({
                            'text': text,
                            'label': best_label,
                            'label_name': label_map[best_label]['name'],
                            'emoji': label_map[best_label]['emoji'],
                            'negative_score': scores['负面'],
                            'positive_score': scores['正面'],
                            'neutral_score': scores['中性'],
                            'confidence': scores[label_map[best_label]['name']]
                        })

                    st.session_state['batch_results'] = batch_results
                    st.success(f'✅ 预测完成！共处理 {len(batch_texts)} 条文本')

        if 'batch_results' in st.session_state:
            batch_results = st.session_state['batch_results']

            st.markdown("---")
            section_title('📊 批量预测结果')

            # 统计概览
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            label_counts = Counter([r['label'] for r in batch_results])
            with col_stat1:
                pos_count = label_counts.get(1, 0)
                st.metric("😊 正面", f"{pos_count} ({pos_count/len(batch_results):.1%})")
            with col_stat2:
                neg_count = label_counts.get(0, 0)
                st.metric("😞 负面", f"{neg_count} ({neg_count/len(batch_results):.1%})")
            with col_stat3:
                neu_count = label_counts.get(2, 0)
                st.metric("😐 中性", f"{neu_count} ({neu_count/len(batch_results):.1%})")

            st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

            # 详细结果表格
            section_title('📋 详细预测结果')

            df_results = pd.DataFrame(batch_results)
            df_display = df_results[['emoji', 'text', 'label_name', 'confidence']].copy()
            df_display.columns = ['情感', '文本', '类别', '置信度']
            df_display['置信度'] = df_display['置信度'].apply(lambda x: f"{x:.2%}")
            df_display['文本'] = df_display['文本'].str.slice(0, 50) + '...'

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    '情感': st.column_config.TextColumn('情感', width='small'),
                    '文本': st.column_config.TextColumn('文本', width='large'),
                    '类别': st.column_config.TextColumn('类别', width='small'),
                    '置信度': st.column_config.TextColumn('置信度', width='small'),
                }
            )

            # 导出功能
            st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                # CSV导出
                df_export = df_results[['text', 'label', 'label_name', 'negative_score',
                                     'positive_score', 'neutral_score', 'confidence']].copy()
                df_export.columns = ['文本', '标签', '类别', '负面得分', '正面得分', '中性得分', '置信度']

                csv = df_export.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label='📥 导出CSV',
                    data=csv,
                    file_name='batch_predictions.csv',
                    mime='text/csv',
                    use_container_width=True
                )

            with col_export2:
                # 简要统计CSV
                stats_csv = pd.DataFrame([
                    {'类别': '正面', '数量': label_counts.get(1, 0), '占比': f"{label_counts.get(1, 0)/len(batch_results):.2%}"},
                    {'类别': '负面', '数量': label_counts.get(0, 0), '占比': f"{label_counts.get(0, 0)/len(batch_results):.2%}"},
                    {'类别': '中性', '数量': label_counts.get(2, 0), '占比': f"{label_counts.get(2, 0)/len(batch_results):.2%}"},
                ])
                stats_csv_csv = stats_csv.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label='📥 导出统计CSV',
                    data=stats_csv_csv,
                    file_name='batch_statistics.csv',
                    mime='text/csv',
                    use_container_width=True
                )

            # 置信度分布图
            st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)
            section_title('📈 置信度分布')

            fig_conf = go.Figure()
            for label_id, label_info in label_map.items():
                scores = [r['confidence'] for r in batch_results if r['label'] == label_id]
                fig_conf.add_trace(go.Histogram(
                    name=f"{label_info['emoji']} {label_info['name']}",
                    x=scores,
                    marker_color=label_info['color'],
                    opacity=0.7,
                    nbinsx=20
                ))

            fig_conf.update_layout(
                barmode='overlay',
                title='各情感类别置信度分布',
                xaxis_title='置信度',
                yaxis_title='数量',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#111827'),
                margin=dict(l=40, r=30, t=60, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info('请确保模型目录中有训练好的模型。')


# 训练页
with tab_train:
    section_title('📁 数据集选择')

    dataset_mode = st.radio('选择数据来源', ['本地 CSV', '上传 CSV'], horizontal=True)

    train_path = None
    train_df = None
    val_path = None
    val_df = None

    # 训练集选择
    section_title('📄 训练集')
    if dataset_mode == '本地 CSV':
        csv_files = discover_csv_files('.')
        if csv_files:
            default_index = 0
            if 'train.csv' in csv_files:
                default_index = csv_files.index('train.csv')
            elif 'data.csv' in csv_files:
                default_index = csv_files.index('data.csv')
            train_path = st.selectbox('选择训练集', csv_files, index=default_index)
            train_df = load_csv_data(train_path)
        else:
            st.warning('当前目录下没有找到 CSV 文件。')
    else:
        uploaded_train = st.file_uploader('上传训练集', type=['csv'], key='train_upload')
        if uploaded_train is not None:
            train_df = pd.read_csv(uploaded_train)
            os.makedirs('uploaded_datasets', exist_ok=True)
            train_path = os.path.join('uploaded_datasets', uploaded_train.name)
            train_df.to_csv(train_path, index=False, encoding='utf-8-sig')

    # 验证集选择
    section_title('📋 验证集')
    if dataset_mode == '本地 CSV':
        if csv_files:
            default_val_index = 0
            if 'validation.csv' in csv_files:
                default_val_index = csv_files.index('validation.csv')
            elif 'val.csv' in csv_files:
                default_val_index = csv_files.index('val.csv')
            val_path = st.selectbox('选择验证集', csv_files, index=default_val_index)
            val_df = load_csv_data(val_path)
    else:
        uploaded_val = st.file_uploader('上传验证集', type=['csv'], key='val_upload')
        if uploaded_val is not None:
            val_df = pd.read_csv(uploaded_val)
            os.makedirs('uploaded_datasets', exist_ok=True)
            val_path = os.path.join('uploaded_datasets', uploaded_val.name)
            val_df.to_csv(val_path, index=False, encoding='utf-8-sig')

    # 数据预览（显示训练集）
    preview_df = train_df

    st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

    if preview_df is not None:
        columns = preview_df.columns.tolist()
        default_text_idx = columns.index('sentence') if 'sentence' in columns else 0
        default_label_idx = columns.index('label') if 'label' in columns else min(1, len(columns) - 1)

        preview_text_col = columns[default_text_idx]
        preview_label_col = columns[default_label_idx]

        section_title('📄 数据预览')

        preview_show = preview_df[[preview_text_col, preview_label_col]].head(10).copy()
        preview_show[preview_text_col] = preview_show[preview_text_col].astype(str).apply(
            lambda x: x if len(x) <= 48 else x[:48] + '...'
        )

        st.dataframe(
            preview_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                preview_text_col: st.column_config.TextColumn(preview_text_col, width='large'),
                preview_label_col: st.column_config.TextColumn(preview_label_col, width='small'),
            }
        )

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        section_title('🧾 样本文本预览')
        for _, row in preview_df[[preview_text_col, preview_label_col]].head(5).iterrows():
            st.markdown(
                f"""
<div class="card" style="padding:0.8rem 1rem;">
    <p style="margin:0; color:#111827 !important; word-break: break-word;"><strong>文本：</strong>{html.escape(str(row[preview_text_col]))}</p>
    <p style="margin:0.45rem 0 0 0; color:#111827 !important;"><strong>标签：</strong>{row[preview_label_col]}</p>
</div>
""",
                unsafe_allow_html=True
            )

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        section_title('🧩 字段映射')
        col_a, col_b = st.columns(2)
        with col_a:
            text_column = st.selectbox('文本列', columns, index=default_text_idx)
        with col_b:
            label_column = st.selectbox('标签列', columns, index=default_label_idx)

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        section_title('🧪 数据预处理展示')

        preprocess_use_word = st.checkbox(
            '预处理展示时使用词级分词（仅预览）',
            value=True,
            key='preprocess_use_word_preview'
        )

        preprocess_stats = build_preprocess_preview(
            preview_df,
            text_column=text_column,
            label_column=label_column,
            use_word=preprocess_use_word,
            max_vocab_size=50000,
            min_freq=1
        )

        st.markdown("### 清洗前后与分词结果预览")

        for idx, item in enumerate(preprocess_stats["sample_pairs"], start=1):
            token_text = " / ".join(item["tokens"]) if item["tokens"] else "(空)"
            st.markdown(
                f"""
<div class="card">
    <h4>样本 {idx}</h4>
    <p><strong>原文本：</strong>{html.escape(item["raw"])}</p>
    <p><strong>清洗后：</strong>{html.escape(item["clean"])}</p>
    <p><strong>分词结果：</strong>{html.escape(token_text)}</p>
</div>
""",
                unsafe_allow_html=True
            )

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        col_stats1, col_stats2 = st.columns(2)

        with col_stats1:
            section_title('🏷️ 标签分布')
            st.plotly_chart(
                plot_label_distribution(preprocess_stats["label_counter"]),
                use_container_width=True
            )

        with col_stats2:
            section_title('📏 文本长度分布')
            st.plotly_chart(
                plot_length_distribution(preprocess_stats["text_lengths"]),
                use_container_width=True
            )

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        col_vocab1, col_vocab2 = st.columns([1, 2])
        with col_vocab1:
            st.metric("词表大小", preprocess_stats["vocab_size"])
        with col_vocab2:
            section_title('🔤 高频 Token 预览')
            token_preview = " / ".join(preprocess_stats["top_tokens"]) if preprocess_stats["top_tokens"] else "(无)"
            st.markdown(
                f"""
<div class="card">
    <p style="word-break: break-word;"><strong>Top Tokens：</strong>{html.escape(token_preview)}</p>
</div>
""",
                unsafe_allow_html=True
            )

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        section_title('⚙️ 训练参数')

        c1, c2 = st.columns(2)

        with c1:
            batch_size = st.number_input('Batch Size', min_value=4, max_value=256, value=16, step=4)
            num_epochs = st.number_input('训练轮数', min_value=1, max_value=200, value=30, step=1)
            learning_rate = st.number_input(
                '学习率',
                min_value=0.000001,
                max_value=0.01,
                value=0.0001,
                format='%.6f',
                help='推荐值：5e-5 到 2e-4 之间，梯度爆炸时建议降低学习率',
            )
            early_stopping_patience = st.number_input(
                '早停耐心值',
                min_value=1,
                max_value=50,
                value=8,
                step=1,
            )

        with c2:
            max_length = st.number_input('最大序列长度', min_value=16, max_value=512, value=128, step=16)
            max_vocab_size = st.number_input('最大词表大小', min_value=1000, max_value=50000, value=50000, step=1000)
            min_freq = st.number_input('最小词频', min_value=1, max_value=20, value=1, step=1)
            use_word = st.checkbox('使用词级分词（需要 jieba）', value=True)
            save_dir = st.text_input('模型保存目录', value='transformer_checkpoints_jieba')

        st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

        section_title('🚀 开始训练')
        start_train = st.button('🚀 开始训练', type='primary', use_container_width=True)

        if start_train:
            try:
                # 检查是否选择了训练集和验证集
                if train_path is None:
                    st.error('请选择训练集')
                    st.stop()
                if val_path is None:
                    st.error('请选择验证集')
                    st.stop()

                validate_dataset(train_df, text_column, label_column)
                validate_dataset(val_df, text_column, label_column)

                progress_bar = st.progress(0)
                log_placeholder = st.empty()
                logs = []

                def progress_callback(progress):
                    progress_bar.progress(max(0.0, min(1.0, float(progress))))

                def log_callback(message):
                    logs.append(message)
                    log_placeholder.text_area(
                        "训练日志",
                        value='\n\n'.join(logs[-20:]),
                        height=280,
                        disabled=True
                    )

                with st.spinner('训练中，请耐心等待...'):
                    config = {
                        'train_data_path': train_path,
                        'val_data_path': val_path,
                        'text_column': text_column,
                        'label_column': label_column,
                        'd_model': 128,
                        'num_heads': 2,
                        'num_layers': 2,
                        'd_ff': 256,
                        'dropout': 0.1,
                        'max_length': int(max_length),
                        'batch_size': int(batch_size),
                        'num_epochs': int(num_epochs),
                        'learning_rate': float(learning_rate),
                        'weight_decay': 0.01,
                        'label_smoothing': 0.1,
                        'early_stopping_patience': int(early_stopping_patience),
                        'max_vocab_size': int(max_vocab_size),
                        'min_freq': int(min_freq),
                        'use_word': bool(use_word),
                        'save_dir': save_dir,
                    }
                    result = run_training(
                        config,
                        progress_callback=progress_callback,
                        log_callback=log_callback,
                    )

                st.success('训练完成')

                st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

                # 显示验证集评估结果
                if 'val_acc' in result and 'val_f1' in result:
                    st.markdown("""
                    <div class="card">
                        <h3>📊 验证集评估结果</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    col_test1, col_test2 = st.columns(2)
                    with col_test1:
                        st.metric("验证集准确率", f"{result['val_acc']:.4f}")
                    with col_test2:
                        st.metric("验证集F1分数", f"{result['val_f1']:.4f}")

                    cm_path = result.get('confusion_matrix_path')
                    if cm_path and os.path.isfile(cm_path):
                        st.markdown("**混淆矩阵（验证集）**")
                        st.image(cm_path, use_container_width=True)

                st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric('最佳验证准确率', f"{result['best_val_acc']:.4f}")
                with m2:
                    st.metric('最佳验证F1', f"{result['best_val_f1']:.4f}")

                st.markdown('<div class="small-gap"></div>', unsafe_allow_html=True)

                section_title('📈 训练曲线')
                fig_hist = plot_training_history(result['history'])
                if fig_hist is not None:
                    st.plotly_chart(fig_hist, use_container_width=True)

                load_transformer_model.clear()
                st.info('模型缓存已清理。切回「模型推理」页时会重新加载最新权重。')

            except Exception as e:
                st.error(f'训练失败: {e}')
    else:
        st.info('先选择或上传一个 CSV 数据集。')
