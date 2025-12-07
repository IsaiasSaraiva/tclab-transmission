import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import heapq
from io import StringIO, BytesIO
import time
import numpy as np
import json

# ====================================================
# Huffman (opera sobre BYTES)
# ====================================================
class HuffmanNode:
    def __init__(self, byte_val, freq):
        self.byte_val = byte_val
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree_from_bytes(data_bytes: bytes):
    freq = Counter(data_bytes)
    heap = [HuffmanNode(b, f) for b, f in freq.items()]
    heapq.heapify(heap)
    if not heap:
        return None
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]


def build_codes_from_tree(root):
    codes = {}
    def walk(node, prefix):
        if node is None:
            return
        if node.byte_val is not None:
            codes[node.byte_val] = prefix or "0"
            return
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")
    walk(root, "")
    return codes


def huffman_compress_bytes(data_bytes: bytes):
    start = time.perf_counter()
    root = build_huffman_tree_from_bytes(data_bytes)
    if root is None:
        return "", {}, 0.0
    codes = build_codes_from_tree(root)
    encoded_bits = "".join(codes[b] for b in data_bytes)
    comp_time = time.perf_counter() - start
    return encoded_bits, codes, comp_time


def huffman_decompress_bits(encoded_bits: str, codes: dict):
    start = time.perf_counter()
    reverse_map = {v: k for k, v in codes.items()}
    decoded_bytes = bytearray()
    cur = ""
    for bit in encoded_bits:
        cur += bit
        if cur in reverse_map:
            decoded_bytes.append(reverse_map[cur])
            cur = ""
    dec_time = time.perf_counter() - start
    return bytes(decoded_bytes), dec_time


# ====================================================
# Helpers
# ====================================================
def downsample_df_for_plot(df: pd.DataFrame, x_col: str, max_points: int = 3000):
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


# ====================================================
# Streamlit UI
# ====================================================
st.set_page_config(
    page_title="TCLab — Transmissão Ruidosa + Huffman",
    layout="wide",
    page_icon="📡"
)

st.title("📡 TCLab — Transmissão Ruidosa + Detecção/Correção de Erros + Compressão")

st.markdown("""
### Sistema Integrado de Transmissão e Compressão
Este sistema simula:
1. **Transmissão multi-canal** com ruídos independentes (T1, T2, Q1, Q2)
2. **Detecção de erros** via CRC-16
3. **Correção de erros** via Código de Hamming (7,4)
4. **Compressão de dados** via Huffman
5. **Visualização interativa** com análise de erros

📂 Envie o CSV gerado pelo script `tclab_transmission_simulation.py`
""")

# Upload
uploaded_file = st.file_uploader(
    "📂 Envie o arquivo CSV da transmissão TCLab:",
    type=["csv"],
    help="Arquivo gerado por tclab_transmission_simulation.py"
)

if uploaded_file is None:
    st.info("👆 Envie um CSV para começar a análise")
    
    # Exemplo de como gerar os dados
    with st.expander("🔧 Como gerar os dados?"):
        st.code("""
# Instale as dependências:
pip install tclab numpy

# Execute a simulação:
python tclab_transmission_simulation.py --duration-days 0.1 --noise-level medium

# Níveis de ruído disponíveis: low, medium, high, extreme
        """, language="bash")
    
    st.stop()

# ====================================================
# PROCESSAMENTO
# ====================================================
file_bytes = uploaded_file.read()
original_size_bytes = len(file_bytes)

# Cache de compressão
@st.cache_data(show_spinner=False)
def compress_decompress_cached(file_bytes_blob: bytes):
    encoded_bits, codes_map, comp_time = huffman_compress_bytes(file_bytes_blob)
    compressed_bytes_est = (len(encoded_bits) + 7) // 8
    decoded_bytes, dec_time = huffman_decompress_bits(encoded_bits, codes_map)
    return {
        "encoded_bits": encoded_bits,
        "codes_map": codes_map,
        "comp_time": comp_time,
        "dec_time": dec_time,
        "compressed_bytes_est": compressed_bytes_est,
        "decoded_bytes": decoded_bytes
    }

with st.spinner("🔧 Processando compressão Huffman..."):
    comp_info = compress_decompress_cached(file_bytes)

# Verificação
decoded_bytes = comp_info["decoded_bytes"]
if decoded_bytes != file_bytes:
    st.warning("⚠️ Aviso: conteúdo descomprimido difere do original")

compression_ratio = 100.0 * (1 - comp_info["compressed_bytes_est"] / max(1, original_size_bytes))

# Lê DataFrame
try:
    text = decoded_bytes.decode("utf-8")
except UnicodeDecodeError:
    text = decoded_bytes.decode("latin1")

df = pd.read_csv(StringIO(text))

# Validação de colunas
required_cols = ["Time (s)", "T1", "T2", "Q1", "Q2"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"❌ Colunas faltando: {missing_cols}")
    st.stop()

# Converte para numérico
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Verifica se há dados de erro (gerados pela simulação)
has_error_data = all(col in df.columns for col in ["T1_real", "T2_real", "T1_error", "T2_error"])

# ====================================================
# SIDEBAR: Configurações
# ====================================================
st.sidebar.header("⚙️ Configurações")

# Compressão
st.sidebar.subheader("📦 Compressão Huffman")
st.sidebar.metric("Tamanho original", f"{original_size_bytes:,} bytes")
st.sidebar.metric("Tamanho comprimido", f"{comp_info['compressed_bytes_est']:,} bytes")
st.sidebar.metric("Redução", f"{compression_ratio:.2f}%")
st.sidebar.metric("Tempo compressão", f"{comp_info['comp_time']:.3f} s")
st.sidebar.metric("Tempo descompressão", f"{comp_info['dec_time']:.3f} s")

st.sidebar.markdown("---")

# Performance
st.sidebar.subheader("📊 Visualização")
max_points = st.sidebar.number_input(
    "Máx pontos (downsample)",
    min_value=500,
    max_value=50000,
    value=3000,
    step=500
)
auto_downsample = st.sidebar.checkbox("Downsample automático", value=True)

st.sidebar.markdown("---")

# Setpoints
st.sidebar.subheader("🎯 Setpoints")
if "SP1" in df.columns and "SP2" in df.columns:
    sp1_vals = df["SP1"].unique()
    sp2_vals = df["SP2"].unique()
    st.sidebar.info(f"SP1: {sp1_vals}")
    st.sidebar.info(f"SP2: {sp2_vals}")

t1_min, t1_max = float(df["T1"].min()), float(df["T1"].max())
t2_min, t2_max = float(df["T2"].min()), float(df["T2"].max())
pad1 = max(1.0, (t1_max - t1_min) * 0.1)
pad2 = max(1.0, (t2_max - t2_min) * 0.1)

T1_setpoint = st.sidebar.slider(
    "T1 setpoint",
    min_value=(t1_min - pad1),
    max_value=(t1_max + pad1),
    value=float((t1_min + t1_max) / 2),
    step=0.1
)
T2_setpoint = st.sidebar.slider(
    "T2 setpoint",
    min_value=(t2_min - pad2),
    max_value=(t2_max + pad2),
    value=float((t2_min + t2_max) / 2),
    step=0.1
)

st.sidebar.markdown("---")

# Filtro temporal
st.sidebar.subheader("⏱️ Filtro Time (s)")
min_time_s = int(df["Time (s)"].min())
max_time_s = int(df["Time (s)"].max())
time_range = st.sidebar.slider(
    "Intervalo",
    min_value=min_time_s,
    max_value=max_time_s,
    value=(min_time_s, max_time_s),
    step=1
)

# ====================================================
# FILTROS E DOWNSAMPLING
# ====================================================
df_filtered = df[
    (df["Time (s)"] >= time_range[0]) & 
    (df["Time (s)"] <= time_range[1])
].reset_index(drop=True)

if auto_downsample:
    plot_df = downsample_df_for_plot(df_filtered, "Time (s)", max_points=int(max_points))
else:
    plot_df = df_filtered.copy()

# ====================================================
# MÉTRICAS DE ERRO (se disponível)
# ====================================================
if has_error_data:
    st.header("📈 Estatísticas de Erro de Transmissão")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        t1_mean_err = df_filtered["T1_error"].mean()
        t1_max_err = df_filtered["T1_error"].max()
        st.metric("T1 Erro Médio", f"{t1_mean_err:.4f}°C")
        st.metric("T1 Erro Máximo", f"{t1_max_err:.4f}°C")
    
    with col2:
        t2_mean_err = df_filtered["T2_error"].mean()
        t2_max_err = df_filtered["T2_error"].max()
        st.metric("T2 Erro Médio", f"{t2_mean_err:.4f}°C")
        st.metric("T2 Erro Máximo", f"{t2_max_err:.4f}°C")
    
    with col3:
        # Taxa de erro significativo (>0.1°C)
        t1_sig_errors = (df_filtered["T1_error"] > 0.1).sum()
        t1_error_rate = 100 * t1_sig_errors / len(df_filtered)
        st.metric("T1 Erros >0.1°C", f"{t1_sig_errors}")
        st.metric("Taxa", f"{t1_error_rate:.2f}%")
    
    with col4:
        t2_sig_errors = (df_filtered["T2_error"] > 0.1).sum()
        t2_error_rate = 100 * t2_sig_errors / len(df_filtered)
        st.metric("T2 Erros >0.1°C", f"{t2_sig_errors}")
        st.metric("Taxa", f"{t2_error_rate:.2f}%")

# ====================================================
# GRÁFICOS
# ====================================================
st.header("📊 Visualizações")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Temperaturas",
    "⚡ Atuadores",
    "❌ Análise de Erros",
    "📉 Comparação Real vs Recebido"
])

with tab1:
    st.subheader("Temperaturas T1 e T2 com Setpoints")
    
    fig_temp = go.Figure()
    
    # T1 e T2
    fig_temp.add_trace(go.Scatter(
        x=plot_df["Time (s)"],
        y=plot_df["T1"],
        mode="lines",
        name="T1",
        line=dict(color="#FF6B6B", width=2)
    ))
    fig_temp.add_trace(go.Scatter(
        x=plot_df["Time (s)"],
        y=plot_df["T2"],
        mode="lines",
        name="T2",
        line=dict(color="#4ECDC4", width=2)
    ))
    
    # Setpoints
    x0, x1 = plot_df["Time (s)"].min(), plot_df["Time (s)"].max()
    fig_temp.add_trace(go.Scatter(
        x=[x0, x1],
        y=[T1_setpoint, T1_setpoint],
        mode="lines",
        name="T1 Setpoint",
        line=dict(dash="dash", color="#FF6B6B", width=1)
    ))
    fig_temp.add_trace(go.Scatter(
        x=[x0, x1],
        y=[T2_setpoint, T2_setpoint],
        mode="lines",
        name="T2 Setpoint",
        line=dict(dash="dash", color="#4ECDC4", width=1)
    ))
    
    fig_temp.update_layout(
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Temperatura (°C)",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)

with tab2:
    st.subheader("Potência dos Atuadores Q1 e Q2")
    
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(
        x=plot_df["Time (s)"],
        y=plot_df["Q1"],
        mode="lines",
        name="Q1",
        fill='tozeroy',
        line=dict(color="#FFD93D", width=1.5)
    ))
    fig_q.add_trace(go.Scatter(
        x=plot_df["Time (s)"],
        y=plot_df["Q2"],
        mode="lines",
        name="Q2",
        fill='tozeroy',
        line=dict(color="#6BCB77", width=1.5)
    ))
    
    fig_q.update_layout(
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Potência (%)",
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_q, use_container_width=True)

with tab3:
    if has_error_data:
        st.subheader("Erros de Transmissão ao Longo do Tempo")
        
        fig_err = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Erro T1", "Erro T2"),
            vertical_spacing=0.12
        )
        
        fig_err.add_trace(
            go.Scatter(
                x=plot_df["Time (s)"],
                y=plot_df["T1_error"],
                mode="lines",
                name="T1 Error",
                line=dict(color="#E74C3C", width=1.5),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        fig_err.add_trace(
            go.Scatter(
                x=plot_df["Time (s)"],
                y=plot_df["T2_error"],
                mode="lines",
                name="T2 Error",
                line=dict(color="#9B59B6", width=1.5),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig_err.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig_err.update_yaxes(title_text="Erro (°C)", row=1, col=1)
        fig_err.update_yaxes(title_text="Erro (°C)", row=2, col=1)
        
        fig_err.update_layout(
            template="plotly_dark",
            height=600,
            showlegend=True,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_err, use_container_width=True)
        
        # Histograma de erros
        st.subheader("Distribuição dos Erros")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig_hist_t1 = px.histogram(
                df_filtered,
                x="T1_error",
                nbins=50,
                title="Distribuição Erro T1",
                template="plotly_dark",
                color_discrete_sequence=["#E74C3C"]
            )
            fig_hist_t1.update_layout(xaxis_title="Erro (°C)", yaxis_title="Frequência")
            st.plotly_chart(fig_hist_t1, use_container_width=True)
        
        with col_b:
            fig_hist_t2 = px.histogram(
                df_filtered,
                x="T2_error",
                nbins=50,
                title="Distribuição Erro T2",
                template="plotly_dark",
                color_discrete_sequence=["#9B59B6"]
            )
            fig_hist_t2.update_layout(xaxis_title="Erro (°C)", yaxis_title="Frequência")
            st.plotly_chart(fig_hist_t2, use_container_width=True)
    else:
        st.info("📝 Dados de erro não disponíveis neste CSV")

with tab4:
    if has_error_data:
        st.subheader("Comparação: Valor Real vs Valor Recebido")
        
        # T1
        fig_comp_t1 = go.Figure()
        fig_comp_t1.add_trace(go.Scatter(
            x=plot_df["Time (s)"],
            y=plot_df["T1_real"],
            mode="lines",
            name="T1 Real",
            line=dict(color="#3498DB", width=2)
        ))
        fig_comp_t1.add_trace(go.Scatter(
            x=plot_df["Time (s)"],
            y=plot_df["T1"],
            mode="lines",
            name="T1 Recebido",
            line=dict(color="#E74C3C", width=1, dash="dot")
        ))
        fig_comp_t1.update_layout(
            template="plotly_dark",
            title="T1: Real vs Recebido",
            xaxis_title="Time (s)",
            yaxis_title="Temperatura (°C)",
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig_comp_t1, use_container_width=True)
        
        # T2
        fig_comp_t2 = go.Figure()
        fig_comp_t2.add_trace(go.Scatter(
            x=plot_df["Time (s)"],
            y=plot_df["T2_real"],
            mode="lines",
            name="T2 Real",
            line=dict(color="#2ECC71", width=2)
        ))
        fig_comp_t2.add_trace(go.Scatter(
            x=plot_df["Time (s)"],
            y=plot_df["T2"],
            mode="lines",
            name="T2 Recebido",
            line=dict(color="#9B59B6", width=1, dash="dot")
        ))
        fig_comp_t2.update_layout(
            template="plotly_dark",
            title="T2: Real vs Recebido",
            xaxis_title="Time (s)",
            yaxis_title="Temperatura (°C)",
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig_comp_t2, use_container_width=True)
    else:
        st.info("📝 Dados de comparação não disponíveis neste CSV")

# ====================================================
# TABELA DE DADOS
# ====================================================
st.header("📋 Preview dos Dados")

show_cols = st.multiselect(
    "Colunas para exibir:",
    options=df_filtered.columns.tolist(),
    default=["Time (s)", "T1", "T2", "Q1", "Q2"]
)

if show_cols:
    st.dataframe(
        df_filtered[show_cols].head(500),
        use_container_width=True,
        height=300
    )

# ====================================================
# ESTATÍSTICAS PARA ARTIGO
# ====================================================
if has_error_data:
    st.markdown("---")
    st.header("📊 Estatísticas para Artigo ")
    
    # Calcula todas as estatísticas
    stats_data = {
        'dataset': {
            'total_samples': len(df),
            'duration_seconds': max_time_s - min_time_s,
            'duration_days': (max_time_s - min_time_s) / 86400,
            'sampling_rate': 1.0  # Hz
        },
        'compression': {
            'original_bytes': original_size_bytes,
            'compressed_bytes': comp_info['compressed_bytes_est'],
            'compression_ratio': compression_ratio,
            'compression_time_s': comp_info['comp_time'],
            'decompression_time_s': comp_info['dec_time']
        },
        'errors_T1': {
            'mean_error': df["T1_error"].mean(),
            'max_error': df["T1_error"].max(),
            'std_error': df["T1_error"].std(),
            'errors_gt_0_1C': int((df["T1_error"] > 0.1).sum()),
            'error_rate_percent': float((df["T1_error"] > 0.1).sum() / len(df) * 100)
        },
        'errors_T2': {
            'mean_error': df["T2_error"].mean(),
            'max_error': df["T2_error"].max(),
            'std_error': df["T2_error"].std(),
            'errors_gt_0_1C': int((df["T2_error"] > 0.1).sum()),
            'error_rate_percent': float((df["T2_error"] > 0.1).sum() / len(df) * 100)
        }
    }
    
    # Se tiver colunas de Q1/Q2 error
    if "Q1_error" in df.columns:
        stats_data['errors_Q1'] = {
            'mean_error': df["Q1_error"].mean(),
            'max_error': df["Q1_error"].max(),
            'std_error': df["Q1_error"].std()
        }
    if "Q2_error" in df.columns:
        stats_data['errors_Q2'] = {
            'mean_error': df["Q2_error"].mean(),
            'max_error': df["Q2_error"].max(),
            'std_error': df["Q2_error"].std()
        }
    
    # Mostra estatísticas
    tab_stats1, tab_stats2, tab_stats3 = st.tabs([
        "📊 Resumo Executivo",
        "📋 Tabelas LaTeX",
        "💾 Exportar Dados"
    ])
    
    with tab_stats1:
        st.subheader("Resumo das Métricas Principais")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("**🔢 Dataset**")
            st.metric("Total de Amostras", f"{stats_data['dataset']['total_samples']:,}")
            st.metric("Duração", f"{stats_data['dataset']['duration_days']:.2f} dias")
            st.metric("Taxa Amostragem", f"{stats_data['dataset']['sampling_rate']} Hz")
        
        with col_b:
            st.markdown("**📦 Compressão**")
            st.metric("Taxa Compressão", f"{stats_data['compression']['compression_ratio']:.2f}%")
            st.metric("Tempo Comp.", f"{stats_data['compression']['compression_time_s']:.3f} s")
            st.metric("Tempo Decomp.", f"{stats_data['compression']['decompression_time_s']:.3f} s")
        
        with col_c:
            st.markdown("**❌ Erros (T1)**")
            st.metric("EMA", f"{stats_data['errors_T1']['mean_error']:.4f}°C")
            st.metric("Erro Máximo", f"{stats_data['errors_T1']['max_error']:.4f}°C")
            st.metric("Taxa Erro >0.1°C", f"{stats_data['errors_T1']['error_rate_percent']:.3f}%")
        
        st.markdown("---")
        
        # Tabela comparativa
        st.markdown("**Comparação T1 vs T2:**")
        comparison_df = pd.DataFrame({
            'Canal': ['T1', 'T2'],
            'EMA (°C)': [
                f"{stats_data['errors_T1']['mean_error']:.4f}",
                f"{stats_data['errors_T2']['mean_error']:.4f}"
            ],
            'Erro Máx (°C)': [
                f"{stats_data['errors_T1']['max_error']:.4f}",
                f"{stats_data['errors_T2']['max_error']:.4f}"
            ],
            'Desvio Padrão (°C)': [
                f"{stats_data['errors_T1']['std_error']:.4f}",
                f"{stats_data['errors_T2']['std_error']:.4f}"
            ],
            'Erros >0.1°C': [
                stats_data['errors_T1']['errors_gt_0_1C'],
                stats_data['errors_T2']['errors_gt_0_1C']
            ],
            'Taxa (%)': [
                f"{stats_data['errors_T1']['error_rate_percent']:.3f}",
                f"{stats_data['errors_T2']['error_rate_percent']:.3f}"
            ]
        })
        st.dataframe(comparison_df, use_container_width=True)
    
    with tab_stats2:
        st.subheader("Tabelas em Formato LaTeX")
        
        st.markdown("**Copie e cole estas tabelas diretamente no seu artigo!**")
        
        # TABELA 1: Estatísticas de erro por canal
        latex_table_4 = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Estatísticas de erro por canal}}
\\label{{tab:estatisticas_erro}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Canal}} & \\textbf{{EMA}} & \\textbf{{Erro Máx.}} & \\textbf{{Desvio Padrão}} & \\textbf{{Erros $>$0,1°C}} \\\\
\\midrule
T1 & \\SI{{{stats_data['errors_T1']['mean_error']:.4f}}}{{\\celsius}} & \\SI{{{stats_data['errors_T1']['max_error']:.2f}}}{{\\celsius}} & \\SI{{{stats_data['errors_T1']['std_error']:.3f}}}{{\\celsius}} & {stats_data['errors_T1']['errors_gt_0_1C']} ({stats_data['errors_T1']['error_rate_percent']:.3f}\\%) \\\\
T2 & \\SI{{{stats_data['errors_T2']['mean_error']:.4f}}}{{\\celsius}} & \\SI{{{stats_data['errors_T2']['max_error']:.2f}}}{{\\celsius}} & \\SI{{{stats_data['errors_T2']['std_error']:.3f}}}{{\\celsius}} & {stats_data['errors_T2']['errors_gt_0_1C']} ({stats_data['errors_T2']['error_rate_percent']:.3f}\\%) \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
        
        st.code(latex_table_4, language="latex")
        
        st.markdown("---")
        
        # TABELA 2: Compressão
        latex_table_comp = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Desempenho da compressão Huffman}}
\\label{{tab:compressao}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Métrica}} & \\textbf{{Valor}} \\\\
\\midrule
Tamanho original & \\SI{{{stats_data['compression']['original_bytes']}}}{{\\byte}} \\\\
Tamanho comprimido & \\SI{{{stats_data['compression']['compressed_bytes']}}}{{\\byte}} \\\\
Taxa de compressão & {stats_data['compression']['compression_ratio']:.2f}\\% \\\\
Tempo compressão & \\SI{{{stats_data['compression']['compression_time_s']:.3f}}}{{\\second}} \\\\
Tempo descompressão & \\SI{{{stats_data['compression']['decompression_time_s']:.3f}}}{{\\second}} \\\\
Taxa processamento & \\SI{{{stats_data['compression']['original_bytes']/stats_data['compression']['compression_time_s']/1e6:.2f}}}{{\\mega\\byte\\per\\second}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
        
        st.code(latex_table_comp, language="latex")
    
    with tab_stats3:
        st.subheader("Exportar Resultados")
        
        # JSON completo
        st.markdown("**📄 Arquivo JSON com todas as estatísticas:**")
        json_str = json.dumps(stats_data, indent=2)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name="resultados_tclab.json",
            mime="application/json"
        )
        
        st.code(json_str, language="json")
        
        st.markdown("---")
        
        # CSV com dados filtrados
        st.markdown("**📊 Exportar dados filtrados (CSV):**")
        csv_export = df_filtered.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV Filtrado",
            data=csv_export,
            file_name="dados_filtrados_tclab.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Resumo para copiar
        st.markdown("**📝 Resumo para Copiar (Markdown):**")
        summary_md = f"""## Resultados da Transmissão TCLab

### Dataset
- Total de amostras: {stats_data['dataset']['total_samples']:,}
- Duração: {stats_data['dataset']['duration_days']:.2f} dias
- Taxa de amostragem: {stats_data['dataset']['sampling_rate']} Hz

### Compressão Huffman
- Taxa de compressão: {stats_data['compression']['compression_ratio']:.2f}%
- Tamanho original: {stats_data['compression']['original_bytes']:,} bytes
- Tamanho comprimido: {stats_data['compression']['compressed_bytes']:,} bytes

### Erros de Transmissão

#### Canal T1
- EMA: {stats_data['errors_T1']['mean_error']:.4f}°C
- Erro máximo: {stats_data['errors_T1']['max_error']:.4f}°C
- Desvio padrão: {stats_data['errors_T1']['std_error']:.4f}°C
- Erros >0.1°C: {stats_data['errors_T1']['errors_gt_0_1C']} ({stats_data['errors_T1']['error_rate_percent']:.3f}%)

#### Canal T2
- EMA: {stats_data['errors_T2']['mean_error']:.4f}°C
- Erro máximo: {stats_data['errors_T2']['max_error']:.4f}°C
- Desvio padrão: {stats_data['errors_T2']['std_error']:.4f}°C
- Erros >0.1°C: {stats_data['errors_T2']['errors_gt_0_1C']} ({stats_data['errors_T2']['error_rate_percent']:.3f}%)
"""
        st.code(summary_md, language="markdown")
        
        st.download_button(
            label="⬇️ Download Resumo (MD)",
            data=summary_md,
            file_name="resumo_resultados.md",
            mime="text/markdown"
        )

# ====================================================
# RESUMO FINAL
# ====================================================
st.markdown("---")
st.header("📊 Resumo Geral")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Amostras", f"{len(df):,}")
    st.metric("Amostras Filtradas", f"{len(df_filtered):,}")

with col2:
    st.metric("Tamanho Original", f"{original_size_bytes:,} B")
    st.metric("Compressão", f"{compression_ratio:.2f}%")

with col3:
    duration_min = (max_time_s - min_time_s) / 60
    st.metric("Duração Total", f"{duration_min:.1f} min")
    st.metric("Tempo/Amostra", f"{comp_info['comp_time']/len(df)*1000:.3f} ms")

with col4:
    if has_error_data:
        avg_error = (df_filtered["T1_error"].mean() + df_filtered["T2_error"].mean()) / 2
        st.metric("Erro Médio Geral", f"{avg_error:.4f}°C")
        max_error = max(df_filtered["T1_error"].max(), df_filtered["T2_error"].max())
        st.metric("Erro Máximo", f"{max_error:.4f}°C")
    else:
        st.metric("Status", "✅ OK")

# ====================================================
# RODAPÉ
# ====================================================
st.markdown("---")
st.caption("""
**Sistema de Transmissão TCLab com Detecção/Correção de Erros**
- 🔐 CRC-16: Detecção de erros
- 🔧 Hamming (7,4): Correção de até 1 bit por pacote
- 📦 Huffman: Compressão de dados
- 📡 Multi-canal: T1, T2, Q1, Q2 independentes
""")