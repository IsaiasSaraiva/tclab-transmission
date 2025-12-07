# 📡 Sistema de Transmissão Robusta TCLab

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

> Sistema de transmissão de dados do Temperature Control Laboratory (TCLab) com detecção/correção de erros e compressão para ambientes industriais ruidosos.

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Arquitetura](#-arquitetura)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Resultados](#-resultados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Tecnologias](#-tecnologias)
- [Autores](#-autores)
- [Licença](#-licença)

---

## 🎯 Sobre o Projeto

Este projeto implementa um **sistema completo de transmissão robusta** para dados de sensores industriais, desenvolvido como trabalho final de Resolução de Problemas Industriais do programa de pós graduação dem engenharia elétrica da Universidade Federal do Amazonas (PPGEE).

### O Problema

Ambientes industriais apresentam alta interferência eletromagnética (motores, inversores, etc.), corrompendo dados transmitidos por sensores sem fio. A perda de dados compromete sistemas de controle e monitoramento.

### A Solução

Sistema em **camadas de proteção** que:
- ✅ **Detecta** erros via CRC-16 (100% de eficácia)
- ✅ **Corrige** erros via Hamming (7,4) (81,3% de correção)
- ✅ **Comprime** dados via Huffman (53,82% de redução)
- ✅ **Visualiza** em tempo real via Streamlit

### Dados Reais

Utiliza **610.800 amostras** coletadas do TCLab ao longo de **7 dias** (4 canais: T1, T2, Q1, Q2).

---

## 🚀 Funcionalidades

### 🔐 Proteção Multi-Camada

```
Sensor (float 32-bit)
    ↓
[1] Serialização IEEE 754
    ↓
[2] Hamming (7,4) → Correção de 1 bit/bloco
    ↓
[3] CRC-16 → Detecção 100%
    ↓
📡 Canal Ruidoso (Bit Flip + Burst)
    ↓
[3] Verificação CRC
    ↓
[2] Decodificação Hamming
    ↓
[1] Desserialização
    ↓
Valor Recuperado
```

### 📊 Interface Interativa

- **Gráficos temporais** (Plotly) com zoom e pan
- **Análise de erros** (histogramas, estatísticas)
- **Comparação Real vs Recebido**
- **Exportação** (JSON, CSV, LaTeX)

### 🎚️ Níveis de Ruído Configuráveis

| Nível | BER | Burst Prob. | Burst Len. | Descrição |
|-------|-----|-------------|------------|-----------|
| Baixo | 0,01% | 0,1% | 1 byte | Condições ideais |
| **Médio** | **0,1%** | **1%** | **2 bytes** | **Ambiente urbano** |
| Alto | 0,5% | 5% | 3 bytes | Interferência severa |
| Extremo | 2% | 15% | 5 bytes | Condições adversas |

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    TCLab (7 dias)                       │
│            T1, T2 (°C)  |  Q1, Q2 (%)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│          process_existing_csv.py                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Emulação de Transmissão Ruidosa                 │   │
│  │  • 4 canais independentes (T1, T2, Q1, Q2)       │   │
│  │  • Bit Flip: BER configurável                    │   │
│  │  • Burst Error: Probabilidade + Comprimento      │   │
│  │  • CRC-16 (detecção)                             │   │
│  │  • Hamming (7,4) (correção)                      │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│     tclab_noisy_medium_TIMESTAMP.csv                    │
│     (610.800 amostras × 4 canais com erros corrigidos)  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│       tclab_streamlit_integrated.py                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Interface Web Interativa                        │   │
│  │  • Compressão Huffman (53,82%)                   │   │
│  │  • Visualizações (Plotly)                        │   │
│  │  • Análise de erros                              │   │
│  │  • Exportação de resultados                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Pré-requisitos

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Navegador web** (Chrome, Firefox, Edge)

---

## 🔧 Instalação

### 1️⃣ Clone o Repositório

```bash
git clone https://github.com/seu-usuario/tclab-transmission.git
cd tclab-transmission
```

### 2️⃣ Crie o Ambiente Virtual

**Linux/macOS:**
```bash
python3 -m venv venv
```

**Windows:**
```cmd
python -m venv venv
```

### 3️⃣ Ative o Ambiente Virtual

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

> 💡 **Dica:** Você saberá que está no ambiente virtual quando ver `(venv)` no início da linha de comando.

### 4️⃣ Instale as Dependências

```bash
pip install -r requirements.txt
```

**Dependências instaladas:**
- `pandas` - Manipulação de dados
- `numpy` - Operações numéricas
- `streamlit` - Interface web
- `plotly` - Visualizações interativas

---

## 🎮 Como Usar

### Passo 1: Processar Dados com Ruído

Emule a transmissão ruidosa sobre os dados do TCLab:

```bash
python process_existing_csv.py tclab_data_7days.csv --noise-level medium
```

**Opções de ruído:**
- `--noise-level low` → Ruído baixo (0,01% BER)
- `--noise-level medium` → Ruído médio (0,1% BER) ⭐ **Recomendado**
- `--noise-level high` → Ruído alto (0,5% BER)
- `--noise-level extreme` → Ruído extremo (2% BER)

**Saída esperada:**
```
======================================================================
EMULANDO TRANSMISSÃO RUIDOSA SOBRE DADOS EXISTENTES
======================================================================
📂 Arquivo de entrada: tclab_data_7days.csv
🔊 Nível de ruído: medium
  - BER (Bit Error Rate): 0.100%
  - Burst probability: 1.0%
  - Burst length: 2 bytes
======================================================================

📖 Lendo CSV original...
✅ CSV válido! 610800 amostras encontradas

🔄 Processando transmissão canal por canal...

[610000/610800]  99.9% | T1_err=0.0000°C | CRC=52982 | Hamming=43056

✅ Processamento completo!

💾 Salvando arquivos...
  ✓ CSV: tclab_noisy_medium_20251207_163329.csv
  ✓ Stats: tclab_noisy_stats_medium_20251207_163329.json
```

**Arquivos gerados:**
- `tclab_noisy_medium_TIMESTAMP.csv` → Dados processados ✅ **Use este!**
- `tclab_noisy_stats_medium_TIMESTAMP.json` → Estatísticas

⏱️ **Tempo de processamento:** ~10-15 minutos (610.800 amostras × 4 canais)

---

### Passo 2: Visualizar Resultados

Abra a interface web interativa:

```bash
streamlit run tclab_streamlit_integrated.py
```

**Acesse:** [http://localhost:8501](http://localhost:8501)

#### 📤 Na Interface Web:

1. **Faça upload** do CSV processado (`tclab_noisy_medium_*.csv`)
2. **Explore as abas:**
   - 🌡️ **Temperaturas** - Séries temporais de T1 e T2
   - ⚡ **Atuadores** - Padrão de Q1 e Q2
   - ❌ **Análise de Erros** - Erros ao longo do tempo
   - 📉 **Comparação** - Real vs Recebido
3. **Vá até o final** → Seção **"📊 Estatísticas para Artigo Científico"**
4. **Copie as tabelas LaTeX** prontas!

---

## 📊 Resultados

### Desempenho sob Ruído Médio

| Métrica | Valor |
|---------|-------|
| **Pacotes transmitidos** | 2.443.200 (610.800 × 4 canais) |
| **Taxa de corrupção** | 8,6% (210.152 pacotes) |
| **Taxa de detecção (CRC)** | **100%** ✅ |
| **Taxa de correção (Hamming)** | **81,3%** ✅ |
| **Taxa de erro residual** | **0,0003%** (8 pacotes) |
| **Erro médio absoluto (EMA)** | **0,0024°C** |
| **Compressão Huffman** | **53,82%** (78,71 MB → 36,34 MB) |

### Taxa de Sucesso Global

```
✅ 99,9997% de sucesso
   ├─ 91,4% recebidos sem corrupção
   ├─ 8,3% corrompidos mas recuperados
   └─ 0,0003% irrecuperáveis
```

### Desempenho Computacional

- **Processamento:** 813× mais rápido que tempo real
- **Escalabilidade:** Suporta centenas de canais simultâneos
- **Latência:** < 200 ms na interface web

---

## 📁 Estrutura do Projeto

```
tclab-transmission/
│
├── 📄 README.md                              ← Você está aqui!
├── 📄 requirements.txt                       ← Dependências Python
│
├── 🐍 process_existing_csv.py                ← Script de processamento
│   └─ Emula transmissão ruidosa
│   └─ Aplica CRC-16 + Hamming (7,4)
│   └─ Gera CSV processado + JSON de stats
│
├── 🐍 tclab_streamlit_integrated.py          ← Interface web
│   └─ Compressão Huffman
│   └─ Visualizações Plotly
│   └─ Análise de erros
│   └─ Exportação de resultados
│
├── 📊 tclab_data_7days.csv                   ← Dados originais (610.800 amostras)
│
├── 📊 tclab_noisy_medium_TIMESTAMP.csv       ← Dados processados (gerado)
├── 📄 tclab_noisy_stats_medium_TIMESTAMP.json ← Estatísticas (gerado)
│
└── 📁 venv/                                  ← Ambiente virtual (criar localmente)
```

---

## 🛠️ Tecnologias

### Linguagem
- **Python 3.10+**

### Bibliotecas Principais
- **Pandas** - Manipulação de dados tabulares
- **NumPy** - Operações matriciais (Hamming)
- **Streamlit** - Interface web interativa
- **Plotly** - Visualizações interativas

### Algoritmos Implementados

#### 🔐 CRC-16-CCITT
- Polinômio: `0x1021`
- Detecção: 100% erros de 1-2 bits
- Overhead: 2 bytes (16 bits)

#### 🔧 Hamming (7,4)
- Correção: 1 bit por bloco de 7 bits
- Detecção: 2 bits por bloco
- Overhead: 75% (4 bits → 7 bits)

#### 📦 Huffman
- Compressão sem perdas
- Baseado em frequência de bytes
- Taxa: 30-55% (típico para CSV)

---

## 🎓 Contexto Acadêmico

**Disciplina:** Resolução de Problemas Industriais  
**Programa:** Pós-Graduação em Engenharia Elétrica (PPGEE)  
**Instituição:** [Sua Universidade]  
**Ano:** 2024/2025

---

## 👨‍💻 Autores

**Isaías [Sobrenome]**  
📧 Email: [seu-email@email.com]  
🔗 GitHub: [@seu-usuario](https://github.com/seu-usuario)  
🎓 PPGEE - [Sua Universidade]

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 🙏 Agradecimentos

- **Prof. [Nome do Orientador]** - Orientação e supervisão
- **TCLab Community** - Hardware e documentação

---

## 📚 Referências

1. **TCLab Documentation** - https://apmonitor.com/pdc
2. **CRC-16-CCITT** - ITU-T Recommendation V.41
3. **Hamming Codes** - R. W. Hamming, 1950
4. **Huffman Coding** - D. A. Huffman, 1952

---

## 💡 Melhorias Futuras

- [ ] Implementar Reed-Solomon para maior robustez
- [ ] Seleção adaptativa de proteção baseada em qualidade do canal
- [ ] Validação em hardware embarcado (ESP32, STM32)
- [ ] Comunicação via LoRa/Zigbee
- [ ] Comparação com padrões WirelessHART

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela!**

**Feito com ❤️ e ☕ por Isaías**

</div>
