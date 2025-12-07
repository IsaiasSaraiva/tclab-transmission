#!/usr/bin/env python3
"""
process_existing_csv.py

Lê um CSV existente do TCLab (tclab_data_7days.csv) e EMULA transmissão 
ruidosa sobre ele, aplicando detecção (CRC-16) e correção (Hamming) de erros.

Uso:
    python process_existing_csv.py tclab_data_7days.csv --noise-level medium
"""
import argparse
import json
import pandas as pd
import struct
import random
import numpy as np
from datetime import datetime
from typing import Tuple


# ====================================================
# CRC-16: Detecção de erro
# ====================================================
def crc16(data: bytes) -> int:
    """Calcula CRC-16-CCITT"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


def add_crc(data: bytes) -> bytes:
    """Adiciona CRC-16"""
    crc = crc16(data)
    return data + struct.pack('>H', crc)


def verify_crc(data_with_crc: bytes) -> Tuple[bool, bytes]:
    """Verifica CRC"""
    if len(data_with_crc) < 2:
        return False, data_with_crc
    data = data_with_crc[:-2]
    received_crc = struct.unpack('>H', data_with_crc[-2:])[0]
    calculated_crc = crc16(data)
    return received_crc == calculated_crc, data


# ====================================================
# HAMMING (7,4): Correção de erro
# ====================================================
class HammingCode:
    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)
    
    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ], dtype=int)
    
    @classmethod
    def encode_nibble(cls, nibble: int) -> int:
        data_bits = [(nibble >> i) & 1 for i in range(4)]
        encoded = np.dot(data_bits, cls.G) % 2
        result = 0
        for i, bit in enumerate(encoded):
            result |= (bit << i)
        return result
    
    @classmethod
    def decode_7bits(cls, received: int) -> Tuple[int, bool]:
        received_bits = [(received >> i) & 1 for i in range(7)]
        syndrome = np.dot(cls.H, received_bits) % 2
        syndrome_val = syndrome[0] + syndrome[1] * 2 + syndrome[2] * 4
        
        corrected = False
        if syndrome_val != 0:
            error_pos = syndrome_val - 1
            if 0 <= error_pos < 7:
                received_bits[error_pos] ^= 1
                corrected = True
        
        nibble = received_bits[0] + (received_bits[1] << 1) + (received_bits[2] << 2) + (received_bits[3] << 3)
        return nibble, corrected
    
    @classmethod
    def encode_bytes(cls, data: bytes) -> bytes:
        result = bytearray()
        for byte in data:
            low_nibble = byte & 0x0F
            high_nibble = (byte >> 4) & 0x0F
            
            encoded_low = cls.encode_nibble(low_nibble)
            encoded_high = cls.encode_nibble(high_nibble)
            
            result.append(encoded_low & 0x7F)
            result.append(encoded_high & 0x7F)
        
        return bytes(result)
    
    @classmethod
    def decode_bytes(cls, encoded_data: bytes) -> Tuple[bytes, int]:
        result = bytearray()
        corrections = 0
        
        for i in range(0, len(encoded_data), 2):
            if i + 1 >= len(encoded_data):
                break
            
            encoded_low = encoded_data[i] & 0x7F
            encoded_high = encoded_data[i + 1] & 0x7F
            
            low_nibble, corr1 = cls.decode_7bits(encoded_low)
            high_nibble, corr2 = cls.decode_7bits(encoded_high)
            
            corrections += corr1 + corr2
            
            byte_val = low_nibble | (high_nibble << 4)
            result.append(byte_val)
        
        return bytes(result), corrections


# ====================================================
# RUÍDO
# ====================================================
def apply_bit_flip_noise(data: bytes, ber: float) -> bytes:
    """Aplica ruído de bit flip"""
    byte_array = bytearray(data)
    for i in range(len(byte_array)):
        for bit in range(8):
            if random.random() < ber:
                byte_array[i] ^= (1 << bit)
    return bytes(byte_array)


def apply_burst_error(data: bytes, prob: float, length: int) -> bytes:
    """Aplica erro em rajada"""
    byte_array = bytearray(data)
    if random.random() < prob and len(byte_array) >= length:
        start = random.randint(0, len(byte_array) - length)
        for i in range(start, start + length):
            byte_array[i] = random.randint(0, 255)
    return bytes(byte_array)


# ====================================================
# PROCESSAMENTO DO CSV
# ====================================================
def process_csv(input_file: str, noise_level: str, output_prefix: str):
    """Processa CSV existente aplicando transmissão ruidosa"""
    
    # Configurações de ruído
    noise_configs = {
        'low': {'ber': 0.0001, 'burst_prob': 0.001, 'burst_len': 1},
        'medium': {'ber': 0.001, 'burst_prob': 0.01, 'burst_len': 2},
        'high': {'ber': 0.005, 'burst_prob': 0.05, 'burst_len': 3},
        'extreme': {'ber': 0.02, 'burst_prob': 0.15, 'burst_len': 5}
    }
    
    config = noise_configs[noise_level]
    
    print(f"{'='*70}")
    print(f"EMULANDO TRANSMISSÃO RUIDOSA SOBRE DADOS EXISTENTES")
    print(f"{'='*70}")
    print(f"📂 Arquivo de entrada: {input_file}")
    print(f"🔊 Nível de ruído: {noise_level}")
    print(f"  - BER (Bit Error Rate): {config['ber']*100:.3f}%")
    print(f"  - Burst probability: {config['burst_prob']*100:.1f}%")
    print(f"  - Burst length: {config['burst_len']} bytes")
    print(f"{'='*70}\n")
    
    # Lê CSV original
    print("📖 Lendo CSV original...")
    df_original = pd.read_csv(input_file)
    
    required_cols = ['Time (s)', 'T1', 'T2', 'Q1', 'Q2']
    if not all(col in df_original.columns for col in required_cols):
        print(f"❌ ERRO: CSV deve conter as colunas: {required_cols}")
        print(f"   Colunas encontradas: {list(df_original.columns)}")
        return
    
    print(f"✅ CSV válido! {len(df_original)} amostras encontradas\n")
    
    # Estatísticas por canal
    stats = {
        'T1': {'total': 0, 'corrupted': 0, 'crc_errors': 0, 'hamming_corr': 0, 'unrecoverable': 0},
        'T2': {'total': 0, 'corrupted': 0, 'crc_errors': 0, 'hamming_corr': 0, 'unrecoverable': 0},
        'Q1': {'total': 0, 'corrupted': 0, 'crc_errors': 0, 'hamming_corr': 0, 'unrecoverable': 0},
        'Q2': {'total': 0, 'corrupted': 0, 'crc_errors': 0, 'hamming_corr': 0, 'unrecoverable': 0},
    }
    
    # Processa cada linha
    rows_output = []
    
    print("🔄 Processando transmissão canal por canal...")
    print("   (isso pode demorar alguns minutos)\n")
    
    for idx, row in df_original.iterrows():
        time_s = row['Time (s)']
        
        # Valores originais
        T1_orig = float(row['T1'])
        T2_orig = float(row['T2'])
        Q1_orig = float(row['Q1'])
        Q2_orig = float(row['Q2'])
        
        # Função para simular transmissão de um valor
        def transmit_value(value: float, channel: str) -> Tuple[float, dict]:
            stats[channel]['total'] += 1
            
            # 1. Serializa (float → bytes)
            data = struct.pack('f', value)
            
            # 2. Codifica com Hamming (7,4)
            encoded = HammingCode.encode_bytes(data)
            
            # 3. Adiciona CRC-16
            with_crc = add_crc(encoded)
            
            # 4. CANAL RUIDOSO (aqui acontece a corrupção!)
            noisy = apply_bit_flip_noise(with_crc, config['ber'])
            noisy = apply_burst_error(noisy, config['burst_prob'], config['burst_len'])
            
            # 5. Receptor: Verifica CRC
            crc_ok, data_no_crc = verify_crc(noisy)
            
            if not crc_ok:
                stats[channel]['crc_errors'] += 1
                stats[channel]['corrupted'] += 1
            
            # 6. Decodifica Hamming (tenta corrigir)
            decoded, corrections = HammingCode.decode_bytes(data_no_crc)
            stats[channel]['hamming_corr'] += corrections
            
            # 7. Desserializa (bytes → float)
            try:
                recovered = struct.unpack('f', decoded)[0]
                if np.isnan(recovered) or np.isinf(recovered):
                    stats[channel]['unrecoverable'] += 1
                    recovered = value  # fallback para valor original
            except:
                stats[channel]['unrecoverable'] += 1
                recovered = value  # fallback
            
            return recovered, {
                'crc_ok': crc_ok,
                'corrections': corrections,
                'error': abs(value - recovered)
            }
        
        # Transmite cada variável por seu canal
        T1_recv, T1_info = transmit_value(T1_orig, 'T1')
        T2_recv, T2_info = transmit_value(T2_orig, 'T2')
        Q1_recv, Q1_info = transmit_value(Q1_orig, 'Q1')
        Q2_recv, Q2_info = transmit_value(Q2_orig, 'Q2')
        
        # Salva resultado
        rows_output.append({
            'Time (s)': time_s,
            'T1': T1_recv,
            'T2': T2_recv,
            'Q1': Q1_recv,
            'Q2': Q2_recv,
            'T1_real': T1_orig,
            'T2_real': T2_orig,
            'Q1_real': Q1_orig,
            'Q2_real': Q2_orig,
            'T1_error': T1_info['error'],
            'T2_error': T2_info['error'],
            'Q1_error': Q1_info['error'],
            'Q2_error': Q2_info['error']
        })
        
        # Progress a cada 1000 amostras
        if idx % 1000 == 0:
            percent = (idx / len(df_original)) * 100
            print(f"  [{idx:6d}/{len(df_original)}] {percent:5.1f}% | "
                  f"T1_err={T1_info['error']:.4f}°C | "
                  f"CRC={stats['T1']['crc_errors']:4d} | "
                  f"Hamming={stats['T1']['hamming_corr']:4d}")
    
    print(f"\n✅ Processamento completo!\n")
    
    # Salva resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"{output_prefix}_{noise_level}_{timestamp}.csv"
    output_stats = f"{output_prefix}_stats_{noise_level}_{timestamp}.json"
    
    print("💾 Salvando arquivos...")
    df_output = pd.DataFrame(rows_output)
    df_output.to_csv(output_csv, index=False)
    print(f"  ✓ CSV: {output_csv}")
    
    # Salva estatísticas
    with open(output_stats, 'w') as f:
        json.dump({
            'input_file': input_file,
            'noise_level': noise_level,
            'total_samples': len(df_output),
            'channel_statistics': stats,
            'noise_config': config
        }, f, indent=2)
    print(f"  ✓ Stats: {output_stats}\n")
    
    # Exibe resumo
    print(f"{'='*70}")
    print(f"RESUMO DA TRANSMISSÃO")
    print(f"{'='*70}\n")
    
    for channel, ch_stats in stats.items():
        print(f"📡 Canal {channel}:")
        print(f"  └─ Pacotes enviados: {ch_stats['total']:,}")
        print(f"  └─ Pacotes corrompidos: {ch_stats['corrupted']:,} "
              f"({100*ch_stats['corrupted']/max(1,ch_stats['total']):.2f}%)")
        print(f"  └─ Erros detectados (CRC): {ch_stats['crc_errors']:,}")
        print(f"  └─ Correções (Hamming): {ch_stats['hamming_corr']:,}")
        print(f"  └─ Erros irrecuperáveis: {ch_stats['unrecoverable']:,}\n")
    
    print(f"{'='*70}")
    print(f"PRÓXIMOS PASSOS")
    print(f"{'='*70}\n")
    print(f"1. Abra o Streamlit:")
    print(f"   streamlit run tclab_streamlit_integrated.py\n")
    print(f"2. Faça upload do arquivo gerado:")
    print(f"   📂 {output_csv}\n")
    print(f"3. Vá na aba '📊 Estatísticas para Artigo Científico'\n")
    print(f"4. Copie as tabelas LaTeX prontas!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Emula transmissão ruidosa sobre CSV existente do TCLab"
    )
    parser.add_argument(
        'input_file', 
        help='Arquivo CSV de entrada (ex: tclab_data_7days.csv)'
    )
    parser.add_argument(
        '--noise-level', 
        type=str, 
        default='medium',
        choices=['low', 'medium', 'high', 'extreme'],
        help='Nível de ruído (padrão: medium)'
    )
    parser.add_argument(
        '--output-prefix', 
        type=str, 
        default='tclab_noisy',
        help='Prefixo dos arquivos de saída (padrão: tclab_noisy)'
    )
    
    args = parser.parse_args()
    
    process_csv(args.input_file, args.noise_level, args.output_prefix)


if __name__ == "__main__":
    main()