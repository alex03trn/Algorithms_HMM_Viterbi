# -*- coding: utf-8 -*-

import numpy as np
import random
from Bio import SeqIO
from collections import defaultdict
import random
import re

def check_fasta_for_N(fasta_file):
    found_N = False
    with open(fasta_file, 'r') as f:
        seq_name = ""
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_name and 'N' in seq.upper():
                    print(f"В последовательности {seq_name} есть N")
                    found_N = True
                seq_name = line[1:]
                seq = ""
            else:
                seq += line.replace(" ", "").replace("\t","")

        if seq_name and 'N' in seq.upper():
            print(f"В последовательности {seq_name} есть N")
            found_N = True

    if not found_N:
        print("В файле нет последовательностей с буквой N.")

def read_sequence(input_file):
    try:
        seq_record = SeqIO.read(input_file, "fasta")
        return str(seq_record.seq).upper()
    except:
        with open(input_file, 'r') as f:
            raw = f.read()
            seq = re.sub(r'[^ATGC]', '', raw.upper())
            return seq

def compute_emissions_symmetric(seq1, seq2):

    gc1 = (seq1.count('G') + seq1.count('C')) / len(seq1)
    gc2 = (seq2.count('G') + seq2.count('C')) / len(seq2)

    # G=C=GC/2, A=T=(1-GC)/2
    emi1 = {'A': (1-gc1)/2, 'T': (1-gc1)/2, 'G': gc1/2, 'C': gc1/2}
    emi2 = {'A': (1-gc2)/2, 'T': (1-gc2)/2, 'G': gc2/2, 'C': gc2/2}

    for d in [emi1, emi2]:
        for k in d:
            d[k] = max(d[k], 1e-10)

    return emi1, emi2

def viterbi(seq, emi1, emi2, mean_length=300):

    n = len(seq)
    p_stay = 1 - 1/mean_length
    p_switch = 1/mean_length

    delta = np.full((2, n), -np.inf)
    psi = np.zeros((2, n), dtype=int)

    delta[0, 0] = np.log(0.5) + np.log(emi1[seq[0]])
    delta[1, 0] = np.log(0.5) + np.log(emi2[seq[0]])

    for t in range(1, n):
        for j in [0, 1]:
            emi = emi1 if j == 0 else emi2
            p_emit = np.log(emi[seq[t]])

            trans0 = delta[0, t-1] + (np.log(p_stay) if j == 0 else np.log(p_switch))
            trans1 = delta[1, t-1] + (np.log(p_switch) if j == 0 else np.log(p_stay))

            delta[j, t] = max(trans0, trans1) + p_emit
            psi[j, t] = 0 if trans0 > trans1 else 1

    path = np.zeros(n, dtype=int)
    path[-1] = 0 if delta[0, -1] > delta[1, -1] else 1
    for t in range(n-2, -1, -1):
        path[t] = psi[path[t+1], t+1]

    return np.array([x+1 for x in path])

def make_chimera(seq1, seq2, total_length=8000, mean_frag_len=300, seed=42):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    s = ''
    true_states = []
    sources = [seq1, seq2]
    state = np.random.choice([0,1])

    while len(s) < total_length:
        curseq = sources[state]
        frag_len = min(int(np.random.exponential(mean_frag_len)) + 1,
                      total_length - len(s), len(curseq)//2)
        if frag_len < 50:
            state = 1 - state
            continue

        start = np.random.randint(0, len(curseq) - frag_len)
        frag = curseq[start:start+frag_len]
        s += frag
        true_states.extend([state+1] * frag_len)
        state = 1 - state

    with open("chimera.fasta", "w") as f:
        f.write(">chimera_8000bp\n")
        for i in range(0, total_length, 80):
            f.write(s[i:i+80] + "\n")
    with open("true_states.txt", "w") as f:
        f.write("".join(map(str, true_states[:total_length])))

    return s[:total_length], true_states[:total_length]

def accuracy(true_path, pred_path):
    return np.mean(true_path == pred_path) * 100

# === ГЛАВНАЯ ПРОГРАММА ===
if __name__ == "__main__":
    print("VITERBI")
    print("=" * 60)

    seq1 = read_sequence("/content/CP048767.1.fasta")  # Campy
    seq2 = read_sequence("/content/CP033071.1.fasta")  # Strep

    print("ПРОВЕРКА N...")
    check_fasta_for_N("/content/CP048767.1.fasta")     # Campy
    check_fasta_for_N("/content/CP033071.1.fasta")     # Strep

    print("=" * 60)

    emi1, emi2 = compute_emissions_symmetric(seq1, seq2)
    print(f"Эмиссии Campy(1): A=T={emi1['A']:.3f}, G=C={emi1['G']:.3f}")
    print(f"Эмиссии Strep(2): A=T={emi2['A']:.3f}, G=C={emi2['G']:.3f}")

    print("\nХИМЕРА:")
    chimera, true_states = make_chimera(seq1, seq2)
    pred_chimera = viterbi(chimera, emi1, emi2)
    acc_chimera = accuracy(np.array(true_states), pred_chimera)
    print(f"  Точность: {acc_chimera:.2f}%")

    print("\nЧИСТЫЕ ГЕНОМЫ (участки по 8000 bp):")

    # Campylobacter
    starts1 = np.random.randint(0, len(seq1)-8000, 3)
    print("   CAMPYLOBACTER (должно быть 1):")
    for i, start in enumerate(starts1):
        seg = seq1[start:start+8000]
        pred = viterbi(seg, emi1, emi2)
        acc = accuracy([1]*8000, pred)
        print(f"     Участок {i+1}: {acc:.2f}%")

    # Streptomyces
    starts2 = np.random.randint(0, len(seq2)-8000, 3)
    print("   STREPTOMYCES (должно быть 2):")
    for i, start in enumerate(starts2):
        seg = seq2[start:start+8000]
        pred = viterbi(seg, emi1, emi2)
        acc = accuracy([2]*8000, pred)
        print(f"     Участок {i+1}: {acc:.2f}%")

    with open("viterbi_result.txt", "w") as f:
        f.write("".join(map(str, pred_chimera)))

    print(f"\n СОХРАНЕНО: chimera.fasta, true_states.txt, viterbi_result.txt")
