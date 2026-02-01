
#!/usr/bin/env python3

import sys
import numpy as np
import re

try:
    from Bio import SeqIO
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

def check_fasta_for_N(fasta_file):
    """Проверка наличия N"""
    found_N = False
    with open(fasta_file, 'r') as f:
        seq_name, seq = "", ""
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if seq_name and 'N' in seq.upper():
                    print(f"{seq_name}: содержит N")
                    found_N = True
                seq_name, seq = line[1:], ""
            else:
                seq += re.sub(r'[^ATGC]', '', line.upper())
        if seq_name and 'N' in seq.upper():
            print(f"{seq_name}: содержит N")
            found_N = True

    if not found_N:
        print(f"{fasta_file}: N не найдено")
    return not found_N

def read_sequence(fasta_file):
    try:
        if HAS_BIO:
            return str(SeqIO.read(fasta_file, "fasta").seq).upper()
        with open(fasta_file, 'r') as f:
            return re.sub(r'[^ATGC]', '', f.read().upper())
    except Exception as e:
        print(f"Ошибка чтения {fasta_file}: {e}")
        sys.exit(1)

def compute_emissions(seq1, seq2):
    """Эмиссии из GC-состава: G=C, A=T"""
    gc1 = (seq1.count('G') + seq1.count('C')) / len(seq1)
    gc2 = (seq2.count('G') + seq2.count('C')) / len(seq2)

    emi1 = {'A': max((1-gc1)/2, 1e-10), 'T': max((1-gc1)/2, 1e-10),
            'G': max(gc1/2, 1e-10), 'C': max(gc1/2, 1e-10)}
    emi2 = {'A': max((1-gc2)/2, 1e-10), 'T': max((1-gc2)/2, 1e-10),
            'G': max(gc2/2, 1e-10), 'C': max(gc2/2, 1e-10)}

    print(f"Референс 1: GC={gc1*100:.1f}% (A=T={emi1['A']:.3f}, G=C={emi1['G']:.3f})")
    print(f"Референс 2: GC={gc2*100:.1f}% (A=T={emi2['A']:.3f}, G=C={emi2['G']:.3f})")
    return emi1, emi2

def viterbi(seq, emi1, emi2, mean_length=300):
    """Алгоритм Витерби"""
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

    return ''.join(str(x+1) for x in path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("""
ИСПОЛЬЗОВАНИЕ:
   python3 viterbi.py ref1.fasta ref2.fasta test.fasta

АРГУМЕНТЫ:
   ref1.fasta  - геном с низким GC (референс состояния 1)
   ref2.fasta  - геном с высоким GC (референс состояния 2)
   test.fasta  - произвольная ДНК последовательность

ПРИМЕР test.fasta:
   >test
   ATGCCCGGGATAGCT...

ИЛИ чистый файл:
   ATGCCCGGGATAGCT...

ВЫВОД: 12112211... (1=ref1, 2=ref2)

УСТАНОВКА:
   pip3 install numpy biopython
        """)
        sys.exit(1)

    ref1_file, ref2_file, test_file = sys.argv[1:]

    print("Проверка test.fasta на N...")
    if not check_fasta_for_N(test_file):
        print("Прервано: найдены N")
        sys.exit(1)

    print("Чтение файлов...")
    ref1 = read_sequence(ref1_file)
    ref2 = read_sequence(ref2_file)
    test = read_sequence(test_file)

    print(f"Тестовая последовательность: {len(test)} bp")

    print("\nОбучение модели...")
    emi1, emi2 = compute_emissions(ref1, ref2)

    print("\nДекодирование Витерби...")
    result = viterbi(test, emi1, emi2)

    print(f"\nРЕЗУЛЬТАТ ({len(result)} позиций):")
    print(result[:200] + "..." if len(result) > 200 else result)

    with open("viterbi_path.txt", "w") as f:
        f.write(result)
    print("Сохранено: viterbi_path.txt")
