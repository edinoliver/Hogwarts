from engine import prepare_all, run_test

def main():
    # Ajuste aqui se quiser frases mais curtas/longas:
    MIN_WORDS = 3
    MAX_WORDS = 12

    oxford_set, filtered, counts = prepare_all(min_words=MIN_WORDS, max_words=MAX_WORDS)

    print("✅ Oxford A1/A2/B1 carregado.")
    print("A1:", len(counts["A1"]), "| A2:", len(counts["A2"]), "| B1:", len(counts["B1"]))
    print("✅ Frases filtradas (só Oxford A1–B1):", len(filtered))

    if len(filtered) < 50:
        print("⚠️ Poucas frases após filtro. Tente aumentar MAX_WORDS ou diminuir MIN_WORDS.")

    # Começa o teste
    run_test(oxford_set, filtered, n_words=10, n_sentences=10, threshold=0.86, seed=None)

if __name__ == "__main__":
    main()
``
