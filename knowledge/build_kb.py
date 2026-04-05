"""
Bilgi Tabani Olusturma Scripti
===============================
knowledge/sources/ altindaki tum markdown dosyalarini okur,
parcalara boler, embed eder ve ChromaDB'ye kaydeder.

Kullanim:
    python knowledge/build_kb.py
    python knowledge/build_kb.py --rebuild   # Sifirdan yeniden olustur
"""

import os
import sys
import argparse
import time

# Proje kokunu Python path'ine ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="CV Bilgi Tabani Olusturucu")
    parser.add_argument("--rebuild", action="store_true",
                        help="Mevcut index'i sil ve sifirdan olustur")
    args = parser.parse_args()

    # Config'den ayarlari al
    try:
        import config
        persist_dir = config.AGENT_CHROMA_DIR
        embedding_model = config.AGENT_EMBEDDING_MODEL
        kb_dir = config.AGENT_KB_DIR
    except ImportError:
        # Fallback
        kb_dir = os.path.dirname(os.path.abspath(__file__))
        persist_dir = os.path.join(kb_dir, "chroma_db")
        embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"

    sources_dir = os.path.join(kb_dir, "sources")

    print("=" * 55)
    print("  CV BILGI TABANI OLUSTURUCU")
    print("=" * 55)
    print(f"  Kaynak dizini : {sources_dir}")
    print(f"  ChromaDB dizini: {persist_dir}")
    print(f"  Embedding modeli: {embedding_model}")
    print("=" * 55)

    # Kaynak dosyalari kontrol et
    if not os.path.isdir(sources_dir):
        print(f"\nHATA: Kaynak dizini bulunamadi: {sources_dir}")
        sys.exit(1)

    # Dosya sayisi
    md_files = []
    for root, dirs, files in os.walk(sources_dir):
        for f in files:
            if f.endswith(('.md', '.txt')):
                md_files.append(os.path.join(root, f))

    if not md_files:
        print(f"\nHATA: {sources_dir} altinda markdown dosyasi bulunamadi.")
        sys.exit(1)

    print(f"\n  {len(md_files)} dokuman bulundu:")
    for f in md_files:
        rel = os.path.relpath(f, sources_dir)
        print(f"    - {rel}")

    # RAG Engine ile index olustur
    from core.agent.rag_engine import RAGEngine

    engine = RAGEngine(
        persist_dir=persist_dir,
        embedding_model_name=embedding_model,
        sources_dir=sources_dir,
    )

    print(f"\n  Embedding modeli yukleniyor: {embedding_model}")
    print("  (Ilk seferinde model indirilecek, ~500MB)")

    t0 = time.time()

    if args.rebuild:
        print("\n  Mevcut index siliniyor ve yeniden olusturuluyor...")
        engine.build_index()
    else:
        # Mevcut varsa yukle, yoksa olustur
        stats = engine.get_stats()
        if stats["total_chunks"] > 0:
            print(f"\n  Mevcut index bulundu: {stats['total_chunks']} parca")
            rebuild = input("  Yeniden olusturmak ister misin? (e/h): ").strip().lower()
            if rebuild in ('e', 'evet', 'y', 'yes'):
                engine.build_index()
            else:
                print("  Mevcut index korunuyor.")
        else:
            engine.build_index()

    elapsed = time.time() - t0
    stats = engine.get_stats()

    print(f"\n{'=' * 55}")
    print(f"  TAMAMLANDI!")
    print(f"  Toplam parca: {stats['total_chunks']}")
    print(f"  Sure: {elapsed:.1f} saniye")
    print(f"  Konum: {persist_dir}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
