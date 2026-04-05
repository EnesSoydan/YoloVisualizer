"""
CV Agent — Ana Orkestrator
============================
Tum bilesenleri (LLM, RAG, DataAnalyzer, Memory) birlestiren
interaktif CV uzman agent.
"""

import os
import json

from .llm_backend import LLMBackend
from .rag_engine import RAGEngine
from .data_analyzer import DataAnalyzer
from .memory import ConversationMemory
from .prompts import (
    SYSTEM_PROMPT,
    DATASET_ANALYSIS_TEMPLATE,
    TRAINING_ANALYSIS_TEMPLATE,
    MODEL_ANALYSIS_TEMPLATE,
    TEACHING_TEMPLATE,
)


class CVAgent:
    """Computer Vision uzman agent."""

    def __init__(self, config):
        self.config = config

        # LLM backend
        self.llm = LLMBackend(
            model_name=getattr(config, 'AGENT_MODEL', 'mistral:7b-instruct-v0.3-q4_K_M'),
            base_url=getattr(config, 'AGENT_OLLAMA_URL', 'http://localhost:11434'),
            temperature=getattr(config, 'AGENT_TEMPERATURE', 0.3),
            num_ctx=getattr(config, 'AGENT_NUM_CTX', 8192),
        )

        # RAG engine
        kb_dir = getattr(config, 'AGENT_KB_DIR', 'knowledge')
        self.rag = RAGEngine(
            persist_dir=getattr(config, 'AGENT_CHROMA_DIR',
                                os.path.join(kb_dir, 'chroma_db')),
            embedding_model_name=getattr(config, 'AGENT_EMBEDDING_MODEL',
                                         'paraphrase-multilingual-MiniLM-L12-v2'),
            sources_dir=os.path.join(kb_dir, 'sources'),
        )

        # Data analyzer
        self.analyzer = DataAnalyzer()

        # Konusma hafizasi
        self.memory = ConversationMemory(max_turns=20)

    def analyze_dataset(self):
        """Dataset analizi yap."""
        dataset_path = getattr(self.config, 'DATASET_PATH', '')
        class_names = getattr(self.config, 'CLASS_NAMES', {})

        if not os.path.exists(dataset_path):
            return "HATA: Dataset yolu bulunamadi: " + dataset_path

        print("  Dataset taranıyor...")
        stats = self.analyzer.analyze_dataset(dataset_path, class_names)

        # RAG'dan ilgili bilgi cek
        rag_context = self.rag.get_context(
            "YOLO dataset dengesi sinif dagilimi augmentation onerileri"
        )

        # Prompt olustur
        class_dist = json.dumps(stats["class_counts"], ensure_ascii=False, indent=2)
        prompt = DATASET_ANALYSIS_TEMPLATE.format(
            n_train=stats["n_train"],
            n_valid=stats["n_valid"],
            val_ratio=stats["val_ratio"],
            empty_labels=stats["empty_labels"],
            total_objects=stats["total_objects"],
            class_distribution=class_dist,
            small_objects=stats["small_objects"],
            medium_objects=stats["medium_objects"],
            large_objects=stats["large_objects"],
            rag_context=rag_context,
        )

        # LLM'e sor
        print("\n")
        response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        self.memory.add("user", "Dataset analizi istendi")
        self.memory.add("assistant", response)
        return response

    def analyze_training(self, csv_path=None):
        """Egitim metriklerini analiz et."""
        if not csv_path:
            csv_path = DataAnalyzer.find_results_csv()
            if csv_path:
                print(f"  Bulundu: {csv_path}")
            else:
                return "results.csv bulunamadi. Yolu manuel gir."

        print("  Egitim verileri analiz ediliyor...")
        result = self.analyzer.analyze_training(csv_path)
        if result is None:
            return f"CSV okunamadi: {csv_path}"

        # RAG'dan ilgili bilgi
        rag_context = self.rag.get_context(
            "YOLO egitim loss mAP overfitting learning rate ayarlama"
        )

        # Son metrikler
        last_str = ""
        for k, v in result["last_metrics"].items():
            if isinstance(v, float):
                last_str += f"  {k}: {v:.4f}\n"

        # Ilerleme
        progress_str = ""
        for k, vals in result["progress"].items():
            progress_str += f"  {k}: {vals}\n"

        prompt = TRAINING_ANALYSIS_TEMPLATE.format(
            n_epochs=result["n_epochs"],
            last_metrics=last_str,
            progress=progress_str,
            rag_context=rag_context,
        )

        print("\n")
        response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        self.memory.add("user", "Egitim analizi istendi")
        self.memory.add("assistant", response)
        return response

    def analyze_model(self):
        """Model mimarisini analiz et."""
        model_path = getattr(self.config, 'MODEL_PATH', '')
        if not os.path.exists(model_path):
            return "HATA: Model bulunamadi: " + model_path

        print("  Model inceleniyor...")
        info = self.analyzer.analyze_model(model_path)

        if "error" in info:
            return f"Model analiz hatasi: {info['error']}"

        rag_context = self.rag.get_context(
            "YOLO model mimarisi parametre sayisi katman yapisi"
        )

        model_info = json.dumps(info, ensure_ascii=False, indent=2)
        prompt = MODEL_ANALYSIS_TEMPLATE.format(
            model_info=model_info,
            rag_context=rag_context,
        )

        print("\n")
        response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        self.memory.add("user", "Model analizi istendi")
        self.memory.add("assistant", response)
        return response

    def ask(self, question):
        """Serbest soru sor (RAG destekli)."""
        rag_context = self.rag.get_context(question)

        prompt = TEACHING_TEMPLATE.format(
            question=question,
            rag_context=rag_context,
        )

        # Onceki konusma baglamini ekle
        context_summary = self.memory.get_context_summary()
        if context_summary:
            prompt = context_summary + "\n\n" + prompt

        self.memory.add("user", question)
        response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        self.memory.add("assistant", response)
        return response

    def chat(self, message):
        """Multi-turn konusma (hafizali)."""
        rag_context = self.rag.get_context(message)

        self.memory.add("user", message)
        messages = self.memory.get_messages()

        # RAG baglamini son mesaja ekle
        if rag_context:
            enhanced_msg = f"{message}\n\n{rag_context}"
            messages[-1] = {"role": "user", "content": enhanced_msg}

        response = self.llm.chat(messages, system=SYSTEM_PROMPT)
        self.memory.add("assistant", response)
        return response

    def auto_analyze(self):
        """Tam otomatik analiz — dataset + training + model."""
        print("\n" + "=" * 55)
        print("  OTOMATIK ANALIZ BASLATILIYOR")
        print("=" * 55)

        parts = []

        # 1. Dataset
        print("\n[1/3] Dataset analizi...")
        dataset_path = getattr(self.config, 'DATASET_PATH', '')
        if os.path.exists(dataset_path):
            result = self.analyze_dataset()
            parts.append(("Dataset", result))
        else:
            print("  Dataset yolu bulunamadi, atlanıyor.")

        # 2. Training
        print("\n[2/3] Egitim analizi...")
        csv_path = DataAnalyzer.find_results_csv()
        if csv_path:
            result = self.analyze_training(csv_path)
            parts.append(("Egitim", result))
        else:
            print("  results.csv bulunamadi, atlanıyor.")

        # 3. Model
        print("\n[3/3] Model analizi...")
        model_path = getattr(self.config, 'MODEL_PATH', '')
        if os.path.exists(model_path):
            result = self.analyze_model()
            parts.append(("Model", result))
        else:
            print("  Model bulunamadi, atlanıyor.")

        print("\n" + "=" * 55)
        print("  OTOMATIK ANALIZ TAMAMLANDI")
        print("=" * 55)

        return parts

    def get_status(self):
        """Agent durumunu goster."""
        kb_stats = self.rag.get_stats()
        model_info = self.llm.get_model_info()
        return {
            "model": model_info,
            "knowledge_base": kb_stats,
            "memory_turns": self.memory.turn_count,
        }


def run_agent(config_module):
    """Interaktif CV Agent oturumu."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))
    import config as cfg

    print("=" * 58)
    print("  YOLO CV UZMAN AGENT")
    print("  Yerel AI — Ollama + RAG")
    print("=" * 58)

    # Agent olustur
    agent = CVAgent(cfg)

    # Ollama kontrol
    print("\n  Ollama kontrol ediliyor...")
    if not agent.llm.check_and_pull():
        return

    # Bilgi tabani kontrol
    kb_stats = agent.rag.get_stats()
    if kb_stats["total_chunks"] > 0:
        print(f"  Bilgi tabani: {kb_stats['total_chunks']} parca yuklu")
    else:
        print("  Bilgi tabani bos — ilk kullanim icin olusturuluyor...")
        print("  (Bu islem ilk seferinde birkaç dakika surebilir)")
        agent.rag._load_or_build_index()
        kb_stats = agent.rag.get_stats()
        print(f"  Bilgi tabani hazir: {kb_stats['total_chunks']} parca")

    print(f"  Model: {cfg.AGENT_MODEL}")
    print("=" * 58)

    while True:
        print("\nNe yapmak istiyorsun?")
        print("  1. Dataset analizi (sinif dagilimi, denge, oneriler)")
        print("  2. Egitim sonuclari analizi (results.csv)")
        print("  3. Model mimari analizi")
        print("  4. Serbest soru sor (RAG destekli)")
        print("  5. Sohbet modu (coklu tur konusma)")
        print("  6. Tam otomatik analiz (hepsi)")
        print("  7. Agent durumu")
        print("  8. Cikis")

        choice = input("\nSecimin (1-8): ").strip()

        if choice == "1":
            print("\nDataset analiz ediliyor...\n")
            agent.analyze_dataset()

        elif choice == "2":
            csv_path = input(
                "results.csv yolu (veya Enter ile otomatik bul): "
            ).strip()
            if not csv_path:
                csv_path = None
            agent.analyze_training(csv_path)

        elif choice == "3":
            print("\nModel analiz ediliyor...\n")
            agent.analyze_model()

        elif choice == "4":
            question = input("\nSorun: ").strip()
            if question:
                print("\nDusunuyor...\n")
                agent.ask(question)

        elif choice == "5":
            print("\nSohbet modu — 'cikis' yazarak menu'ye don.")
            print("-" * 40)
            while True:
                msg = input("\nSen: ").strip()
                if not msg or msg.lower() in ('cikis', 'exit', 'quit', 'q'):
                    print("Sohbet modu sonlandirildi.")
                    break
                print()
                agent.chat(msg)

        elif choice == "6":
            agent.auto_analyze()

        elif choice == "7":
            status = agent.get_status()
            print("\n--- Agent Durumu ---")
            print(f"  Model: {status['model'].get('model', 'N/A')}")
            kb = status['knowledge_base']
            print(f"  Bilgi Tabani: {kb['total_chunks']} parca")
            print(f"  Hafiza: {status['memory_turns']} tur konusma")

        elif choice == "8":
            print("Agent kapatiliyor.")
            break

        else:
            print("Gecersiz secim. 1-8 arasi bir sayi gir.")
