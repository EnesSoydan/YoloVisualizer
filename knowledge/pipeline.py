"""
CV Knowledge Pipeline — Surekli Bilgi Toplama Boru Hatti
=========================================================
Web ve arXiv'den CV icerigi ceker, tekillestirir, .md olarak kaydeder
ve ChromaDB'ye ekler. Arka planda surekli calisabilir.

Kullanim:
    python knowledge/pipeline.py                        # bir kez calistir
    python knowledge/pipeline.py --watch                # her 6 saatte bir
    python knowledge/pipeline.py --watch --interval 12  # her 12 saatte bir
    python knowledge/pipeline.py --stats                # istatistik goster
    python knowledge/pipeline.py --clear-state          # islenen URL'leri sifirla
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Set

# Proje kokunu Python path'ine ekle
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class PipelineState:
    processed_urls: Set[str] = field(default_factory=set)
    last_run_iso: str = ""
    stats: Dict[str, int] = field(default_factory=lambda: {
        "total_fetched": 0,
        "total_saved": 0,
        "total_skipped_url": 0,
        "total_skipped_similarity": 0,
        "total_skipped_short": 0,
        "total_errors": 0,
    })


# Klasor adi → ChromaDB topic degeri eslesmesi
_TOPIC_META = {
    "architectures": "architecture",
    "cv_fundamentals": "fundamentals",
    "training_guide": "training",
    "general": "general",
}

# Icerik benzerlik esigi (bu deger veya ustu → tekrar sayilir, atlanir)
SIMILARITY_THRESHOLD = 0.92
MIN_WORD_COUNT = 100


class KnowledgePipeline:
    STATE_FILE = os.path.join("knowledge", ".pipeline_state.json")

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.state_file = os.path.join(project_root, self.STATE_FILE)
        self.sources_dir = os.path.join(project_root, "knowledge", "sources")
        self.state = self._load_state()
        self.rag = self._init_rag()

        from knowledge.crawler import WebCrawler
        self.crawler = WebCrawler()

    # ------------------------------------------------------------------ #
    #  State yonetimi
    # ------------------------------------------------------------------ #

    def _load_state(self) -> PipelineState:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return PipelineState(
                    processed_urls=set(data.get("processed_urls", [])),
                    last_run_iso=data.get("last_run_iso", ""),
                    stats=data.get("stats", PipelineState().stats),
                )
            except Exception as exc:
                self._log(f"State dosyasi okunamadi, sifirlaniyor: {exc}")
        return PipelineState()

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                "processed_urls": sorted(self.state.processed_urls),
                "last_run_iso": self.state.last_run_iso,
                "stats": self.state.stats,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log(f"State kaydedilemedi: {exc}")

    # ------------------------------------------------------------------ #
    #  RAG baslatma
    # ------------------------------------------------------------------ #

    def _init_rag(self):
        try:
            import config
            persist_dir = config.AGENT_CHROMA_DIR
            embedding_model = config.AGENT_EMBEDDING_MODEL
            sources_dir = os.path.join(config.AGENT_KB_DIR, "sources")
        except ImportError:
            persist_dir = os.path.join(self.project_root, "knowledge", "chroma_db")
            embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            sources_dir = self.sources_dir

        from core.agent.rag_engine import RAGEngine
        rag = RAGEngine(
            persist_dir=persist_dir,
            embedding_model_name=embedding_model,
            sources_dir=sources_dir,
        )
        # Index'i on yukle (ilk sorguyu hizlandirmak icin)
        try:
            rag._load_or_build_index()
        except Exception:
            pass
        return rag

    # ------------------------------------------------------------------ #
    #  Ana pipeline
    # ------------------------------------------------------------------ #

    def run_once(self) -> Dict[str, int]:
        """Tum sorgulari bir kez calistir ve sonuclari isle."""
        from knowledge.crawler import SEARCH_QUERIES

        run_stats: Dict[str, int] = {
            "fetched": 0,
            "saved": 0,
            "skipped_url": 0,
            "skipped_similarity": 0,
            "skipped_short": 0,
            "errors": 0,
        }

        self._log("Pipeline basliyor...")
        total_queries = sum(len(v) for v in SEARCH_QUERIES.values())
        query_num = 0

        for topic, queries in SEARCH_QUERIES.items():
            for query, source_type in queries:
                query_num += 1
                self._log(
                    f"[{query_num}/{total_queries}] {topic} | {source_type} | {query[:55]}"
                )

                results = []
                if source_type in ("web", "both"):
                    results.extend(self.crawler.fetch_ddg_results(query, topic))
                if source_type in ("arxiv", "both"):
                    results.extend(self.crawler.fetch_arxiv_results(query, topic))

                run_stats["fetched"] += len(results)

                for result in results:
                    outcome = self._process_result(result)
                    run_stats[outcome.replace("skipped_", "skipped_")] = (
                        run_stats.get(outcome, 0) + 1
                    )
                    # Genel istatistikleri guncelle
                    stat_key = f"total_{outcome}"
                    if stat_key in self.state.stats:
                        self.state.stats[stat_key] += 1
                    self.state.stats["total_fetched"] = (
                        self.state.stats.get("total_fetched", 0) + (1 if outcome == "saved" else 0)
                    )

        self.state.last_run_iso = datetime.now(timezone.utc).isoformat()
        self._save_state()

        self._log(
            f"Tamamlandi — "
            f"Cekilen: {run_stats['fetched']} | "
            f"Kaydedilen: {run_stats.get('saved', 0)} | "
            f"Atlanan(URL): {run_stats.get('skipped_url', 0)} | "
            f"Atlanan(benzer): {run_stats.get('skipped_similarity', 0)} | "
            f"Atlanan(kisa): {run_stats.get('skipped_short', 0)} | "
            f"Hata: {run_stats.get('errors', 0)}"
        )
        return run_stats

    def _process_result(self, result) -> str:
        """
        Tek bir CrawlResult'i isle.
        Donus: "saved" | "skipped_url" | "skipped_similarity" | "skipped_short" | "error"
        """
        try:
            # 1. URL tekrari kontrolu
            if result.url in self.state.processed_urls:
                return "skipped_url"

            # 2. Uzunluk kontrolu
            if result.word_count < MIN_WORD_COUNT:
                self.state.processed_urls.add(result.url)
                return "skipped_short"

            # 3. Icerik benzerligi kontrolu (ChromaDB)
            if self._is_too_similar(result.content):
                self.state.processed_urls.add(result.url)
                self._save_state()
                return "skipped_similarity"

            # 4. Diske kaydet
            saved_path = self._save_markdown(result)

            # 5. ChromaDB'ye ekle
            meta_topic = _TOPIC_META.get(result.topic, "general")
            ok = self.rag.insert_document(
                text=result.content,
                metadata={
                    "url": result.url,
                    "title": result.title,
                    "topic": meta_topic,
                    "source_type": result.source_type,
                    "fetched_at": result.fetched_at,
                    "file_path": saved_path,
                },
            )
            if not ok:
                # Disk'e yazildi ama index'e eklenemedi — URL'yi kaydetme, bir sonraki calistirmada tekrar dene
                self._log(f"  UYARI: Index eklenemedi: {result.url[:70]}")
                return "error"

            self.state.processed_urls.add(result.url)
            self._save_state()
            self._log(f"  + Kaydedildi: {result.title[:70]}")
            return "saved"

        except Exception as exc:
            self._log(f"  HATA: {result.url[:60]} — {exc}")
            return "error"

    def _is_too_similar(self, content: str) -> bool:
        """
        Ilk 400 kelimeyi ChromaDB'de ara. Benzerlik >= SIMILARITY_THRESHOLD ise True dondur.
        Hata durumunda False dondur (fail-open).
        """
        try:
            probe = " ".join(content.split()[:400])
            if not probe:
                return False
            index = self.rag._load_or_build_index()
            retriever = index.as_retriever(similarity_top_k=1)
            nodes = retriever.retrieve(probe)
            if not nodes:
                return False
            score = nodes[0].score if hasattr(nodes[0], "score") else 0.0
            if score is None:
                return False
            return score >= SIMILARITY_THRESHOLD
        except Exception:
            return False

    def _save_markdown(self, result) -> str:
        """
        Markdown dosyasini uygun topic klasorune kaydet.
        Donus: kaydedilen dosyanin tam yolu.
        """
        topic_dir = os.path.join(self.sources_dir, result.topic)
        os.makedirs(topic_dir, exist_ok=True)

        # Guvenli dosya adi olustur
        slug = re.sub(r"[^a-z0-9_-]", "_", result.title.lower())[:55]
        slug = re.sub(r"_+", "_", slug).strip("_")
        date_str = datetime.now().strftime("%Y%m%d")
        base_name = f"{slug}_{date_str}"

        # Cakisma varsa sayac ekle
        file_path = os.path.join(topic_dir, f"{base_name}.md")
        counter = 2
        while os.path.exists(file_path):
            file_path = os.path.join(topic_dir, f"{base_name}_{counter}.md")
            counter += 1

        front_matter = (
            f"---\n"
            f"title: {result.title}\n"
            f"url: {result.url}\n"
            f"topic: {result.topic}\n"
            f"source_type: {result.source_type}\n"
            f"fetched_at: {result.fetched_at}\n"
            f"word_count: {result.word_count}\n"
            f"---\n\n"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(front_matter + result.content)

        return file_path

    # ------------------------------------------------------------------ #
    #  Watch modu
    # ------------------------------------------------------------------ #

    def run_watch(self, interval_hours: float = 6.0) -> None:
        """Pipeline'i belirli araliklarla surekli calistir."""
        run_count = 0
        self._log(f"Watch modu baslatildi — her {interval_hours} saatte bir calisir.")
        self._log("Durdurmak icin Ctrl+C kullanin.")
        try:
            while True:
                run_count += 1
                self._log(f"=== Run #{run_count} basliyor ===")
                self.run_once()
                self._log(f"=== Run #{run_count} tamamlandi. Sonraki: {interval_hours} saat sonra ===")
                time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            self._log("Kullanici tarafindan durduruldu. State kaydediliyor...")
            self._save_state()
            self._log("Temiz cikis.")

    # ------------------------------------------------------------------ #
    #  Yardimcilar
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")

    def print_stats(self) -> None:
        print("\n" + "=" * 50)
        print("  PIPELINE ISTATISTIKLERI")
        print("=" * 50)
        print(f"  Son calistirma  : {self.state.last_run_iso or 'Hic calistirilmadi'}")
        print(f"  Islenen URL     : {len(self.state.processed_urls)}")
        for key, val in self.state.stats.items():
            print(f"  {key:<28}: {val}")
        try:
            chroma_stats = self.rag.get_stats()
            print(f"  ChromaDB chunk sayisi   : {chroma_stats.get('total_chunks', '?')}")
        except Exception:
            pass
        print("=" * 50)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="CV Knowledge Pipeline — surekli bilgi toplama boru hatti"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Surekli mod: belirtilen araliklarla calistir",
    )
    parser.add_argument(
        "--interval", type=float, default=6.0,
        help="Watch modunda calistirmalar arasi sure (saat, varsayilan: 6)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Mevcut pipeline durumunu ve istatistikleri goster",
    )
    parser.add_argument(
        "--clear-state", action="store_true",
        help="Islenen URL listesini sifirla (her seyi yeniden isle)",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline = KnowledgePipeline(project_root)

    if args.stats:
        pipeline.print_stats()
    elif args.clear_state:
        pipeline.state.processed_urls = set()
        pipeline._save_state()
        print("State sifirlandi. Bir sonraki calistirmada tum icerik yeniden islenecek.")
    elif args.watch:
        pipeline.run_watch(interval_hours=args.interval)
    else:
        pipeline.run_once()


if __name__ == "__main__":
    main()
