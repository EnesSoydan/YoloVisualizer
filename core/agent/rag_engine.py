"""
RAG Engine — Bilgi Tabani Motoru
=================================
LlamaIndex + ChromaDB ile CV bilgi tabanindan ilgili bilgiyi ceker.
"""

import os
import sys


class RAGEngine:
    """ChromaDB + LlamaIndex tabanli bilgi erisim motoru."""

    def __init__(self, persist_dir, embedding_model_name, sources_dir=None):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name
        self.sources_dir = sources_dir
        self._index = None
        self._embed_model = None
        self._initialized = False

    def _get_embed_model(self):
        """Embedding modelini yukle (lazy)."""
        if self._embed_model is None:
            try:
                import os
                # Model zaten indirilmisse HuggingFace'e baglanti gerektirme
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                self._embed_model = HuggingFaceEmbedding(
                    model_name=self.embedding_model_name
                )
            except ImportError:
                print("HATA: llama-index-embeddings-huggingface gerekli")
                print("  pip install llama-index-embeddings-huggingface")
                sys.exit(1)
        return self._embed_model

    def _load_or_build_index(self):
        """Mevcut index'i yukle veya yeniden olustur."""
        if self._index is not None:
            return self._index

        try:
            from llama_index.core import (
                VectorStoreIndex, StorageContext, Settings,
                SimpleDirectoryReader
            )
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb
        except ImportError as e:
            print(f"HATA: Eksik paket: {e}")
            print("  pip install llama-index-core llama-index-vector-stores-chroma chromadb")
            sys.exit(1)

        Settings.embed_model = self._get_embed_model()
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # ChromaDB client (telemetry kapali — posthog Python 3.8 uyumsuzlugu)
        from chromadb.config import Settings as ChromaSettings
        chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Mevcut koleksiyon var mi?
        try:
            chroma_collection = chroma_client.get_collection("cv_knowledge")
            chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(
                vector_store=chroma_store
            )
            self._index = VectorStoreIndex.from_vector_store(
                chroma_store,
                storage_context=storage_context,
            )
            self._initialized = True
            return self._index
        except Exception:
            pass

        # Koleksiyon yok — sources'tan olustur
        if self.sources_dir and os.path.isdir(self.sources_dir):
            return self._build_from_sources()

        # Hicbir sey yoksa bos index
        chroma_collection = chroma_client.get_or_create_collection("cv_knowledge")
        chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=chroma_store
        )
        self._index = VectorStoreIndex(
            [], storage_context=storage_context
        )
        self._initialized = True
        return self._index

    def _build_from_sources(self):
        """Sources dizininden index olustur."""
        from llama_index.core import (
            VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        Settings.embed_model = self._get_embed_model()
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        print("  Bilgi tabani olusturuluyor...")
        print(f"  Kaynak: {self.sources_dir}")

        # Dokumanlari oku
        reader = SimpleDirectoryReader(
            self.sources_dir,
            recursive=True,
            required_exts=[".md", ".txt"],
        )
        documents = reader.load_data()

        if not documents:
            print("  UYARI: Kaynak dizininde dokuman bulunamadi.")
            return self._load_or_build_index()

        # Metadata ekle — klasor adini topic olarak kullan
        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            if "yolo_docs" in file_path:
                doc.metadata["topic"] = "yolo"
            elif "cv_fundamentals" in file_path:
                doc.metadata["topic"] = "fundamentals"
            elif "training_guide" in file_path:
                doc.metadata["topic"] = "training"
            elif "architectures" in file_path:
                doc.metadata["topic"] = "architecture"
            else:
                doc.metadata["topic"] = "general"

        # ChromaDB'ye kaydet (telemetry kapali)
        from chromadb.config import Settings as ChromaSettings
        chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        chroma_collection = chroma_client.get_or_create_collection("cv_knowledge")
        chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=chroma_store
        )

        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
            insert_batch_size=100,
        )
        self._initialized = True

        print(f"  {len(documents)} dokuman islendi ve indexlendi.")
        return self._index

    def build_index(self):
        """Bilgi tabanini (yeniden) olustur. build_kb.py tarafindan cagirilir."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # Mevcut koleksiyonu sil ve yeniden olustur
        chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            chroma_client.delete_collection("cv_knowledge")
        except Exception:
            pass

        self._index = None
        self._initialized = False
        return self._build_from_sources()

    def query(self, question, top_k=5):
        """Soruya en yakin dokuman parcalarini getir."""
        index = self._load_or_build_index()
        if index is None:
            return []

        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
        )
        try:
            response = query_engine.query(question)
            return response
        except Exception:
            return None

    # YOLO konularini iceren sorgularda topic filtresini devreye sokmak icin
    _YOLO_KEYWORDS = {
        "yolo", "ultralytics", "bbox", "bounding box", "polygon", "segmentation",
        "seg", "detection", "etiket", "annotation", "format", "label", "obb",
        "pose", "keypoint", "inference", "model", "weights", "pt", "train",
        "dataset", "class", "sinif", "maske", "mask",
    }

    def get_context(self, question, top_k=4):
        """Soruyla ilgili bilgi tabanindan baglam metni dondur.

        Iki gecis:
        1. Genel semantik arama (top_k)
        2. YOLO sorusuysa topic='yolo' filtreli ek arama (her zaman dahil)
        """
        index = self._load_or_build_index()
        if index is None:
            return ""

        # --- Gecis 1: Genel semantik arama ---
        retriever = index.as_retriever(similarity_top_k=top_k)
        try:
            nodes = list(retriever.retrieve(question))
        except Exception:
            nodes = []

        # --- Gecis 2: YOLO sorularinda topic filtreli boost ---
        question_lower = question.lower()
        is_yolo_question = any(kw in question_lower for kw in self._YOLO_KEYWORDS)

        if is_yolo_question:
            try:
                from llama_index.core.vector_stores import (
                    MetadataFilter, MetadataFilters, FilterOperator
                )
                filters = MetadataFilters(filters=[
                    MetadataFilter(key="topic", value="yolo",
                                   operator=FilterOperator.EQ)
                ])
                yolo_retriever = index.as_retriever(
                    similarity_top_k=4,
                    filters=filters,
                )
                yolo_nodes = yolo_retriever.retrieve(question)

                # Tekrarlari atlayarak birlestir
                seen = {n.node.get_content()[:80] for n in nodes}
                for n in yolo_nodes:
                    preview = n.node.get_content()[:80]
                    if preview not in seen:
                        nodes.append(n)
                        seen.add(preview)
            except Exception:
                pass  # Filtre desteklenmiyorsa sessizce atla

        if not nodes:
            return ""

        context_parts = []
        for i, node in enumerate(nodes, 1):
            text = node.node.get_content().strip()
            if text:
                context_parts.append(f"[Kaynak {i}]: {text}")

        if not context_parts:
            return ""

        return "ILGILI BILGI TABANI ICERIGI:\n" + "\n\n".join(context_parts)

    def insert_document(self, text: str, metadata: dict = None) -> bool:
        """Mevcut index'e tek belge ekle — koleksiyonu silmeden, yeniden olusturmadan."""
        try:
            from llama_index.core import Document, Settings
            Settings.embed_model = self._get_embed_model()
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            index = self._load_or_build_index()
            doc = Document(text=text, metadata=metadata or {})
            index.insert(doc)
            return True
        except Exception as e:
            print(f"  [RAG] insert_document hatasi: {e}")
            return False

    def get_stats(self):
        """Bilgi tabani istatistikleri."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = chroma_client.get_collection("cv_knowledge")
            return {
                "total_chunks": collection.count(),
                "persist_dir": self.persist_dir,
            }
        except Exception:
            return {"total_chunks": 0, "persist_dir": self.persist_dir}
