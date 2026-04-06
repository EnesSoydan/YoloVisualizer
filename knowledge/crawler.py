"""
CV Knowledge Crawler
====================
DuckDuckGo ve arXiv'den Computer Vision / Object Detection icerigi ceker,
HTML'i temiz markdown'a donusturur.

Kullanim (dogrudan degil, pipeline.py tarafindan import edilir):
    from knowledge.crawler import WebCrawler, SEARCH_QUERIES
"""

import time
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import urllib.robotparser


# --- Sabitler ---
EARLIEST_YEAR = 2022        # Bu yildan eski icerik alinmaz
RATE_LIMIT_SECONDS = 1.5
DDG_RATE_LIMIT_SECONDS = 2.0   # DDG arama cagrisi oncesi ek bekleme
ARXIV_MAX_RESULTS = 3
DDG_MAX_RESULTS = 5
REQUEST_TIMEOUT = 15
MAX_CONTENT_CHARS = 80_000
MIN_WORD_COUNT = 80             # Bu altindaki icerik atlanir

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# --- Arama sorgu listesi: (sorgu, kaynak_tipi) ---
# kaynak_tipi: "both" = DDG + arXiv, "web" = sadece DDG, "arxiv" = sadece arXiv
SEARCH_QUERIES: Dict[str, List[Tuple[str, str]]] = {
    "yolo_docs": [
        ("YOLOv8 segmentation polygon format dataset training", "both"),
        ("YOLO bbox vs segmentation annotation format difference", "web"),
        ("Ultralytics YOLO model task selection detection segmentation", "web"),
        ("YOLOv8 instance segmentation training tutorial guide", "both"),
        ("YOLOv11 segmentation model training inference", "both"),
        ("YOLO OBB oriented bounding box training aerial", "both"),
        ("YOLOv8 export TensorRT ONNX deployment", "web"),
        ("YOLO transfer learning fine-tuning custom dataset", "web"),
        ("YOLOv8 vs YOLOv9 vs YOLOv10 vs YOLOv11 comparison", "both"),
        ("YOLO pose estimation keypoint detection training", "both"),
    ],
    "architectures": [
        ("DETR transformer object detection end-to-end", "both"),
        ("Faster RCNN region proposal network architecture", "both"),
        ("EfficientDet scalable object detection BiFPN", "both"),
        ("RT-DETR real-time detection transformer", "arxiv"),
        ("CenterNet anchor-free object detection keypoints", "both"),
        ("FCOS fully convolutional one-stage detection", "both"),
        ("SAM segment anything model Meta AI", "both"),
        ("DINO self-supervised vision transformer detection", "arxiv"),
        ("Swin Transformer object detection backbone", "both"),
        ("Co-DETR collaborative detection transformer", "arxiv"),
    ],
    "cv_fundamentals": [
        ("attention mechanism computer vision self-attention", "both"),
        ("feature pyramid network FPN multi-scale detection", "both"),
        ("depthwise separable convolution MobileNet efficiency", "web"),
        ("batch normalization deep learning training stability", "web"),
        ("non-maximum suppression NMS object detection", "web"),
        ("IoU intersection over union bounding box regression", "web"),
        ("anchor boxes prior boxes object detection explained", "web"),
        ("receptive field convolution neural network", "web"),
    ],
    "training_guide": [
        ("object detection data augmentation mosaic mixup", "both"),
        ("transfer learning fine-tuning object detection pretrained", "both"),
        ("mixed precision training FP16 AMP object detection", "web"),
        ("cosine annealing learning rate schedule detection", "web"),
        ("mAP mean average precision evaluation detection", "web"),
        ("COCO benchmark object detection leaderboard 2024", "web"),
        ("label smoothing focal loss class imbalance detection", "both"),
    ],
    "general": [
        ("instance segmentation deep learning Mask RCNN", "both"),
        ("video object detection temporal consistency", "both"),
        ("panoptic segmentation unified detection", "both"),
        ("object detection survey 2024 deep learning", "arxiv"),
        ("real-time object detection edge deployment", "both"),
        ("3D object detection point cloud LiDAR", "arxiv"),
        ("open vocabulary object detection CLIP", "arxiv"),
    ],
}


@dataclass
class CrawlResult:
    url: str
    title: str
    content: str          # Temiz markdown metni
    topic: str            # "architectures" | "cv_fundamentals" | "training_guide" | "general"
    source_type: str      # "web" | "arxiv"
    fetched_at: str       # ISO-8601 timestamp
    word_count: int


class WebCrawler:
    """DuckDuckGo ve arXiv'den CV icerigi ceken, HTML'i markdown'a donusturan sinif."""

    def __init__(self, rate_limit: float = RATE_LIMIT_SECONDS):
        self._rate_limit = rate_limit
        self._robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fetch_ddg_results(
        self,
        query: str,
        topic: str,
        max_results: int = DDG_MAX_RESULTS,
    ) -> List[CrawlResult]:
        """DuckDuckGo aramasiyla ilgili sayfayi cek ve markdown'a donustur."""
        results: List[CrawlResult] = []
        try:
            from duckduckgo_search import DDGS
            from duckduckgo_search.exceptions import DuckDuckGoSearchException
        except ImportError:
            print("  [Crawler] HATA: duckduckgo-search yuklu degil. pip install duckduckgo-search")
            return results

        time.sleep(DDG_RATE_LIMIT_SECONDS)

        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results, timelimit='y'))
        except Exception as exc:
            # Rate limit veya baska DDG hatasi
            print(f"  [Crawler] DDG arama hatasi ({query[:40]}): {exc}")
            time.sleep(30)
            try:
                with DDGS() as ddgs:
                    hits = list(ddgs.text(query, max_results=max_results, timelimit='y'))
            except Exception:
                return results

        for hit in hits:
            url = hit.get("href", "")
            title = hit.get("title", "")
            if not url:
                continue
            result = self._fetch_and_convert(url, title, topic, source_type="web")
            if result:
                results.append(result)
            time.sleep(self._rate_limit)

        return results

    def fetch_arxiv_results(
        self,
        query: str,
        topic: str,
        max_results: int = ARXIV_MAX_RESULTS,
    ) -> List[CrawlResult]:
        """arXiv API'sinden paper abstract + metadata'sini markdown olarak cek."""
        results: List[CrawlResult] = []
        try:
            import arxiv
        except ImportError:
            print("  [Crawler] HATA: arxiv yuklu degil. pip install arxiv")
            return results

        # Sadece Bilgisayar Bilimi kategorilerini getir
        cs_query = f"{query} AND (cat:cs.CV OR cat:cs.AI OR cat:cs.LG OR cat:cs.RO OR cat:cs.NE)"

        try:
            search = arxiv.Search(
                query=cs_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,  # En yeni once
            )
            papers = list(search.results())
        except Exception as exc:
            print(f"  [Crawler] arXiv arama hatasi ({query[:40]}): {exc}")
            return results

        for paper in papers:
            # cs.* kategorisi olmayan paper'lari atla (astronomi, fizik vs.)
            if not any(cat.startswith("cs.") for cat in paper.categories):
                continue
            # Eski paper'lari atla
            if paper.published and paper.published.year < EARLIEST_YEAR:
                continue
            try:
                authors = ", ".join(a.name for a in paper.authors[:6])
                if len(paper.authors) > 6:
                    authors += " et al."
                categories = ", ".join(paper.categories)
                pub_date = paper.published.date() if paper.published else "N/A"
                content = (
                    f"# {paper.title}\n\n"
                    f"**Authors:** {authors}\n"
                    f"**Published:** {pub_date}\n"
                    f"**arXiv ID:** {paper.entry_id}\n"
                    f"**Categories:** {categories}\n\n"
                    f"## Abstract\n\n{paper.summary}\n"
                )
                word_count = len(content.split())
                result = CrawlResult(
                    url=paper.entry_id,
                    title=paper.title,
                    content=content,
                    topic=topic,
                    source_type="arxiv",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    word_count=word_count,
                )
                results.append(result)
            except Exception as exc:
                print(f"  [Crawler] arXiv paper isleme hatasi: {exc}")
            time.sleep(self._rate_limit)

        return results

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _fetch_and_convert(
        self,
        url: str,
        title: str,
        topic: str,
        source_type: str = "web",
    ) -> Optional[CrawlResult]:
        """URL'yi cek, HTML'i markdown'a donustur. Basarisizsa None dondur."""
        if not self._is_robots_allowed(url):
            return None

        try:
            import requests
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
            if resp.status_code != 200:
                return None
            html = resp.text[:MAX_CONTENT_CHARS]
        except Exception:
            return None

        markdown = self._html_to_markdown(html)
        word_count = len(markdown.split())
        if word_count < MIN_WORD_COUNT:
            return None

        # Baslik yoksa ilk satiri kullan
        if not title:
            first_line = markdown.strip().split("\n")[0]
            title = first_line.lstrip("#").strip()[:120] or url

        return CrawlResult(
            url=url,
            title=title,
            content=markdown,
            topic=topic,
            source_type=source_type,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            word_count=word_count,
        )

    def _html_to_markdown(self, html: str) -> str:
        """HTML'i temizle ve markdown'a donustur."""
        try:
            from bs4 import BeautifulSoup
            import markdownify

            soup = BeautifulSoup(html, "html.parser")

            # Gereksiz etiketleri kaldir
            for tag in soup.find_all(
                ["script", "style", "nav", "footer", "header", "aside",
                 "noscript", "iframe", "form", "button"]
            ):
                tag.decompose()
            for tag in soup.find_all(True, class_=re.compile(
                r"cookie|banner|popup|modal|sidebar|ad-|advertisement|social|share",
                re.I,
            )):
                tag.decompose()

            # Ana icerigi bul
            content_el = (
                soup.find("main")
                or soup.find("article")
                or soup.find(id=re.compile(r"content|main|article", re.I))
                or soup.find("body")
                or soup
            )

            raw_md = markdownify.markdownify(
                str(content_el),
                heading_style="ATX",
                bullets="-",
                strip=["a", "img"],
            )

            # Uc veya daha fazla ardisik bos satiri iki satira indir
            cleaned = re.sub(r"\n{3,}", "\n\n", raw_md)
            return cleaned.strip()

        except Exception:
            # Fallback: duz metin
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator="\n").strip()
            except Exception:
                return ""

    def _is_robots_allowed(self, url: str) -> bool:
        """robots.txt kontrolu. Erisilemezse izin ver (fail-open)."""
        try:
            parsed = urlparse(url)
            domain = f"{parsed.scheme}://{parsed.netloc}"
            if domain not in self._robots_cache:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(f"{domain}/robots.txt")
                try:
                    rp.read()
                except Exception:
                    # robots.txt alinamazsa izin ver
                    self._robots_cache[domain] = None
                    return True
                self._robots_cache[domain] = rp
            rp = self._robots_cache.get(domain)
            if rp is None:
                return True
            return rp.can_fetch(HEADERS["User-Agent"], url)
        except Exception:
            return True
