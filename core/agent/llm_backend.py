"""
LLM Backend — Ollama Istemci
=============================
Ollama uzerinden yerel LLM ile iletisim.
Model kontrolu, otomatik indirme ve stream destegi.
"""

import sys

try:
    import ollama
except ImportError:
    print("HATA: ollama paketi gerekli: pip install ollama")
    sys.exit(1)


class LLMBackend:
    """Ollama LLM istemcisi."""

    def __init__(self, model_name, base_url="http://localhost:11434",
                 temperature=0.3, num_ctx=8192):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.client = ollama.Client(host=base_url)

    def check_and_pull(self):
        """Model mevcut mu kontrol et, yoksa indir."""
        try:
            # Ollama'nin calistigini kontrol et
            self.client.list()
        except Exception:
            print("\n" + "=" * 55)
            print("  HATA: Ollama calismiyor!")
            print("=" * 55)
            print("  Ollama'yi baslat:")
            print("    ollama serve")
            print("\n  Kurulu degilse:")
            print("    https://ollama.com/download/windows")
            print("=" * 55)
            return False

        # Model mevcut mu?
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models] if models.models else []

            # Tam isim veya kisa isim ile eslesme kontrol et
            found = False
            for name in model_names:
                if self.model_name in name or name in self.model_name:
                    found = True
                    break

            if not found:
                print(f"\n  Model indiriliyor: {self.model_name}")
                print("  (Bu ilk seferinde biraz zaman alabilir...)\n")
                self.client.pull(self.model_name)
                print("  Model basariyla indirildi!")

            return True

        except Exception as e:
            print(f"\n  Model kontrol hatasi: {e}")
            print(f"  Manuel indirmek icin: ollama pull {self.model_name}")
            return False

    def generate(self, prompt, system=None, stream=True):
        """Tek seferlik metin uretimi."""
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }

        if stream:
            return self._generate_stream(prompt, system, options)
        else:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system or "",
                options=options,
            )
            return response.response

    def _generate_stream(self, prompt, system, options):
        """Stream modunda metin uret ve ekrana yaz."""
        full_response = []
        stream = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system=system or "",
            options=options,
            stream=True,
        )
        for chunk in stream:
            token = chunk.response
            print(token, end="", flush=True)
            full_response.append(token)
        print()  # Son satirdan sonra yeni satir
        return "".join(full_response)

    def chat(self, messages, system=None, stream=True):
        """Multi-turn chat."""
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }

        if stream:
            return self._chat_stream(chat_messages, options)
        else:
            response = self.client.chat(
                model=self.model_name,
                messages=chat_messages,
                options=options,
            )
            return response.message.content

    def _chat_stream(self, messages, options):
        """Stream modunda chat."""
        full_response = []
        stream = self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            stream=True,
        )
        for chunk in stream:
            token = chunk.message.content
            print(token, end="", flush=True)
            full_response.append(token)
        print()
        return "".join(full_response)

    def get_model_info(self):
        """Model bilgilerini dondur."""
        try:
            info = self.client.show(self.model_name)
            return {
                "model": self.model_name,
                "parameters": getattr(info, 'parameters', 'Bilinmiyor'),
                "template": getattr(info, 'template', '')[:100],
            }
        except Exception:
            return {"model": self.model_name, "status": "Bilgi alinamadi"}
