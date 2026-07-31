class NLPProcessor:
    def process_text(self, text: str):
        return [token.strip().lower() for token in text.split(",")]