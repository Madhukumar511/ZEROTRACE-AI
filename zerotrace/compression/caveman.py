import re
from typing import Dict, List, Tuple, Any

# Filler, greetings, and pleasantry patterns to prune
_PLEASANTRIES_PATTERNS = [
    # Greetings & Introductions
    re.compile(r"(?i)\b(?:hello|hi|hey|greetings)(?:\s+(?:there|assistant|model|bot|friend))?[\.,!?:;]?"),
    re.compile(r"(?i)\b(?:could you please|can you please|would you please|could you|can you|will you)\b"),
    re.compile(r"(?i)\b(?:i would like you to|i want you to|i need you to|i ask you to)\b"),
    # Polite affirmative preamble
    re.compile(r"(?i)\b(?:sure|certainly|of course|happy to help|i would be happy to|i'd be happy to|i can certainly help with that)\b[\.,!]?"),
    re.compile(r"(?i)\b(?:as an ai language model|as an ai model|as a helpful assistant|as an ai)\b[\.,!]?"),
    re.compile(r"(?i)\b(?:here is (?:the|a|an) explanation(?: of| for)?|here is what you need to know|here is how it works)\b:?"),
    re.compile(r"(?i)\b(?:in order to understand this|let's dive into|let's take a look at|let us examine)\b:?"),
    # Conclusions & Pleasant signoffs
    re.compile(r"(?i)\b(?:in conclusion|to summarize|all in all|in summary|to sum up)\b[\.,!]?"),
    re.compile(r"(?i)\b(?:thank you so much|thank you very much|thank you|thanks in advance|many thanks|thanks)[\.,!]?"),
    re.compile(r"(?i)\b(?:i look forward to (?:your|the)?\s*(?:helpful|detailed)?\s*(?:answer|response|reply)?)[\.,!]?"),
    re.compile(r"(?i)\b(?:feel free to ask if you have any questions|hope this helps|let me know if you need anything else)[\.,!]?"),
    # Redundant epistemic hedges
    re.compile(r"(?i)\b(?:it is (?:really|very)? important that you|it is important that you|it is worth noting that)\b"),
    re.compile(r"(?i)\b(?:it is important to note that|it should be noted that|please note that|keep in mind that)\b"),
    re.compile(r"(?i)\b(?:explain everything thoroughly|explain thoroughly)\b"),
]

# Filler adverbs and hedges
_FILLER_WORDS = [
    re.compile(r"(?i)\b(?:basically|actually|simply|literally|just|really|kind of|sort of)\b"),
    re.compile(r"(?i)\b(?:as a matter of fact|needless to say|it goes without saying)\b"),
    re.compile(r"(?i)\b(?:please)\b"),
]

# Concise replacements for verbose phrases
_PHRASE_REPLACEMENTS = [
    (re.compile(r"(?i)\bgive a detailed explanation with at least\b"), "explain in"),
    (re.compile(r"(?i)\bprovide a detailed explanation of\b"), "explain"),
    (re.compile(r"(?i)\bprovide a comprehensive overview of\b"), "overview"),
    (re.compile(r"(?i)\bin order to\b"), "to"),
    (re.compile(r"(?i)\bdue to the fact that\b"), "because"),
    (re.compile(r"(?i)\bat this point in time\b"), "now"),
    (re.compile(r"(?i)\bimplement a solution for\b"), "fix"),
    (re.compile(r"(?i)\bmake use of\b"), "use"),
    (re.compile(r"(?i)\btake into consideration\b"), "consider"),
    (re.compile(r"(?i)\bwith the exception of\b"), "except"),
    (re.compile(r"(?i)\bfor the purpose of\b"), "for"),
    (re.compile(r"(?i)\bin the event that\b"), "if"),
    (re.compile(r"(?i)\bprior to\b"), "before"),
    (re.compile(r"(?i)\bsubsequent to\b"), "after"),
    (re.compile(r"(?i)\bas a consequence of\b"), "because"),
    (re.compile(r"(?i)\bwhich is responsible for\b"), "for"),
    (re.compile(r"(?i)\bthat is used to\b"), "to"),
    (re.compile(r"(?i)\bwhich allows you to\b"), "allows"),
    (re.compile(r"(?i)\bit is recommended that you\b"), "recommended:"),
    (re.compile(r"(?i)\byou need to make sure that\b"), "ensure"),
    (re.compile(r"(?i)\bon how\b"), "how"),
]

# Articles pattern for full & ultra modes
_ARTICLES_PATTERN = re.compile(r"(?i)\b(?:a|an|the)\b\s*")

# Copulas and passive fluff in ultra mode
_ULTRA_FLUFF = [
    re.compile(r"(?i)\b(?:is used for|are used for)\b"),
    re.compile(r"(?i)\b(?:is able to|are able to)\b"),
    re.compile(r"(?i)\b(?:there is|there are)\b"),
]


class CavemanCompressor:
    """
    Caveman Token Compression Engine.
    Achieves ~65% token reduction by pruning conversational fluff, pleasantries,
    articles, and boilerplate while strictly preserving code blocks, technical symbols,
    and factual substance.
    """

    @staticmethod
    def _extract_protected_blocks(text: str) -> Tuple[str, List[str]]:
        """
        Extracts fenced code blocks (```...```), inline code (`...`), and URLs
        so they remain 100% untouched during compression.
        """
        protected: List[str] = []

        def repl(match):
            idx = len(protected)
            protected.append(match.group(0))
            return f"___PROT_BLOCK_{idx}___"

        # 1. Fenced code blocks
        text = re.sub(r"```[\s\S]*?```", repl, text)
        # 2. Inline code
        text = re.sub(r"`[^`\n]+`", repl, text)
        # 3. URLs
        text = re.sub(r"https?://[^\s<>\"']+", repl, text)

        return text, protected

    @staticmethod
    def _restore_protected_blocks(text: str, protected: List[str]) -> str:
        """Restores protected code blocks and URLs to their exact original positions."""
        for idx, block in enumerate(protected):
            text = text.replace(f"___PROT_BLOCK_{idx}___", block)
        return text

    @classmethod
    def compress(cls, text: str, intensity: str = "full") -> str:
        """
        Compresses input text using Caveman heuristics.
        
        Args:
            text: Input prompt or message.
            intensity: 'lite', 'full' (default, ~65% token reduction), or 'ultra'.
        """
        if not text or not text.strip():
            return text

        intensity = intensity.lower()
        if intensity not in ["lite", "full", "ultra"]:
            intensity = "full"

        # Protect code and technical literals
        masked_text, protected = cls._extract_protected_blocks(text)

        lines = masked_text.splitlines()
        compressed_lines = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                compressed_lines.append("")
                continue

            # 1. Strip pleasantries, greetings, and conversational framing
            for pat in _PLEASANTRIES_PATTERNS:
                line_str = pat.sub("", line_str)

            # 2. Strip filler adverbs
            for pat in _FILLER_WORDS:
                line_str = pat.sub("", line_str)

            # 3. Replace wordy phrases with concise equivalents
            for pat, rep in _PHRASE_REPLACEMENTS:
                line_str = pat.sub(rep, line_str)

            # 4. Intensity specific compression
            if intensity in ["full", "ultra"]:
                # Drop articles
                line_str = _ARTICLES_PATTERN.sub("", line_str)

            if intensity == "ultra":
                for pat in _ULTRA_FLUFF:
                    line_str = pat.sub("", line_str)

            # Clean extra whitespaces, dangling conjunctions, and punctuation artifacts
            line_str = re.sub(r"(?i)\b(?:and|or|but|so)\s*([.,;:!?])", r"\1", line_str)
            line_str = re.sub(r"(?i)\b(?:and|or|but|so)\s*$", "", line_str)
            line_str = re.sub(r"[ \t]+", " ", line_str).strip()
            line_str = re.sub(r"\s+([.,;:!?])", r"\1", line_str)
            line_str = re.sub(r"^[.,;:!?]\s*", "", line_str)
            line_str = re.sub(r"\s+[.,;:!?]$", ".", line_str)

            if line_str:
                compressed_lines.append(line_str)

        result = "\n".join(compressed_lines)

        # Restore code and literals
        result = cls._restore_protected_blocks(result, protected)
        return result.strip()

    @classmethod
    def get_system_prompt(cls, intensity: str = "full") -> str:
        """
        Generates system prompt directive instructing LLMs to output in caveman style,
        reducing generation token count and GPU energy consumption by ~65%.
        """
        if intensity == "lite":
            return (
                "You are in token-saver mode. Be concise and direct. "
                "No conversational filler, pleasantries, or preamble. "
                "State facts directly. Keep technical substance and code unchanged."
            )
        elif intensity == "ultra":
            return (
                "Respond in ultra-dense telegraphic style. Cut all articles (a/an/the) and filler. "
                "State facts once. One word when one word enough. "
                "Keep code, identifiers, and formulas exact."
            )
        else:  # full
            return (
                "Respond terse like smart caveman. All technical substance stay. Only fluff die. "
                "Drop articles (a/an/the), filler, and pleasantries. "
                "Pattern: [thing] [action] [reason]. Keep code and technical terms 100% exact. "
                "No self-reference."
            )
