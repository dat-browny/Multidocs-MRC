import itertools
import re
import unicodedata as ud


__specials__ = [r"==>", r"->", r"\.\.\.", r">>"]
__digit__ = r"\d+([\.,_]\d+)+"
__email__ = r"([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
__web__ = r"\w+://[^\s]+"
__word__ = r"\w+"
__non_word__ = r"[^\w\s]"
__abbreviations__ = [
    r"[A-ZĐ]+\.",
    r"[Tt]p\.",
    r"[Mm]r\.",
    r"[Mm]rs\.",
    r"[Mm]s\.",
    r"[Dd]r\.",
    r"[Tt]h[sS]\.",
]

__PATTERNS__ = (
    __abbreviations__
    + __specials__
    + [__web__, __email__, __digit__, __non_word__, __word__]
)
__REGEX_PATTERNS__ = "(" + "|".join(__PATTERNS__) + ")"


def normalize(text):
    """Hàm chuẩn hóa dữ liệu, xóa các khoảng trắng thừa đi."""
    text = ud.normalize("NFC", text)
    return " ".join(text.split())


def flatten_list(l):
    """Làm phẳng một danh sách hai chiều không đồng nhất, ví dụ:
    [[1,2,3,4], [1,3], [5], [6,7,8]] --> [1, 2, 3, 4, 1, 3, 5, 6, 7, 8]
    """
    return list(itertools.chain.from_iterable(l))


def sylabelize(text):
    """Hàm tách các dấu câu đặc biệt ra khỏi chữ cái.

    Ví dụ: "Tuấn." -> "Tuấn ."
    """
    text = ud.normalize("NFC", text)
    tokens = re.findall(__REGEX_PATTERNS__, text, re.UNICODE)

    return " ".join([token[0] for token in tokens])


def remove_accents(s):
    """Xóa dấu của một câu.

    Ví dụ: Xóa dấu -> Xoa dau
    """
    s = re.sub("Đ", "D", s)
    s = re.sub("đ", "d", s)
    s = ud.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    return s


ALL_CHAR_ACCENTS = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"


def sentence_tokenize(context):
    """Tách câu."""
    def refine_sentences(sentences):
        if len(sentences) > 1:
            for i, s in enumerate(sentences[:-1]):
                if re.fullmatch(r"[0-9a-zA-Z]{1,4}\.", s):
                    sentences[i + 1] = s + " " + sentences[i + 1]
                    sentences[i] = ""

        sentences = [s for s in sentences if s != ""]

        if len(sentences) > 1:
            for i in range(1, len(sentences)):
                if len(sentences[i]) < 5:
                    sentences[i - 1] = sentences[i - 1] + " " + sentences[i]
                    sentences[i] = ""
        return [s for s in sentences if s != ""]

    vi_alpha = f"[A-Z{ALL_CHAR_ACCENTS.upper()}]"
    vi_alpha_lower = vi_alpha.lower()
    split_regex = f'(?<![A-Z][a-z]\.)(?<!{vi_alpha}\.)'
    split_regex += r'(?<=\.)\s'
    split_regex += f'(?![0-9])(?!{vi_alpha_lower})'
    split_regex += r'(?![\(])(?![\{])(?![\[])'
    contexts = [re.split(split_regex, s) for s in context.splitlines()]
    contexts = [refine_sentences(sentences) for sentences in contexts]
    contexts = flatten_list(contexts)
    return contexts
