from __future__ import annotations

import html
import re
from dataclasses import dataclass

import requests
import spacy
import streamlit as st
from requests import RequestException
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

RAW_SAMPLE_URL = (
    "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/"
    "master/data/rst/DEV/wsj_0603.out"
)
EDUS_SAMPLE_URL = (
    "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/"
    "master/data/rst/DEV/wsj_0603.out.edus"
)
SAMPLE_NAME = "wsj_0603.out"
MIN_SEGMENT_TOKENS = 3
SUBORDINATORS = {
    "after",
    "although",
    "because",
    "before",
    "if",
    "since",
    "though",
    "unless",
    "when",
    "while",
}
CLAUSAL_DEPENDENCIES = {"acl", "advcl", "ccomp", "relcl", "xcomp"}
PUNCTUATION_BOUNDARIES = {".", "!", "?", ";"}
DEFAULT_DISCOURSE_SENTENCE = (
    "Third-quarter sales in Europe were exceptionally strong, boosted by "
    "promotional programs and new products - although weaker foreign currencies "
    "reduced the company's earnings."
)
DISCOURSE_MARKERS = {
    ("when",): "TEMPORAL",
    ("after",): "TEMPORAL",
    ("before",): "TEMPORAL",
    ("while",): "TEMPORAL",
    ("once",): "TEMPORAL",
    ("until",): "TEMPORAL",
    ("because",): "CONTINGENCY",
    ("since",): "CONTINGENCY",
    ("as",): "CONTINGENCY",
    ("so",): "CONTINGENCY",
    ("therefore",): "CONTINGENCY",
    ("thus",): "CONTINGENCY",
    ("but",): "COMPARISON",
    ("although",): "COMPARISON",
    ("though",): "COMPARISON",
    ("however",): "COMPARISON",
    ("whereas",): "COMPARISON",
    ("and",): "EXPANSION",
    ("or",): "EXPANSION",
    ("also",): "EXPANSION",
    ("in", "addition"): "EXPANSION",
    ("for", "example"): "EXPANSION",
    ("for", "instance"): "EXPANSION",
}
ARG_BOUNDARY_PUNCTUATION = {"-", "--", ",", ";", ":"}
TEMPORAL_HINTS = {
    "year",
    "years",
    "month",
    "months",
    "week",
    "weeks",
    "day",
    "days",
    "today",
    "yesterday",
    "tomorrow",
    "morning",
    "evening",
    "night",
    "ago",
    "earlier",
    "later",
    "recently",
}
DEFAULT_COREFERENCE_TEXT = (
    "Barack Obama was born in Hawaii. He became the 44th President of the United States. "
    "Obama served two terms, and his presidency drew global attention."
)
COREFERENCE_CLUSTER_COLORS = [
    "#dbeafe",
    "#dcfce7",
    "#fce7f3",
    "#fef3c7",
    "#ede9fe",
    "#fee2e2",
]


@dataclass
class BaselineEdu:
    text: str
    token_indices: list[int]


@dataclass
class MarkerMatch:
    text: str
    category: str
    start_token: int
    end_token: int


@dataclass
class CoreferenceMention:
    text: str
    start_char: int
    end_char: int


@dataclass
class CoreferenceCluster:
    cluster_id: int
    color: str
    mentions: list[CoreferenceMention]


def fetch_text(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


@st.cache_data(show_spinner=False)
def load_sample_data() -> dict[str, object]:
    raw_source = fetch_text(RAW_SAMPLE_URL)
    edus_source = fetch_text(EDUS_SAMPLE_URL)

    raw_lines = [line.strip() for line in raw_source.splitlines() if line.strip()]
    ground_truth_edus = [line.strip() for line in edus_source.splitlines() if line.strip()]

    return {
        "sample_name": SAMPLE_NAME,
        "raw_text": "\n".join(raw_lines),
        "comparison_text": " ".join(raw_lines),
        "ground_truth_edus": ground_truth_edus,
        "source_urls": {
            "raw": RAW_SAMPLE_URL,
            "edus": EDUS_SAMPLE_URL,
        },
    }


@st.cache_resource(show_spinner=False)
def load_spacy_model() -> Language:
    return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner=False)
def load_fastcoref_model():
    try:
        from fastcoref import FCoref
    except ImportError as error:
        raise ImportError("缺少 `fastcoref` 依赖。") from error

    return FCoref(device="cpu")


def should_split_before(token: Token) -> bool:
    if token.text == "--":
        return True

    if token.pos_ == "SCONJ" and token.lower_ in SUBORDINATORS:
        return True

    if token.pos_ in {"VERB", "AUX"} and token.dep_ in CLAUSAL_DEPENDENCIES:
        return True

    return False


def should_split_after(token: Token) -> bool:
    return token.text in PUNCTUATION_BOUNDARIES


def label_sentence_starts(sentence: Span) -> dict[int, int]:
    tokens = [token for token in sentence if not token.is_space]
    if not tokens:
        return {}

    start_labels = {token.i: 0 for token in tokens}
    start_labels[tokens[0].i] = 1
    current_segment_start = 0

    for index, token in enumerate(tokens[1:], start=1):
        if should_split_before(token) and (index - current_segment_start) >= MIN_SEGMENT_TOKENS:
            start_labels[token.i] = 1
            current_segment_start = index
            continue

        has_next_token = index < len(tokens) - 1
        if has_next_token and should_split_after(token) and (
            (index - current_segment_start + 1) >= MIN_SEGMENT_TOKENS
        ):
            next_token = tokens[index + 1]
            start_labels[next_token.i] = 1
            current_segment_start = index + 1

    return start_labels


def build_sequence_labels(doc: Doc) -> dict[int, int]:
    start_labels: dict[int, int] = {}

    for sentence in doc.sents:
        start_labels.update(label_sentence_starts(sentence))

    return start_labels


def labels_to_segments(doc: Doc, start_labels: dict[int, int]) -> list[BaselineEdu]:
    baseline_edus: list[BaselineEdu] = []

    for sentence in doc.sents:
        tokens = [token for token in sentence if not token.is_space]
        if not tokens:
            continue

        current_segment: list[Token] = []
        for token in tokens:
            if start_labels.get(token.i, 0) == 1 and current_segment:
                span = doc[current_segment[0].i : current_segment[-1].i + 1]
                baseline_edus.append(
                    BaselineEdu(
                        text=span.text.strip(),
                        token_indices=[segment_token.i for segment_token in current_segment],
                    )
                )
                current_segment = [token]
            else:
                current_segment.append(token)

        if current_segment:
            span = doc[current_segment[0].i : current_segment[-1].i + 1]
            baseline_edus.append(
                BaselineEdu(
                    text=span.text.strip(),
                    token_indices=[segment_token.i for segment_token in current_segment],
                )
            )

    return baseline_edus


def build_baseline_segments(
    text: str,
    nlp: Language,
) -> tuple[Doc, list[BaselineEdu], dict[int, int]]:
    doc = nlp(text)
    start_labels = build_sequence_labels(doc)
    baseline_edus = labels_to_segments(doc, start_labels)
    return doc, baseline_edus, start_labels


def render_baseline_text(segment: BaselineEdu, doc: Doc, start_labels: dict[int, int]) -> str:
    pieces: list[str] = []

    for token_index in segment.token_indices:
        token = doc[token_index]
        token_html = html.escape(token.text)
        if start_labels.get(token_index, 0) == 1:
            token_html = f"<span class='boundary-token'>{token_html}</span>"

        pieces.append(token_html)
        pieces.append(html.escape(token.whitespace_))

    return "".join(pieces).strip()


def render_card(card_index: int, title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="edu-card">
            <div class="edu-card-title">{html.escape(title)} {card_index}</div>
            <div class="edu-card-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ground_truth_cards(ground_truth_edus: list[str]) -> None:
    for index, edu in enumerate(ground_truth_edus, start=1):
        render_card(index, "Ground Truth EDU", html.escape(edu))


def render_baseline_cards(
    baseline_edus: list[BaselineEdu],
    doc: Doc,
    start_labels: dict[int, int],
) -> None:
    for index, edu in enumerate(baseline_edus, start=1):
        render_card(
            index,
            "Baseline EDU",
            render_baseline_text(edu, doc, start_labels),
        )


def find_explicit_marker_matches(doc: Doc) -> list[MarkerMatch]:
    matches: list[MarkerMatch] = []
    tokens = list(doc)
    token_lowers = [token.lower_ for token in tokens]
    marker_patterns = sorted(DISCOURSE_MARKERS.items(), key=lambda item: len(item[0]), reverse=True)
    index = 0

    while index < len(tokens):
        matched = False
        for phrase_tokens, category in marker_patterns:
            phrase_length = len(phrase_tokens)
            if token_lowers[index : index + phrase_length] == list(phrase_tokens):
                span = doc[index : index + phrase_length]
                resolved_category = resolve_marker_category(span, category)
                matches.append(
                    MarkerMatch(
                        text=span.text,
                        category=resolved_category,
                        start_token=index,
                        end_token=index + phrase_length,
                    )
                )
                index += phrase_length
                matched = True
                break

        if not matched:
            index += 1

    return matches


def previous_content_token(doc: Doc, index: int) -> Token | None:
    for token in reversed(doc[:index]):
        if not token.is_space:
            return token
    return None


def next_content_token(doc: Doc, index: int) -> Token | None:
    for token in doc[index + 1 :]:
        if not token.is_space:
            return token
    return None


def has_temporal_hint(span: Span) -> bool:
    if any(entity.label_ in {"DATE", "TIME"} for entity in span.ents):
        return True

    for token in span:
        if token.like_num:
            return True
        if token.lower_ in TEMPORAL_HINTS:
            return True

    return False


def has_past_event_hint(span: Span) -> bool:
    for token in span:
        if token.pos_ == "VERB" and token.tag_ == "VBD":
            return True
    return False


def has_aspectual_auxiliary_hint(span: Span) -> bool:
    for token in span:
        if token.dep_ == "aux" and token.lemma_ in {"have", "be"}:
            return True
    return False


def resolve_marker_category(span: Span, default_category: str) -> str:
    marker_text = span.text.lower()
    doc = span.doc
    prev_token = previous_content_token(doc, span.start)
    next_token = next_content_token(doc, span.end - 1)
    left_context = doc[max(0, span.start - 8) : span.start]
    right_context = doc[span.end : min(len(doc), span.end + 6)]

    if marker_text == "since":
        if span.root.pos_ == "ADP" or has_temporal_hint(right_context):
            return "TEMPORAL"
        if has_aspectual_auxiliary_hint(left_context) and has_past_event_hint(right_context):
            return "TEMPORAL"
        if has_past_event_hint(right_context):
            return "TEMPORAL"
        return "CONTINGENCY"

    if marker_text == "while":
        if prev_token is not None and prev_token.text in {",", ";", "-", "--"}:
            return "COMPARISON"
        return "TEMPORAL"

    if marker_text == "as":
        if has_temporal_hint(right_context):
            return "TEMPORAL"
        return "CONTINGENCY"

    if marker_text in {"and", "or"}:
        if span.root.pos_ != "CCONJ":
            return default_category
        if prev_token is not None and next_token is not None:
            if prev_token.pos_ == next_token.pos_ == "ADJ":
                return "EXPANSION"
        return "EXPANSION"

    if marker_text in {"however", "whereas", "although", "though", "but"}:
        return "COMPARISON"

    return default_category


def render_marker_highlight_text(doc: Doc, matches: list[MarkerMatch]) -> str:
    pieces: list[str] = []
    matches_by_start = {match.start_token: match for match in matches}
    index = 0

    while index < len(doc):
        match = matches_by_start.get(index)
        if match is not None:
            span = doc[match.start_token : match.end_token]
            category_class = match.category.lower()
            pieces.append(
                f"<span class='marker-highlight marker-{category_class}'><strong>{html.escape(span.text)}</strong> "
                f"<span class='marker-tag'>[{html.escape(match.category)}]</span></span>"
            )
            pieces.append(html.escape(span[-1].whitespace_))
            index = match.end_token
            continue

        token = doc[index]
        pieces.append(html.escape(token.text))
        pieces.append(html.escape(token.whitespace_))
        index += 1

    return "".join(pieces).strip()


def trim_argument_tokens(tokens: list[Token], strip_from_start: bool) -> list[Token]:
    trimmed_tokens = tokens[:]

    while trimmed_tokens and trimmed_tokens[0 if strip_from_start else -1].text in ARG_BOUNDARY_PUNCTUATION:
        if strip_from_start:
            trimmed_tokens = trimmed_tokens[1:]
        else:
            trimmed_tokens = trimmed_tokens[:-1]

    return trimmed_tokens


def extract_argument_pair(doc: Doc, match: MarkerMatch) -> tuple[str, str]:
    arg1_tokens = trim_argument_tokens(list(doc[: match.start_token]), strip_from_start=False)
    arg2_tokens = trim_argument_tokens(list(doc[match.end_token :]), strip_from_start=True)

    arg1_text = "".join(token.text_with_ws for token in arg1_tokens).strip()
    arg2_text = "".join(token.text_with_ws for token in arg2_tokens).strip()
    return arg1_text, arg2_text


def render_argument_box(title: str, body: str, css_class: str) -> None:
    safe_body = html.escape(body) if body else "未抽取到对应论据。"
    st.markdown(
        f"""
        <div class="argument-card {css_class}">
            <div class="argument-card-title">{html.escape(title)}</div>
            <div class="argument-card-body">{safe_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dedupe_mentions_in_order(mentions: list[CoreferenceMention]) -> list[CoreferenceMention]:
    seen: set[tuple[int, int, str]] = set()
    deduped: list[CoreferenceMention] = []

    for mention in mentions:
        key = (mention.start_char, mention.end_char, mention.text)
        if key not in seen:
            seen.add(key)
            deduped.append(mention)

    return deduped


def find_non_overlapping_span(
    text: str,
    mention_text: str,
    used_spans: list[tuple[int, int]],
    start_hint: int = 0,
) -> tuple[int, int] | None:
    if not mention_text.strip():
        return None

    candidates = list(re.finditer(re.escape(mention_text), text))
    if not candidates:
        candidates = list(re.finditer(re.escape(mention_text), text, flags=re.IGNORECASE))

    sorted_candidates = sorted(
        candidates,
        key=lambda match: (
            match.start() < start_hint,
            abs(match.start() - start_hint),
            match.start(),
        ),
    )

    for candidate in sorted_candidates:
        start, end = candidate.start(), candidate.end()
        overlaps = any(not (end <= used_start or start >= used_end) for used_start, used_end in used_spans)
        if not overlaps:
            return start, end

    return None


def build_clusters_from_strings(text: str, raw_clusters: list[list[str]]) -> list[CoreferenceCluster]:
    used_spans: list[tuple[int, int]] = []
    clusters: list[CoreferenceCluster] = []

    for cluster_index, cluster_mentions in enumerate(raw_clusters, start=1):
        mentions: list[CoreferenceMention] = []
        search_hint = 0

        for mention_text in cluster_mentions:
            span = find_non_overlapping_span(text, mention_text, used_spans, start_hint=search_hint)
            if span is None:
                continue

            start_char, end_char = span
            mention = CoreferenceMention(
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
            mentions.append(mention)
            used_spans.append(span)
            search_hint = end_char

        mentions = dedupe_mentions_in_order(sorted(mentions, key=lambda item: item.start_char))
        if mentions:
            clusters.append(
                CoreferenceCluster(
                    cluster_id=cluster_index,
                    color=COREFERENCE_CLUSTER_COLORS[(cluster_index - 1) % len(COREFERENCE_CLUSTER_COLORS)],
                    mentions=mentions,
                )
            )

    return clusters


def extract_coreference_clusters(text: str) -> list[CoreferenceCluster]:
    model = load_fastcoref_model()
    prediction = model.predict(texts=[text])[0]
    raw_clusters = prediction.get_clusters(as_strings=True)
    if not raw_clusters:
        return []
    return build_clusters_from_strings(text, raw_clusters)


def render_coreference_html(text: str, clusters: list[CoreferenceCluster]) -> str:
    mentions = []
    for cluster in clusters:
        for mention in cluster.mentions:
            mentions.append((mention.start_char, mention.end_char, cluster.cluster_id, cluster.color))

    mentions.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    pieces: list[str] = []
    cursor = 0

    for start_char, end_char, cluster_id, color in mentions:
        if start_char < cursor:
            continue

        pieces.append(html.escape(text[cursor:start_char]))
        pieces.append(
            f"<span class='coref-mention' style='background:{color};'>"
            f"{html.escape(text[start_char:end_char])}"
            f"<span class='coref-label'>C{cluster_id}</span>"
            f"</span>"
        )
        cursor = end_char

    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def render_cluster_list(clusters: list[CoreferenceCluster]) -> None:
    st.markdown("### 指代簇列表")

    for cluster in clusters:
        mention_texts = [mention.text for mention in cluster.mentions]
        list_html = ", ".join(f"'{html.escape(text)}'" for text in mention_texts)
        st.markdown(
            f"""
            <div class="cluster-list-card">
                <div class="cluster-list-title">
                    <span class="cluster-color-dot" style="background:{cluster.color};"></span>
                    Cluster {cluster.cluster_id}
                </div>
                <div class="cluster-list-body">[{list_html}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .source-note {
                padding: 0.85rem 1rem;
                border: 1px solid #d0d7de;
                border-radius: 0.75rem;
                background: #f8fafc;
                margin-bottom: 1rem;
            }
            .edu-card {
                border: 1px solid #d0d7de;
                border-radius: 0.85rem;
                padding: 0.9rem 1rem;
                margin-bottom: 0.85rem;
                background: #ffffff;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            .edu-card-title {
                font-weight: 600;
                margin-bottom: 0.55rem;
                color: #1f2937;
                font-size: 0.95rem;
            }
            .edu-card-body {
                white-space: pre-wrap;
                line-height: 1.7;
                color: #111827;
            }
            .boundary-token {
                background: #fde68a;
                border-radius: 0.35rem;
                padding: 0.05rem 0.2rem;
                font-weight: 600;
            }
            .panel-note {
                color: #475569;
                margin-bottom: 0.8rem;
            }
            .marker-preview {
                border: 1px solid #d0d7de;
                border-radius: 0.85rem;
                padding: 0.95rem 1rem;
                background: #ffffff;
                margin-bottom: 1rem;
                line-height: 1.8;
            }
            .marker-highlight {
                display: inline-block;
                border-radius: 0.45rem;
                padding: 0.05rem 0.3rem;
                font-weight: 600;
            }
            .marker-tag {
                font-size: 0.82rem;
                margin-left: 0.1rem;
            }
            .marker-temporal {
                background: #dbeafe;
                color: #1d4ed8;
            }
            .marker-contingency {
                background: #dcfce7;
                color: #15803d;
            }
            .marker-comparison {
                background: #fce7f3;
                color: #be185d;
            }
            .marker-expansion {
                background: #fef3c7;
                color: #b45309;
            }
            .marker-list-item {
                border: 1px solid #d0d7de;
                border-radius: 0.8rem;
                padding: 0.75rem 0.9rem;
                background: #ffffff;
                margin-bottom: 0.75rem;
            }
            .marker-list-token {
                font-weight: 700;
            }
            .argument-card {
                border-radius: 0.9rem;
                padding: 0.95rem 1rem;
                margin-bottom: 0.85rem;
                border: 1px solid transparent;
            }
            .argument-card.arg1-card {
                background: #eff6ff;
                border-color: #bfdbfe;
            }
            .argument-card.arg2-card {
                background: #ecfdf5;
                border-color: #bbf7d0;
            }
            .argument-card-title {
                font-weight: 700;
                margin-bottom: 0.55rem;
                color: #0f172a;
            }
            .argument-card-body {
                white-space: pre-wrap;
                line-height: 1.7;
                color: #111827;
            }
            .coref-preview {
                border: 1px solid #d0d7de;
                border-radius: 0.9rem;
                padding: 1rem;
                background: #ffffff;
                line-height: 1.9;
                margin-bottom: 1rem;
            }
            .coref-mention {
                border-radius: 0.35rem;
                padding: 0.05rem 0.2rem;
                font-weight: 600;
            }
            .coref-label {
                display: inline-block;
                margin-left: 0.25rem;
                padding: 0 0.3rem;
                border-radius: 999px;
                background: rgba(15, 23, 42, 0.1);
                font-size: 0.75rem;
                vertical-align: middle;
            }
            .cluster-list-card {
                border: 1px solid #d0d7de;
                border-radius: 0.9rem;
                padding: 0.9rem 1rem;
                margin-bottom: 0.85rem;
                background: #ffffff;
            }
            .cluster-list-title {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.45rem;
            }
            .cluster-color-dot {
                width: 0.85rem;
                height: 0.85rem;
                border-radius: 999px;
                display: inline-block;
                border: 1px solid rgba(15, 23, 42, 0.12);
            }
            .cluster-list-body {
                white-space: pre-wrap;
                line-height: 1.7;
                color: #334155;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_edu_comparison_tab() -> None:
    st.subheader("第一个标签页：EDU 切分对比页")
    st.markdown(
        """
        <div class="source-note">
            使用 <code>requests</code> 获取 NeuralEDUSeg 的 DEV 样本，
            左栏先将基线规则转成 token 级序列标注 <code>y_t in {0,1}</code>，
            再由标签序列恢复 EDU 切分结果，并在 EDU 卡片中高亮 <code>y_t = 1</code> 的起始词，
            右栏展示数据集中的 Ground Truth EDU 标注。
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        sample_data = load_sample_data()
    except RequestException as error:
        st.error(f"样本抓取失败：{error}")
        st.stop()

    try:
        nlp = load_spacy_model()
    except OSError:
        st.error("未检测到 spaCy 模型 `en_core_web_sm`。请先安装该模型后再运行应用。")
        st.code("python -m spacy download en_core_web_sm")
        st.stop()

    doc, baseline_edus, start_labels = build_baseline_segments(
        sample_data["comparison_text"],
        nlp,
    )
    ground_truth_edus = sample_data["ground_truth_edus"]
    predicted_start_count = sum(start_labels.values())

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Baseline EDU 数", len(baseline_edus))
    metric_col2.metric("Ground Truth EDU 数", len(ground_truth_edus))
    metric_col3.metric("预测 y_t=1 数", predicted_start_count)

    st.markdown(
        f"""
        **样本来源**：[`{sample_data["sample_name"]}`]({sample_data["source_urls"]["raw"]})，
        对应真实标注 [`{sample_data["sample_name"]}.edus`]({sample_data["source_urls"]["edus"]})。
        """
    )

    with st.expander("查看原始样本文本", expanded=False):
        st.text(sample_data["raw_text"])

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown("### 规则基线切分结果")
        st.markdown(
            "<div class='panel-note'>基线规则先预测 token 级起始标签，再恢复 EDU。卡片中高亮词表示预测为 `y_t=1` 的 EDU 起始词。</div>",
            unsafe_allow_html=True,
        )
        render_baseline_cards(baseline_edus, doc, start_labels)

    with right_column:
        st.markdown("### NeuralEDUSeg 数据集真实标注结果")
        st.markdown(
            "<div class='panel-note'>每个卡片对应 `.edus` 文件中的一行 Ground Truth EDU。从序列标注角度看，每一条 EDU 的首词都可视为一个 gold 起始标签 `1`。</div>",
            unsafe_allow_html=True,
        )
        render_ground_truth_cards(ground_truth_edus)


def render_discourse_marker_tab() -> None:
    st.subheader("第二个标签页：浅层篇章分析（显式连接词）")
    st.markdown(
        """
        <div class="source-note">
            输入一个英文句子后，系统会基于显式连接词规则进行浅层篇章分析，
            识别连接词及其 PDTB 顶级语义类别，并给出简易的 <code>Arg1</code> / <code>Arg2</code> 切分结果。
            当前版本已加入按钮触发分析和轻量上下文消歧规则。
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "module2_last_sentence" not in st.session_state:
        st.session_state["module2_last_sentence"] = DEFAULT_DISCOURSE_SENTENCE
    if "module2_input_sentence" not in st.session_state:
        st.session_state["module2_input_sentence"] = DEFAULT_DISCOURSE_SENTENCE

    with st.form("module2_analysis_form", clear_on_submit=False):
        sentence = st.text_area(
            "请输入待分析句子",
            key="module2_input_sentence",
            height=120,
        )
        submitted = st.form_submit_button("开始分析")

    if submitted:
        st.session_state["module2_last_sentence"] = sentence.strip()

    analyzed_sentence = st.session_state["module2_last_sentence"].strip()

    if not analyzed_sentence:
        st.warning("请输入一个英文句子后再进行显式连接词分析。")
        return

    try:
        nlp = load_spacy_model()
    except OSError:
        st.error("未检测到 spaCy 模型 `en_core_web_sm`。请先安装该模型后再运行应用。")
        st.code("python -m spacy download en_core_web_sm")
        st.stop()

    doc = nlp(analyzed_sentence)
    matches = find_explicit_marker_matches(doc)

    st.markdown("### 连接词识别结果")
    if not matches:
        st.info("未在当前句子中匹配到预设的显式连接词。")
        st.markdown(f"<div class='marker-preview'>{html.escape(analyzed_sentence)}</div>", unsafe_allow_html=True)
        return

    st.markdown(
        f"<div class='marker-preview'>{render_marker_highlight_text(doc, matches)}</div>",
        unsafe_allow_html=True,
    )

    for match in matches:
        category_class = match.category.lower()
        st.markdown(
            f"""
            <div class="marker-list-item">
                命中连接词：
                <span class="marker-list-token">{html.escape(match.text)}</span>
                <span class="marker-highlight marker-{category_class}">[{html.escape(match.category)}]</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    selected_match = matches[0]
    arg1_text, arg2_text = extract_argument_pair(doc, selected_match)

    if len(matches) > 1:
        st.info("当前句子命中了多个显式连接词，下面的 Arg1 / Arg2 展示默认基于第一个命中的连接词进行简易切分。")

    st.markdown("### 简易论据提取")
    st.markdown(
        f"<div class='panel-note'>当前使用连接词 <strong>{html.escape(selected_match.text)}</strong> 作为切分边界，进行简易的论据划分。</div>",
        unsafe_allow_html=True,
    )

    arg_col1, arg_col2 = st.columns(2)
    with arg_col1:
        render_argument_box("Arg1（前置论据）", arg1_text, "arg1-card")
    with arg_col2:
        render_argument_box("Arg2（后置论据）", arg2_text, "arg2-card")


def render_coreference_tab() -> None:
    st.subheader("第三个标签页：指代消解（Coreference）")
    st.markdown(
        """
        <div class="source-note">
            输入一段英文文本后，系统会尝试基于 <code>fastcoref</code> 提取指代簇，
            并在原文中用同色高亮同一实体的不同 mention，同时在下方列出对应的 cluster 等价类。
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "module3_last_text" not in st.session_state:
        st.session_state["module3_last_text"] = DEFAULT_COREFERENCE_TEXT
    if "module3_input_text" not in st.session_state:
        st.session_state["module3_input_text"] = DEFAULT_COREFERENCE_TEXT

    with st.form("module3_coref_form", clear_on_submit=False):
        input_text = st.text_area(
            "请输入待做指代消解的英文段落",
            key="module3_input_text",
            height=180,
        )
        submitted = st.form_submit_button("开始分析")

    if submitted:
        st.session_state["module3_last_text"] = input_text.strip()

    analyzed_text = st.session_state["module3_last_text"].strip()
    if not analyzed_text:
        st.warning("请输入英文段落后再进行指代消解分析。")
        return

    try:
        clusters = extract_coreference_clusters(analyzed_text)
    except ImportError:
        st.error("当前环境缺少 `fastcoref`，暂时无法执行模块三分析。")
        st.markdown("需要的依赖：")
        st.code("pip install fastcoref")
        st.info("首次运行 `fastcoref` 时还可能自动下载模型权重，请保持网络可用。")
        return
    except Exception as error:
        st.error(f"指代消解分析失败：{error}")
        st.info("请确认 `fastcoref` 已正确安装，并且模型权重可以正常加载。")
        return

    st.markdown("### 原文高亮结果")
    if not clusters:
        st.info("未检测到明显的指代簇。")
        st.markdown(f"<div class='coref-preview'>{html.escape(analyzed_text)}</div>", unsafe_allow_html=True)
        return

    st.markdown(
        f"<div class='coref-preview'>{render_coreference_html(analyzed_text, clusters)}</div>",
        unsafe_allow_html=True,
    )
    render_cluster_list(clusters)


def render_placeholder_tab(module_name: str) -> None:
    st.subheader(module_name)
    st.info("该模块待你提供详细需求后继续实现。")


def main() -> None:
    st.set_page_config(page_title="篇章分析与指代消解系统", layout="wide")
    st.title("篇章分析与指代消解系统")
    render_styles()

    tab1, tab2, tab3 = st.tabs(
        ["EDU 切分对比页", "浅层篇章分析（显式连接词）", "指代消解（Coreference）"]
    )

    with tab1:
        render_edu_comparison_tab()

    with tab2:
        render_discourse_marker_tab()

    with tab3:
        render_coreference_tab()


if __name__ == "__main__":
    main()
