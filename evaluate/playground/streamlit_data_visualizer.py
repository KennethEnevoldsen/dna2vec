import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧬 Alignment Analysis Viewer (Top-75 Fragments)")

uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_data(uploaded_file)
    total_rows = len(df)

    # Boolean Filters with All/True/False
    st.sidebar.header("🔍 Filters (Booleans)")
    filters = {}
    for col in [
        "is_read_gt_index_exist_in_top75",
        "is_read/frag_gt_top_match_in_top75",
        "is_read/frag_gt_sw_score_best_in_top75",
        "is_read/frag_gt_index_same_as_gt_index"
    ]:
        choice = st.sidebar.selectbox(
            f"{col}",
            options=["All", "True", "False"],
            index=0
        )
        filters[col] = choice

    # Apply filters
    for key, val in filters.items():
        if val != "All":
            df = df[df[key] == (val == "True")]
    total_rows = len(df)

    if "row_index" not in st.session_state or st.session_state.row_index >= total_rows:
        st.session_state.row_index = 0

    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous") and st.session_state.row_index > 0:
            st.session_state.row_index -= 1
    with col3:
        if st.button("Next ➡️") and st.session_state.row_index < total_rows - 1:
            st.session_state.row_index += 1

    if total_rows == 0:
        st.warning("No rows match the selected filters.")
    else:
        row_data = df.iloc[st.session_state.row_index]
        st.markdown(f"### 🔎 Viewing Row {st.session_state.row_index + 1} of {total_rows}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📌 Read & Fragment Info")
            for key in [
                "read_gt", "read_gt_idx", "read_gt_cigar_str",
                "read/frag_gt_reference_interval", "read/frag_gt_idx",
                "read_gt_idx_diff_from_read/frag_gt_index"
            ]:
                st.markdown(f"**{key}:**")
                st.code(str(row_data.get(key, "")), language="text")

            st.subheader("✅ Index & Match Checks")
            for key in [
                "is_read_gt_index_exist_in_top75",
                "read/frag_gt_top75_index",
                "is_read/frag_gt_top_match_in_top75",
                "is_read/frag_gt_index_same_as_gt_index"
            ]:
                st.markdown(f"**{key}:**")
                st.code(str(row_data.get(key, "")), language="text")

        with col_right:
            st.subheader("📊 Smith-Waterman Scores")
            for key in [
                "read/frag_gt",
                "best_sw_score_in_top75",
                "read/frag_gt_best_sw_score",
                "is_read/frag_gt_sw_score_best_in_top75"
            ]:
                st.markdown(f"**{key}:**")
                st.code(str(row_data.get(key, "")), language="text")

            st.subheader("🧬 Alignment Viewer")
            options = {
                "Best Alignment": ("alignment_str", "alignment_cigar_str"),
                "2nd Best Alignment": ("second_best_alignment_str", "second_best_alignment_cigar_str"),
                "3rd Best Alignment": ("third_best_alignment_str", "third_best_alignment_cigar_str")
            }
            selected = st.radio("Select alignment to view", list(options.keys()), horizontal=True)
            align_key, cigar_key = options[selected]

            st.markdown("**Alignment CIGAR:**")
            st.code(str(row_data.get(cigar_key, "")))
            st.markdown("**Alignment String:**")
            st.text(row_data.get(align_key, ""))

            st.subheader("📦 Read/Frag Ground Truth Alignment")
            st.text(row_data.get("read/frag_gt_alignment_str", ""))

else:
    st.info("Please upload a CSV file to begin.")
