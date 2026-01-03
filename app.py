import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO


DATA_DIR = Path(__file__).parent


def read_table(path: Path, header=None):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=header)
    return pd.read_excel(path, sheet_name=0, header=header)


def detect_report_type(raw_df: pd.DataFrame) -> str:
    """Tìm loại sổ xét nghiệm từ vài dòng đầu."""
    texts = (
        raw_df.iloc[:5, :].astype(str).stack().str.upper().dropna().tolist()
    )
    for txt in texts:
        if "SỔ XÉT NGHIỆM" in txt:
            if "VI SINH" in txt:
                return "VI SINH"
            if "HÓA SINH" in txt:
                return "HÓA SINH"
            if "HUYẾT HỌC" in txt:
                return "HUYẾT HỌC"
            if "NƯỚC TIỂU" in txt:
                return "NƯỚC TIỂU"
    return "UNKNOWN"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa MultiIndex về chuỗi, tránh NaN."""
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(
        tuple((str(a).strip() if pd.notna(a) else "") for a in col)
        for col in df.columns
    )
    return df


def pick_col(df: pd.DataFrame, level0: str, level1: str | None = None):
    cols = [
        col
        for col in df.columns
        if col[0] == level0 and (level1 is None or col[1] == level1)
    ]
    return cols[0] if cols else None


def load_one_file(path: Path) -> pd.DataFrame:
    """Đọc 1 file và trả về bảng từng dòng đã chuẩn hóa."""
    raw = read_table(path, header=None)
    report_type = detect_report_type(raw)

    df = read_table(path, header=[4, 5])
    df = normalize_columns(df)

    stt_col = pick_col(df, "STT")
    if stt_col is None:
        return pd.DataFrame()

    data = df[df[stt_col].notna()].copy()

    # Chỉ giữ dòng có Người thực hiện không trống
    performer_col = pick_col(df, "Người thực hiện")
    if performer_col:
        performer_series = data[performer_col].astype(str)
        performer_series = performer_series.replace({"nan": ""})
        performer_series = performer_series.replace(r"^\s*$", "", regex=True)
        data = data.loc[performer_series != ""].copy()

    noi_gui_col = pick_col(df, "Khoa/Phòng chỉ định")
    time_recv_col = pick_col(df, "Thời gian nhận mẫu (2)")
    time_valid_col = pick_col(df, "Thời gian valid (3)")
    bh_col = pick_col(df, "Đối tượng", "BH")
    vp_col = pick_col(df, "Đối tượng", "VP")

    recv_dt = (
        pd.to_datetime(data[time_recv_col], errors="coerce")
        if time_recv_col
        else pd.Series(pd.NaT, index=data.index)
    )
    valid_dt = (
        pd.to_datetime(data[time_valid_col], errors="coerce")
        if time_valid_col
        else pd.Series(pd.NaT, index=data.index)
    )
    tat_minutes = (valid_dt - recv_dt).dt.total_seconds() / 60

    try:
        result_block = data.xs("Kết quả xét nghiệm", level=0, axis=1)
    except KeyError:
        result_block = pd.DataFrame(index=data.index)
    result_counts = (
        result_block.notna().sum(axis=1)
        if not result_block.empty
        else pd.Series(0, index=data.index)
    )

    # Chuẩn hóa Nơi gửi: nếu có "phòng khám" -> "Khoa khám bệnh"
    noi_gui_series = data[noi_gui_col] if noi_gui_col else pd.Series("", index=data.index)
    noi_gui_series = noi_gui_series.astype(str)
    mask_pk = noi_gui_series.str.contains("phòng khám", case=False, na=False)
    mask_pl = noi_gui_series.str.contains(
        "phòng lưu khoa khám bệnh", case=False, na=False
    )
    noi_gui_series = noi_gui_series.mask(mask_pk | mask_pl, "Khoa khám bệnh")
    noi_gui_series = noi_gui_series.replace({"nan": ""})
    noi_gui_series = noi_gui_series.replace(r"^\s*$", "Không rõ", regex=True)

    # Đếm số xét nghiệm trên dòng
    total_tests = (
        result_counts if report_type in ["VI SINH", "HÓA SINH"] else 0
    ) + (
        1 if report_type == "HUYẾT HỌC" else 0
    ) + (
        1 if report_type == "NƯỚC TIỂU" else 0
    )

    # BH/VP: tổng số xét nghiệm của dòng nếu cột tương ứng có giá trị, ngược lại 0
    bh_flag = data[bh_col].notna() if bh_col else pd.Series(False, index=data.index)
    vp_flag = data[vp_col].notna() if vp_col else pd.Series(False, index=data.index)
    bh_count = total_tests * bh_flag
    vp_count = total_tests * vp_flag

    # TAT theo từng loại (NaN nếu không thuộc loại đó để mean bỏ qua)
    def tat_for(kind: str):
        return tat_minutes if report_type == kind else pd.Series(
            np.nan, index=data.index, dtype="float"
        )

    tat_vs = tat_for("VI SINH")
    tat_sh = tat_for("HÓA SINH")
    tat_hh = tat_for("HUYẾT HỌC")
    tat_nt = tat_for("NƯỚC TIỂU")

    rows = pd.DataFrame(
        {
            "Nơi gửi": noi_gui_series,
            "Tháng/năm": valid_dt.dt.strftime("%m/%Y"),
            "Đối tượng bảo hiểm": bh_count,
            "Đối tượng thu phí": vp_count,
            "XN vi sinh": result_counts if report_type == "VI SINH" else 0,
            "XN sinh hóa": result_counts if report_type == "HÓA SINH" else 0,
            "XN huyết học": 1 if report_type == "HUYẾT HỌC" else 0,
            "XN nước tiểu": 1 if report_type == "NƯỚC TIỂU" else 0,
            "Thời gian trung bình (phút)": tat_minutes,
            "TAT vi sinh (phút)": tat_vs,
            "TAT sinh hóa (phút)": tat_sh,
            "TAT huyết học (phút)": tat_hh,
            "TAT nước tiểu (phút)": tat_nt,
        }
    )

    for col in ["XN vi sinh", "XN sinh hóa", "XN huyết học", "XN nước tiểu"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0)

    rows["Thời gian trung bình (phút)"] = pd.to_numeric(
        rows["Thời gian trung bình (phút)"], errors="coerce"
    )

    return rows


def aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Gom 1 dòng cho mỗi (Tháng/năm, Nơi gửi)
    group_cols = ["Nơi gửi", "Tháng/năm"]
    agg = df.groupby(group_cols, dropna=False).agg(
        {
            "XN vi sinh": "sum",
            "XN sinh hóa": "sum",
            "XN huyết học": "sum",
            "XN nước tiểu": "sum",
            "Đối tượng bảo hiểm": "sum",
            "Đối tượng thu phí": "sum",
            "Thời gian trung bình (phút)": "mean",
            "TAT vi sinh (phút)": "mean",
            "TAT sinh hóa (phút)": "mean",
            "TAT huyết học (phút)": "mean",
            "TAT nước tiểu (phút)": "mean",
        }
    )
    agg = agg.reset_index()
    for col in [
        "Thời gian trung bình (phút)",
        "TAT vi sinh (phút)",
        "TAT sinh hóa (phút)",
        "TAT huyết học (phút)",
        "TAT nước tiểu (phút)",
    ]:
        if col in agg.columns:
            agg[col] = agg[col].round(2)

    # Sắp xếp cột: BH/TP ngay sau Tháng/năm
    desired_order = [
        "Nơi gửi",
        "Tháng/năm",
        "Đối tượng bảo hiểm",
        "Đối tượng thu phí",
        "XN vi sinh",
        "XN sinh hóa",
        "XN huyết học",
        "XN nước tiểu",
        "TAT vi sinh (phút)",
        "TAT sinh hóa (phút)",
        "TAT huyết học (phút)",
        "TAT nước tiểu (phút)",
        "Thời gian trung bình (phút)",
    ]
    existing = [c for c in desired_order if c in agg.columns]
    remaining = [c for c in agg.columns if c not in existing]
    agg = agg[existing + remaining]
    return agg


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="TongHop")
    return buffer.getvalue()


def main():
    st.title("Báo cáo tổng hợp kết quả xét nghiệm")
    st.caption("Đọc file và tổng hợp theo Tháng/năm - Nơi gửi.")

    selected = []
    run = False
    with st.sidebar:
        st.header("Cài đặt")
        st.subheader("Tải tệp dữ liệu")
        st.markdown(
            """
            <style>
            /* Đổi nhãn kéo thả sang tiếng Việt */
            [data-testid="stFileUploadDropzone"] div div:has(span[data-testid="stFileUploadDropzoneLabel"]) span[data-testid="stFileUploadDropzoneLabel"] {
                visibility: hidden;
                position: relative;
            }
            [data-testid="stFileUploadDropzone"] div div:has(span[data-testid="stFileUploadDropzoneLabel"]) span[data-testid="stFileUploadDropzoneLabel"]::after {
                content: "Kéo và thả tệp tại đây";
                visibility: visible;
                position: absolute;
                left: 0;
                top: 0;
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        uploads = st.file_uploader(
            "Chọn hoặc kéo thả (xls, xlsx, csv)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
        )
        uploaded_names = []
        if uploads:
            for f in uploads:
                dest = DATA_DIR / f.name
                dest.write_bytes(f.read())
                uploaded_names.append(f.name)
            st.success(
                f"Đã lưu {len(uploaded_names)} file vào {DATA_DIR}",
                icon="✅",
            )
        st.divider()

        files = sorted(
            [p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".xlsx", ".xls", ".csv"}]
        )
        file_names = [f.name for f in files]

        st.subheader("Xóa tệp")
        to_delete = st.multiselect("Chọn tệp để xóa", file_names, key="delete_files")
        if st.button("Xóa tệp đã chọn", use_container_width=True):
            removed = []
            for name in to_delete:
                path = DATA_DIR / name
                try:
                    path.unlink(missing_ok=True)
                    removed.append(name)
                except Exception as e:
                    st.error(f"Không xóa được {name}: {e}")
            if removed:
                st.success(f"Đã xóa: {', '.join(removed)}", icon="🗑️")
            st.stop()

        mode = st.radio(
            "Chế độ xử lý",
            ["Tất cả các file", "Chọn file cụ thể"],
            index=0,
        )

        selected = files
        if mode == "Chọn file cụ thể":
            chosen = st.multiselect("Chọn file", file_names, default=file_names[:1])
            name_to_path = {f.name: f for f in files}
            selected = [name_to_path[n] for n in chosen if n in name_to_path]

        st.markdown(f"**Đã chọn {len(selected)} file**")
        st.divider()
        st.caption("Bấm nút bên dưới để xử lý")
        run = st.button("Xử lý dữ liệu", use_container_width=True)

    if not selected:
        st.warning("Chưa có file nào được chọn.")
        return

    if run:
        st.info(
            f"Đang xử lý {len(selected)} file: "
            + ", ".join([p.name for p in selected]),
            icon="⏳",
        )
        all_rows = []
        for path in selected:
            # Bảo đảm đường dẫn tuyệt đối và tồn tại
            path = path if path.is_absolute() else (DATA_DIR / path).resolve()
            if not path.exists():
                st.error(f"Không tìm thấy file: {path}")
                continue
            rows = load_one_file(path)
            if rows.empty:
                st.warning(f"Không đọc được dữ liệu từ {path.name}")
            else:
                all_rows.append(rows)

        if not all_rows:
            st.error("Không có dữ liệu để tổng hợp.")
            return

        merged = pd.concat(all_rows, ignore_index=True)
        summary = aggregate_rows(merged)

        st.subheader("Bảng tổng hợp")
        st.dataframe(summary, use_container_width=True)

        xls_bytes = to_excel_bytes(summary)
        st.download_button(
            "Tải Excel tổng hợp",
            data=xls_bytes,
            file_name="tong_hop_xet_nghiem.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()

