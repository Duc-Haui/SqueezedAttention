# Phân tích Định tính Đa mốc (Qualitative Examples across Sparsity Levels)

Tài liệu này trích xuất các mẫu câu sinh ra từ mô hình, thể hiện sự suy thoái dần của thuật toán **SQ Gốc** khi mức độ nén tăng lên (70% -> 90%), đồng thời chứng minh sự tráng kiện của thuật toán **VA-Squeezed** trên cùng các mức độ nén đó.

**Chú giải:**
- <span style="color: green; font-weight: bold;">Văn bản màu xanh</span>: Token trùng khớp chính xác với Đáp án (Ground Truth Match).
- <span style="color: red; font-weight: bold;">Văn bản màu đỏ</span>: Token vô nghĩa bị lặp lại nhiều lần (Hallucination Loop).

---

## Ví dụ 1 (Mẫu dữ liệu số #92)

**Đáp án chuẩn (Ground Truth):**
> Yes

| Mô hình | Mức Nén | Thống kê (Tokens) | Văn bản sinh ra (Generated Text) |
| :--- | :---: | :--- | :--- |
| **Baseline** | **0%** | Sinh ra: **71**<br>Ảo giác: **65**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> |
| **SQ Gốc** | **70%** | Sinh ra: **71**<br>Ảo giác: **<span style='color:red'>65</span>**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> |
| **VA-Squeezed** | **70%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **1** | <span style="color: green; font-weight: bold;">yes</span> |
| **SQ Gốc** | **80%** | Sinh ra: **71**<br>Ảo giác: **<span style='color:red'>65</span>**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> |
| **VA-Squeezed** | **80%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **1** | <span style="color: green; font-weight: bold;">yes</span> |
| **SQ Gốc** | **90%** | Sinh ra: **71**<br>Ảo giác: **<span style='color:red'>65</span>**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> a <span style="color: red; font-weight: bold;">pre</span>-<span style="color: red; font-weight: bold;">trained</span> <span style="color: red; font-weight: bold;">language</span> <span style="color: red; font-weight: bold;">model</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">yes</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">use</span> |
| **VA-Squeezed** | **90%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **1** | <span style="color: green; font-weight: bold;">yes</span> |

---

## Ví dụ 2 (Mẫu dữ liệu số #11)

**Đáp án chuẩn (Ground Truth):**
> English

| Mô hình | Mức Nén | Thống kê (Tokens) | Văn bản sinh ra (Generated Text) |
| :--- | :---: | :--- | :--- |
| **Baseline** | **0%** | Sinh ra: **94**<br>Ảo giác: **60**<br>Khớp: **1** | <span style="color: green; font-weight: bold;">English</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: The <span style="color: red; font-weight: bold;">paper</span> <span style="color: red; font-weight: bold;">proposes</span> a <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">synthetic</span> <span style="color: red; font-weight: bold;">data</span> to <span style="color: red; font-weight: bold;">improve</span> the <span style="color: red; font-weight: bold;">performance</span> of <span style="color: red; font-weight: bold;">neural</span> <span style="color: red; font-weight: bold;">machine</span> <span style="color: red; font-weight: bold;">translation</span> <span style="color: red; font-weight: bold;">models</span> for <span style="color: red; font-weight: bold;">text</span> <span style="color: red; font-weight: bold;">simplification</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: The <span style="color: red; font-weight: bold;">paper</span> <span style="color: red; font-weight: bold;">proposes</span> a <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">synthetic</span> <span style="color: red; font-weight: bold;">data</span> to <span style="color: red; font-weight: bold;">improve</span> the <span style="color: red; font-weight: bold;">performance</span> of <span style="color: red; font-weight: bold;">neural</span> <span style="color: red; font-weight: bold;">machine</span> <span style="color: red; font-weight: bold;">translation</span> <span style="color: red; font-weight: bold;">models</span> for <span style="color: red; font-weight: bold;">text</span> <span style="color: red; font-weight: bold;">simplification</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: The <span style="color: red; font-weight: bold;">paper</span> <span style="color: red; font-weight: bold;">proposes</span> a <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">synthetic</span> <span style="color: red; font-weight: bold;">data</span> to <span style="color: red; font-weight: bold;">improve</span> the <span style="color: red; font-weight: bold;">performance</span> of <span style="color: red; font-weight: bold;">neural</span> <span style="color: red; font-weight: bold;">machine</span> <span style="color: red; font-weight: bold;">translation</span> <span style="color: red; font-weight: bold;">models</span> for <span style="color: red; font-weight: bold;">text</span> <span style="color: red; font-weight: bold;">simplification</span>.
 |
| **SQ Gốc** | **70%** | Sinh ra: **94**<br>Ảo giác: **<span style='color:red'>63</span>**<br>Khớp: **2** | <span style="color: green; font-weight: bold;">English</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.
 |
| **VA-Squeezed** | **70%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **0** | unanswerable |
| **SQ Gốc** | **80%** | Sinh ra: **94**<br>Ảo giác: **<span style='color:red'>63</span>**<br>Khớp: **2** | <span style="color: green; font-weight: bold;">English</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.
 |
| **VA-Squeezed** | **80%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **0** | unanswerable |
| **SQ Gốc** | **90%** | Sinh ra: **94**<br>Ảo giác: **<span style='color:red'>63</span>**<br>Khớp: **2** | <span style="color: green; font-weight: bold;">English</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">what</span> is the <span style="color: red; font-weight: bold;">main</span> <span style="color: red; font-weight: bold;">idea</span> of this <span style="color: red; font-weight: bold;">paper</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">We</span> <span style="color: red; font-weight: bold;">propose</span> a <span style="color: red; font-weight: bold;">simple</span> <span style="color: red; font-weight: bold;">method</span> to <span style="color: red; font-weight: bold;">use</span> <span style="color: red; font-weight: bold;">simplified</span> <span style="color: red; font-weight: bold;">corpora</span> <span style="color: red; font-weight: bold;">during</span> <span style="color: red; font-weight: bold;">training</span> of <span style="color: red; font-weight: bold;">NMT</span> <span style="color: red; font-weight: bold;">systems</span>, with <span style="color: red; font-weight: bold;">no</span> <span style="color: red; font-weight: bold;">changes</span> to the <span style="color: red; font-weight: bold;">network</span> <span style="color: red; font-weight: bold;">architecture</span>.
 |
| **VA-Squeezed** | **90%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **0** | unanswerable |

---

## Ví dụ 3 (Mẫu dữ liệu số #133)

**Đáp án chuẩn (Ground Truth):**
> Yes

| Mô hình | Mức Nén | Thống kê (Tokens) | Văn bản sinh ra (Generated Text) |
| :--- | :---: | :--- | :--- |
| **Baseline** | **0%** | Sinh ra: **69**<br>Ảo giác: **63**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> |
| **SQ Gốc** | **70%** | Sinh ra: **69**<br>Ảo giác: **<span style='color:red'>63</span>**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> |
| **VA-Squeezed** | **70%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **1** | <span style="color: green; font-weight: bold;">unanswerable</span> |
| **SQ Gốc** | **80%** | Sinh ra: **69**<br>Ảo giác: **<span style='color:red'>63</span>**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> |
| **VA-Squeezed** | **80%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **1** | <span style="color: green; font-weight: bold;">unanswerable</span> |
| **SQ Gốc** | **90%** | Sinh ra: **69**<br>Ảo giác: **<span style='color:red'>63</span>**<br>Khớp: **1** | <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> <span style="color: red; font-weight: bold;">they</span> <span style="color: red; font-weight: bold;">report</span> <span style="color: red; font-weight: bold;">results</span> <span style="color: red; font-weight: bold;">only</span> on <span style="color: red; font-weight: bold;">English</span> <span style="color: red; font-weight: bold;">data</span>?

<span style="color: red; font-weight: bold;">Answer</span>: <span style="color: red; font-weight: bold;">unanswerable</span>

<span style="color: red; font-weight: bold;">Question</span>: <span style="color: red; font-weight: bold;">Do</span> |
| **VA-Squeezed** | **90%** | Sinh ra: **1**<br>Ảo giác: **<span style='color:black'>0</span>**<br>Khớp: **1** | <span style="color: green; font-weight: bold;">unanswerable</span> |

---

