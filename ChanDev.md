# PaddleOCR PDF to Markdown 변환 개발 참조 문서

> **목적**: PaddleOCR을 활용하여 PDF 문서를 Markdown 파일로 변환하는 프로그램 개발을 위한 종합 가이드

---

## 📑 목차

1. [개요](#개요)
2. [시스템 요구사항](#시스템-요구사항)
3. [설치 가이드](#설치-가이드)
4. [모델 선택 가이드](#모델-선택-가이드)
5. [PP-StructureV3 사용법](#pp-structurev3-사용법)
6. [PaddleOCR-VL 사용법](#paddleocr-vl-사용법)
7. [고급 설정 및 최적화](#고급-설정-및-최적화)
8. [문제 해결](#문제-해결)
9. [실전 예제 코드](#실전-예제-코드)

---

## 개요

### PaddleOCR이란?

PaddleOCR은 바이두(Baidu)가 개발한 오픈소스 OCR 툴킷으로, PDF와 이미지 문서를 구조화된 데이터로 변환합니다.

**GitHub**: https://github.com/PaddlePaddle/PaddleOCR  
**공식 문서**: https://paddlepaddle.github.io/PaddleOCR/  
**Stars**: 62,800+  
**License**: Apache 2.0

### PDF to MD 변환에 적합한 모델

PDF를 Markdown으로 변환하기 위해 **두 가지 주요 옵션**이 있습니다:

1. **PP-StructureV3** (권장)
   - 복잡한 레이아웃, 표, 수식 처리에 강력
   - 고정확도, 다양한 커스터마이징 옵션
   - 더 많은 리소스 필요

2. **PaddleOCR-VL**
   - 경량화된 0.9B 파라미터 VLM 모델
   - 109개 언어 지원
   - 빠른 처리 속도, 적은 리소스

---

## 시스템 요구사항

### 필수 요구사항

```yaml
Python: 3.8 - 3.12
OS: Windows, Linux, macOS
RAM: 최소 4GB (권장 8GB+)
저장공간: 최소 5GB
```

### GPU 사용 시 (선택사항, 성능 향상)

```yaml
CUDA: 9.0 이상 (CUDA 12까지 지원)
cuDNN: 해당 CUDA 버전과 호환
GPU Memory: 최소 4GB (권장 6GB+)
```

### 지원 플랫폼

- **x86_64**: Intel/AMD CPU
- **ARM64**: Apple Silicon (M1/M2/M3)
- **GPU**: NVIDIA CUDA

---

## 설치 가이드

### 1단계: PaddlePaddle 설치

#### CPU 버전 (기본)

```bash
# 최신 안정 버전
pip install paddlepaddle

# 특정 버전 (Python 3.12 호환)
pip install paddlepaddle==3.0.0
```

#### GPU 버전 (CUDA 지원)

```bash
# CUDA 11.8
pip install paddlepaddle-gpu

# CUDA 12.3
pip install paddlepaddle-gpu==3.0.0.post123 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

**설치 확인**:
```python
import paddle
print(paddle.__version__)
print(paddle.device.get_device())  # CPU 또는 GPU
```

### 2단계: PaddleOCR 설치

#### 기본 설치 (텍스트 인식만)

```bash
pip install paddleocr
```

#### 문서 파싱 포함 (권장)

```bash
# PP-StructureV3, PaddleOCR-VL 포함
pip install "paddleocr[doc-parser]"

# 또는 모든 기능 설치
pip install "paddleocr[all]"
```

### 3단계: 추가 의존성

```bash
# 이미지 처리
pip install opencv-python-headless pillow

# PDF 처리
pip install pymupdf  # 또는 PyPDF2

# 유틸리티
pip install tqdm numpy
```

### 설치 검증

```python
from paddleocr import PPStructureV3, PaddleOCRVL

# 모델 자동 다운로드 확인
pipeline = PPStructureV3()
print("PP-StructureV3 설치 완료!")

pipeline_vl = PaddleOCRVL()
print("PaddleOCR-VL 설치 완료!")
```

---

## 모델 선택 가이드

### PP-StructureV3 vs PaddleOCR-VL 비교

| 특성 | PP-StructureV3 | PaddleOCR-VL |
|------|---------------|--------------|
| **정확도** | ⭐⭐⭐⭐⭐ (매우 높음) | ⭐⭐⭐⭐ (높음) |
| **속도** | ⭐⭐⭐ (보통) | ⭐⭐⭐⭐⭐ (빠름) |
| **메모리 사용** | 높음 (2-4GB) | 낮음 (1-2GB) |
| **복잡한 표** | 매우 우수 | 우수 |
| **수식 인식** | 매우 우수 | 우수 |
| **다국어 지원** | 80+ 언어 | 109 언어 |
| **커스터마이징** | 높음 | 중간 |
| **학습/Fine-tuning** | 가능 | 제한적 |

### 사용 케이스별 추천

```python
# 케이스 1: 복잡한 학술 논문, 기술 문서
# 추천: PP-StructureV3
# 이유: 수식, 복잡한 표, 다단 레이아웃 처리 우수

# 케이스 2: 대량의 일반 문서, 빠른 처리 필요
# 추천: PaddleOCR-VL
# 이유: 빠른 속도, 적은 리소스

# 케이스 3: 다국어 혼합 문서
# 추천: PaddleOCR-VL
# 이유: 109개 언어 지원

# 케이스 4: 최고 정확도 필요
# 추천: PP-StructureV3
# 이유: OmniDocBench 벤치마크 1위
```

---

## PP-StructureV3 사용법

### 기본 구조

PP-StructureV3는 다음 단계로 구성됩니다:

```
1. Layout Analysis (레이아웃 분석)
   └─> 문서의 구조 파악 (제목, 본문, 표, 이미지 등)

2. Element Analysis (요소 분석)
   ├─> Text Recognition (PP-OCRv5)
   ├─> Table Recognition
   ├─> Formula Recognition
   ├─> Chart Understanding
   └─> Seal Recognition

3. Data Formatting (데이터 포맷팅)
   └─> Markdown/JSON 변환
```

### 기본 사용 예제

```python
from paddleocr import PPStructureV3

# 1. 파이프라인 초기화
pipeline = PPStructureV3(
    use_gpu=True,  # GPU 사용 여부
    lang='korean',  # 언어 설정
    show_log=True   # 로그 출력
)

# 2. PDF 파일 처리
output = pipeline.predict(
    input='document.pdf'
)

# 3. 결과 저장
for idx, res in enumerate(output):
    # Markdown으로 저장
    res.save_to_markdown(save_path='output')
    
    # JSON으로 저장 (구조화된 데이터)
    res.save_to_json(save_path='output')
    
    print(f"페이지 {idx+1} 처리 완료")
```

### 상세 설정 옵션

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    # === 기본 설정 ===
    use_gpu=True,
    lang='korean',  # 또는 'ch', 'en', 'japan', 'korean' 등
    show_log=True,
    
    # === OCR 설정 ===
    use_common_ocr=True,  # 일반 텍스트 인식 활성화
    ocr_version='PP-OCRv5',  # OCR 버전
    
    # === 고급 기능 ===
    use_doc_orientation_classify=True,  # 문서 방향 자동 보정
    use_doc_unwarping=True,  # 문서 왜곡 보정
    use_textline_orientation=True,  # 텍스트 라인 방향 분류
    
    # === 특수 요소 인식 ===
    use_seal_recognition=True,  # 도장/인장 인식
    use_table_recognition=True,  # 표 인식
    use_formula_recognition=True,  # 수식 인식
    use_chart_parsing=True,  # 차트/그래프 분석
    
    # === 성능 최적화 ===
    layout_batch_size=4,  # 레이아웃 분석 배치 크기
    ocr_batch_size=8,  # OCR 배치 크기
    
    # === 출력 설정 ===
    return_word_box=True,  # 단어별 바운딩 박스 반환
)
```

### PDF 처리 옵션

```python
# 방법 1: PDF 파일 경로 직접 지정
output = pipeline.predict(input='document.pdf')

# 방법 2: 특정 페이지만 처리
output = pipeline.predict(
    input='document.pdf',
    page_range=[0, 5]  # 첫 5페이지만
)

# 방법 3: 이미지로 변환된 PDF 처리
import fitz  # PyMuPDF

doc = fitz.open('document.pdf')
for page_num in range(len(doc)):
    page = doc[page_num]
    pix = page.get_pixmap(dpi=300)  # 고해상도
    img_path = f'page_{page_num}.png'
    pix.save(img_path)
    
    # 페이지별 처리
    output = pipeline.predict(input=img_path)
```

### 결과 데이터 구조

```python
# output은 리스트로, 각 페이지의 결과를 담고 있음
for page_result in output:
    # page_result는 PipelineResult 객체
    
    # 1. 레이아웃 정보
    layout_result = page_result.layout_parsing_result
    for region in layout_result['regions']:
        print(f"영역 타입: {region['type']}")  # text, title, table, figure 등
        print(f"좌표: {region['bbox']}")
        print(f"내용: {region['text']}")
    
    # 2. 시각화 정보
    visual_info = page_result.visual_info
    
    # 3. 원본 이미지
    img = page_result.input_image
    
    # 4. Markdown 텍스트 직접 접근
    markdown_text = page_result.to_markdown()
    print(markdown_text)
```

### 출력 형식 예시

**Markdown 출력**:
```markdown
# 문서 제목

## 1. 서론

본문 내용...

### 1.1 배경

세부 내용...

| 항목 | 값 |
|------|-----|
| A | 100 |
| B | 200 |

$$
E = mc^2
$$

![그림1](path/to/image.png)
```

---

## PaddleOCR-VL 사용법

### 기본 개념

PaddleOCR-VL은 Vision-Language Model로, 단일 모델이 모든 것을 처리합니다:
- 0.9B 파라미터
- NaViT 스타일 동적 해상도 비주얼 인코더
- ERNIE-4.5-0.3B 언어 모델

### 기본 사용 예제

```python
from paddleocr import PaddleOCRVL

# 1. 파이프라인 초기화
pipeline = PaddleOCRVL(
    use_gpu=True,
    show_log=True
)

# 2. PDF/이미지 처리
output = pipeline.predict('document.pdf')

# 3. 결과 저장
for idx, res in enumerate(output):
    # Markdown 저장
    res.save_to_markdown(save_path='output')
    
    # JSON 저장
    res.save_to_json(save_path='output')
    
    print(f"페이지 {idx+1} 처리 완료")
```

### 고급 설정

```python
pipeline = PaddleOCRVL(
    # GPU/CPU 설정
    use_gpu=True,
    gpu_id=0,
    
    # 성능 설정
    batch_size=4,
    max_tokens=2048,  # 최대 토큰 수
    
    # 출력 설정
    return_confidence=True,  # 신뢰도 점수 반환
    show_log=True
)
```

### URL/원격 파일 처리

```python
# URL에서 직접 처리
output = pipeline.predict(
    input="https://example.com/document.pdf"
)

# 또는
output = pipeline.predict(
    input="https://example.com/image.png"
)
```

### 배치 처리

```python
# 여러 파일 한번에 처리
file_list = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']

for file_path in file_list:
    output = pipeline.predict(input=file_path)
    
    for res in output:
        res.save_to_markdown(save_path=f'output/{file_path}')
```

---

## 고급 설정 및 최적화

### 1. 성능 최적화

#### GPU 메모리 최적화

```python
import paddle

# GPU 메모리 사전 할당 비활성화
paddle.set_flags({'FLAGS_fraction_of_gpu_memory_to_use': 0.8})

# 또는 환경 변수로 설정
import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'
```

#### 멀티프로세싱 처리

```python
from concurrent.futures import ProcessPoolExecutor
from paddleocr import PPStructureV3

def process_pdf(pdf_path):
    pipeline = PPStructureV3(use_gpu=False)  # 각 프로세스에서 CPU 사용
    output = pipeline.predict(input=pdf_path)
    
    for idx, res in enumerate(output):
        res.save_to_markdown(save_path=f'output/{pdf_path}')
    
    return f"{pdf_path} 완료"

# 병렬 처리
pdf_files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf', 'doc4.pdf']

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_pdf, pdf_files)
    
    for result in results:
        print(result)
```

### 2. 품질 향상 설정

#### 이미지 전처리

```python
import cv2
import numpy as np

def preprocess_image(img_path):
    """이미지 품질 향상"""
    img = cv2.imread(img_path)
    
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 3. 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 4. 이진화 (선택적)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 저장
    output_path = img_path.replace('.png', '_preprocessed.png')
    cv2.imwrite(output_path, enhanced)
    
    return output_path

# 사용
preprocessed = preprocess_image('page_0.png')
output = pipeline.predict(input=preprocessed)
```

#### PDF 고해상도 변환

```python
import fitz

def pdf_to_high_quality_images(pdf_path, dpi=300):
    """PDF를 고해상도 이미지로 변환"""
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 고해상도 렌더링
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72 DPI가 기본
        pix = page.get_pixmap(matrix=mat)
        
        # 저장
        img_path = f'page_{page_num}_hq.png'
        pix.save(img_path)
        image_paths.append(img_path)
    
    return image_paths

# 사용
images = pdf_to_high_quality_images('document.pdf', dpi=300)

for img_path in images:
    output = pipeline.predict(input=img_path)
    # 처리...
```

### 3. 커스텀 설정

#### 특정 요소만 추출

```python
from paddleocr import PPStructureV3

# 표만 추출
pipeline = PPStructureV3(
    use_table_recognition=True,
    use_formula_recognition=False,
    use_chart_parsing=False
)

# 수식만 추출
pipeline = PPStructureV3(
    use_table_recognition=False,
    use_formula_recognition=True,
    use_chart_parsing=False
)
```

#### 언어별 최적화

```python
# 한국어 최적화
pipeline = PPStructureV3(
    lang='korean',
    ocr_version='PP-OCRv5',
    use_textline_orientation=True
)

# 영어 최적화
pipeline = PPStructureV3(
    lang='en',
    ocr_version='PP-OCRv5'
)

# 다국어 혼합
pipeline = PPStructureV3(
    lang='ch',  # 기본 중국어
    # 필요시 추가 언어 모델 로드
)
```

### 4. 캐싱 및 재사용

```python
# 파이프라인 재사용 (메모리 효율적)
pipeline = PPStructureV3(use_gpu=True)

pdf_files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']

for pdf_file in pdf_files:
    output = pipeline.predict(input=pdf_file)
    # 처리...
    
# pipeline 객체를 반복 사용하여 모델 재로딩 방지
```

---

## 문제 해결

### 일반적인 문제

#### 1. 모델 다운로드 실패

```python
# 문제: 네트워크 오류로 모델 다운로드 실패
# 해결 1: 수동 다운로드
# https://github.com/PaddlePaddle/PaddleOCR 에서 모델 다운로드

# 해결 2: 다운로드 소스 변경
import os
os.environ['PADDLE_PDX_MODEL_SOURCE'] = 'BOS'  # Baidu Object Storage
# 또는
os.environ['PADDLE_PDX_MODEL_SOURCE'] = 'HuggingFace'
```

#### 2. GPU 메모리 부족

```python
# 해결 1: 배치 크기 감소
pipeline = PPStructureV3(
    use_gpu=True,
    layout_batch_size=2,  # 기본 4에서 감소
    ocr_batch_size=4      # 기본 8에서 감소
)

# 해결 2: 메모리 할당 조정
import paddle
paddle.set_flags({'FLAGS_fraction_of_gpu_memory_to_use': 0.5})

# 해결 3: CPU로 전환
pipeline = PPStructureV3(use_gpu=False)
```

#### 3. 한글 폰트 문제 (시각화 시)

```python
# 문제: 한글이 깨져서 표시됨
# 해결: 한글 폰트 경로 지정

from paddleocr import draw_ocr

# Windows
font_path = 'C:/Windows/Fonts/malgun.ttf'

# Linux
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# macOS
font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'

# 사용
annotated = draw_ocr(image, boxes, texts, scores, font_path=font_path)
```

#### 4. PDF 처리 오류

```python
# 문제: 특정 PDF 파일 처리 실패
# 해결: PDF를 이미지로 변환 후 처리

import fitz

def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        
        img_path = f'temp_page_{page_num}.png'
        pix.save(img_path)
        images.append(img_path)
    
    return images

# 이미지로 변환 후 처리
images = convert_pdf_to_images('problematic.pdf')
for img in images:
    output = pipeline.predict(input=img)
```

#### 5. 표 인식 정확도 낮음

```python
# 해결 1: 이미지 해상도 향상
images = pdf_to_high_quality_images('document.pdf', dpi=300)

# 해결 2: 표 인식 모델 최적화
pipeline = PPStructureV3(
    use_table_recognition=True,
    table_model='SLANet_plus'  # 더 강력한 모델
)

# 해결 3: 전처리
preprocessed = preprocess_image(image_path)
output = pipeline.predict(input=preprocessed)
```

### 성능 이슈

```python
# 처리 속도가 느린 경우

# 방법 1: GPU 사용
pipeline = PPStructureV3(use_gpu=True)

# 방법 2: 배치 크기 증가 (GPU 메모리 충분한 경우)
pipeline = PPStructureV3(
    layout_batch_size=8,
    ocr_batch_size=16
)

# 방법 3: 불필요한 기능 비활성화
pipeline = PPStructureV3(
    use_seal_recognition=False,  # 도장 인식 불필요시
    use_chart_parsing=False,      # 차트 분석 불필요시
    use_doc_unwarping=False       # 왜곡 보정 불필요시
)

# 방법 4: 경량 모델 사용
pipeline = PaddleOCRVL()  # PP-StructureV3 대신
```

---

## 실전 예제 코드

### 예제 1: 기본 PDF to MD 변환기

```python
"""
기본 PDF to Markdown 변환기
PP-StructureV3 사용
"""

from paddleocr import PPStructureV3
import os

def pdf_to_markdown(pdf_path, output_dir='output'):
    """PDF를 Markdown으로 변환"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파이프라인 초기화
    print("모델 로딩 중...")
    pipeline = PPStructureV3(
        use_gpu=True,
        lang='korean',
        use_table_recognition=True,
        use_formula_recognition=True,
        show_log=False
    )
    
    # PDF 처리
    print(f"처리 중: {pdf_path}")
    output = pipeline.predict(input=pdf_path)
    
    # 결과 저장
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for idx, res in enumerate(output):
        # Markdown 저장
        md_path = os.path.join(output_dir, f'{base_name}_page_{idx+1}.md')
        res.save_to_markdown(save_path=output_dir)
        
        print(f"페이지 {idx+1}/{len(output)} 완료")
    
    print(f"완료! 결과: {output_dir}")

# 사용
if __name__ == "__main__":
    pdf_to_markdown('document.pdf')
```

### 예제 2: 배치 처리기

```python
"""
여러 PDF 파일 배치 처리
진행률 표시 포함
"""

from paddleocr import PPStructureV3
from pathlib import Path
from tqdm import tqdm
import os

def batch_convert_pdfs(pdf_dir, output_dir='output'):
    """디렉토리 내 모든 PDF를 Markdown으로 변환"""
    
    # PDF 파일 목록
    pdf_files = list(Path(pdf_dir).glob('*.pdf'))
    
    if not pdf_files:
        print("PDF 파일이 없습니다.")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파이프라인 초기화 (한 번만)
    print("모델 로딩 중...")
    pipeline = PPStructureV3(
        use_gpu=True,
        lang='korean',
        use_table_recognition=True,
        use_formula_recognition=True,
        show_log=False
    )
    
    # 배치 처리
    print(f"\n총 {len(pdf_files)}개 파일 처리 시작\n")
    
    for pdf_path in tqdm(pdf_files, desc="PDF 변환 중"):
        try:
            # PDF 처리
            output = pipeline.predict(input=str(pdf_path))
            
            # 결과 저장
            base_name = pdf_path.stem
            file_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            for idx, res in enumerate(output):
                res.save_to_markdown(save_path=file_output_dir)
            
            tqdm.write(f"✓ {pdf_path.name} 완료 ({len(output)} 페이지)")
            
        except Exception as e:
            tqdm.write(f"✗ {pdf_path.name} 실패: {str(e)}")
    
    print(f"\n완료! 결과: {output_dir}")

# 사용
if __name__ == "__main__":
    batch_convert_pdfs('pdf_folder', 'output')
```

### 예제 3: GUI 애플리케이션 (Tkinter)

```python
"""
PDF to Markdown 변환 GUI
Tkinter 사용
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from paddleocr import PPStructureV3
import threading
import os

class PDFtoMDConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF to Markdown 변환기")
        self.root.geometry("600x400")
        
        self.pipeline = None
        self.create_widgets()
    
    def create_widgets(self):
        # 파일 선택
        frame1 = tk.Frame(self.root, padx=20, pady=10)
        frame1.pack(fill=tk.X)
        
        tk.Label(frame1, text="PDF 파일:").pack(side=tk.LEFT)
        self.file_entry = tk.Entry(frame1, width=40)
        self.file_entry.pack(side=tk.LEFT, padx=10)
        tk.Button(frame1, text="선택", command=self.select_file).pack(side=tk.LEFT)
        
        # 출력 디렉토리
        frame2 = tk.Frame(self.root, padx=20, pady=10)
        frame2.pack(fill=tk.X)
        
        tk.Label(frame2, text="출력 폴더:").pack(side=tk.LEFT)
        self.output_entry = tk.Entry(frame2, width=40)
        self.output_entry.insert(0, "output")
        self.output_entry.pack(side=tk.LEFT, padx=10)
        
        # 옵션
        frame3 = tk.Frame(self.root, padx=20, pady=10)
        frame3.pack(fill=tk.X)
        
        self.use_gpu_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame3, text="GPU 사용", variable=self.use_gpu_var).pack(anchor=tk.W)
        
        self.use_table_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame3, text="표 인식", variable=self.use_table_var).pack(anchor=tk.W)
        
        self.use_formula_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame3, text="수식 인식", variable=self.use_formula_var).pack(anchor=tk.W)
        
        # 진행률
        frame4 = tk.Frame(self.root, padx=20, pady=10)
        frame4.pack(fill=tk.X)
        
        self.progress = ttk.Progressbar(frame4, mode='indeterminate')
        self.progress.pack(fill=tk.X)
        
        # 로그
        frame5 = tk.Frame(self.root, padx=20, pady=10)
        frame5.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(frame5, height=10, width=70)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 변환 버튼
        frame6 = tk.Frame(self.root, padx=20, pady=10)
        frame6.pack(fill=tk.X)
        
        self.convert_btn = tk.Button(frame6, text="변환 시작", 
                                      command=self.start_conversion,
                                      bg='#4CAF50', fg='white',
                                      font=('Arial', 12, 'bold'))
        self.convert_btn.pack()
    
    def select_file(self):
        filename = filedialog.askopenfilename(
            title="PDF 파일 선택",
            filetypes=[("PDF files", "*.pdf")]
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_conversion(self):
        pdf_path = self.file_entry.get()
        
        if not pdf_path:
            messagebox.showerror("오류", "PDF 파일을 선택하세요.")
            return
        
        if not os.path.exists(pdf_path):
            messagebox.showerror("오류", "파일이 존재하지 않습니다.")
            return
        
        # 버튼 비활성화
        self.convert_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self.convert)
        thread.start()
    
    def convert(self):
        try:
            pdf_path = self.file_entry.get()
            output_dir = self.output_entry.get()
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 모델 로딩
            self.log("모델 로딩 중...")
            self.pipeline = PPStructureV3(
                use_gpu=self.use_gpu_var.get(),
                lang='korean',
                use_table_recognition=self.use_table_var.get(),
                use_formula_recognition=self.use_formula_var.get(),
                show_log=False
            )
            
            # PDF 처리
            self.log(f"처리 중: {os.path.basename(pdf_path)}")
            output = self.pipeline.predict(input=pdf_path)
            
            # 결과 저장
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for idx, res in enumerate(output):
                res.save_to_markdown(save_path=output_dir)
                self.log(f"페이지 {idx+1}/{len(output)} 완료")
            
            self.log(f"\n완료! 결과: {output_dir}")
            messagebox.showinfo("완료", "변환이 완료되었습니다!")
            
        except Exception as e:
            self.log(f"\n오류 발생: {str(e)}")
            messagebox.showerror("오류", f"변환 실패:\n{str(e)}")
        
        finally:
            # 버튼 활성화
            self.progress.stop()
            self.convert_btn.config(state=tk.NORMAL)

# 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = PDFtoMDConverter(root)
    root.mainloop()
```

### 예제 4: 웹 서버 (Flask)

```python
"""
PDF to Markdown 웹 서비스
Flask 사용
"""

from flask import Flask, request, jsonify, send_file
from paddleocr import PPStructureV3
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# 디렉토리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 파이프라인 초기화 (서버 시작 시 한 번만)
print("모델 로딩 중...")
pipeline = PPStructureV3(
    use_gpu=True,
    lang='korean',
    use_table_recognition=True,
    use_formula_recognition=True,
    show_log=False
)
print("모델 로딩 완료!")

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF to Markdown 변환</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <h1>PDF to Markdown 변환기</h1>
        <div class="upload-box">
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf" required>
                <br><br>
                <button type="submit">변환 시작</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
    
    if file and file.filename.endswith('.pdf'):
        # 고유 ID 생성
        job_id = str(uuid.uuid4())
        
        # 파일 저장
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(pdf_path)
        
        try:
            # PDF 처리
            output = pipeline.predict(input=pdf_path)
            
            # 결과 저장
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            markdown_files = []
            for idx, res in enumerate(output):
                res.save_to_markdown(save_path=output_dir)
                markdown_files.append(f"page_{idx+1}.md")
            
            # 업로드 파일 삭제
            os.remove(pdf_path)
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'pages': len(output),
                'message': '변환 완료',
                'download_url': f'/download/{job_id}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': '잘못된 파일 형식'}), 400

@app.route('/download/<job_id>')
def download(job_id):
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    if not os.path.exists(output_dir):
        return jsonify({'error': '결과를 찾을 수 없습니다'}), 404
    
    # ZIP으로 압축
    import zipfile
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            zipf.write(file_path, file)
    
    return send_file(zip_path, as_attachment=True, download_name=f"converted_{job_id}.zip")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 예제 5: 고품질 변환 (전처리 포함)

```python
"""
고품질 PDF to Markdown 변환
이미지 전처리 포함
"""

from paddleocr import PPStructureV3
import fitz  # PyMuPDF
import cv2
import numpy as np
import os

class HighQualityPDFConverter:
    def __init__(self, use_gpu=True):
        print("모델 로딩 중...")
        self.pipeline = PPStructureV3(
            use_gpu=use_gpu,
            lang='korean',
            use_table_recognition=True,
            use_formula_recognition=True,
            use_doc_unwarping=True,
            show_log=False
        )
        print("모델 로딩 완료!")
    
    def preprocess_image(self, img):
        """이미지 전처리로 품질 향상"""
        # 그레이스케일
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def pdf_to_images(self, pdf_path, dpi=300):
        """PDF를 고해상도 이미지로 변환"""
        print(f"PDF 로딩: {pdf_path}")
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            print(f"페이지 {page_num+1}/{len(doc)} 변환 중...")
            page = doc[page_num]
            
            # 고해상도 렌더링
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # numpy 배열로 변환
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)
            
            # RGB 변환
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            images.append(img)
        
        return images
    
    def convert(self, pdf_path, output_dir='output', preprocess=True):
        """PDF를 Markdown으로 변환"""
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # PDF를 이미지로 변환
        images = self.pdf_to_images(pdf_path, dpi=300)
        
        # 각 페이지 처리
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for idx, img in enumerate(images):
            print(f"\n페이지 {idx+1}/{len(images)} 처리 중...")
            
            # 전처리
            if preprocess:
                img = self.preprocess_image(img)
            
            # 임시 저장
            temp_path = f'temp_page_{idx}.png'
            cv2.imwrite(temp_path, img)
            
            # OCR 처리
            output = self.pipeline.predict(input=temp_path)
            
            # 결과 저장
            for res in output:
                md_filename = f'{base_name}_page_{idx+1}.md'
                md_path = os.path.join(output_dir, md_filename)
                res.save_to_markdown(save_path=output_dir)
                print(f"저장: {md_filename}")
            
            # 임시 파일 삭제
            os.remove(temp_path)
        
        print(f"\n완료! 결과: {output_dir}")
        
        # 모든 MD 파일을 하나로 병합
        self.merge_markdown_files(output_dir, base_name)
    
    def merge_markdown_files(self, output_dir, base_name):
        """개별 페이지 MD 파일을 하나로 병합"""
        import glob
        
        md_files = sorted(glob.glob(os.path.join(output_dir, f'{base_name}_page_*.md')))
        
        if not md_files:
            return
        
        merged_path = os.path.join(output_dir, f'{base_name}_merged.md')
        
        with open(merged_path, 'w', encoding='utf-8') as outfile:
            for idx, md_file in enumerate(md_files):
                with open(md_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                    # 페이지 구분자 추가
                    if idx > 0:
                        outfile.write(f'\n\n---\n\n# Page {idx+1}\n\n')
                    
                    outfile.write(content)
        
        print(f"병합 완료: {merged_path}")

# 사용
if __name__ == "__main__":
    converter = HighQualityPDFConverter(use_gpu=True)
    converter.convert('document.pdf', output_dir='output', preprocess=True)
```

---

## 추가 참고 자료

### 공식 문서
- **PaddleOCR GitHub**: https://github.com/PaddlePaddle/PaddleOCR
- **공식 문서**: https://paddlepaddle.github.io/PaddleOCR/
- **공식 웹사이트**: https://www.paddleocr.ai

### 논문
- PaddleOCR 3.0 Technical Report: https://arxiv.org/abs/2507.05595
- PaddleOCR-VL Technical Report: https://arxiv.org/abs/2510.14528

### 커뮤니티
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **GitHub Discussions**: 사용법 질문 및 토론

### 모델 다운로드
- **HuggingFace**: https://huggingface.co/PaddlePaddle
- **모델 목록**: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md

---

## 라이선스

PaddleOCR는 **Apache 2.0 라이선스**로 제공됩니다.
- 상업적 사용 가능
- 수정 및 배포 가능
- 특허 사용 가능

---

## 버전 정보

**현재 버전**: 3.3.1 (2025년 10월)

**주요 변경사항**:
- PaddleOCR-VL 추가
- PP-OCRv5 다국어 지원 확대 (109개 언어)
- PP-StructureV3 성능 개선
- MCP 서버 지원

---

## 결론

이 문서는 PaddleOCR을 사용하여 PDF를 Markdown으로 변환하는 프로그램을 개발하기 위한 모든 정보를 담고 있습니다.

**핵심 요약**:
1. **PP-StructureV3**: 복잡한 문서, 높은 정확도 필요시
2. **PaddleOCR-VL**: 빠른 처리, 경량화 필요시
3. 전처리를 통한 품질 향상 가능
4. 배치 처리 및 병렬 처리로 성능 최적화
5. GUI, 웹 서버 등 다양한 형태로 구현 가능

**개발 시작을 위한 단계**:
1. PaddlePaddle 및 PaddleOCR 설치
2. 기본 예제로 동작 확인
3. 요구사항에 맞는 모델 선택
4. 전처리 및 최적화 적용
5. 원하는 인터페이스 구현 (CLI, GUI, 웹 등)

이 문서를 참고하여 성공적인 PDF to Markdown 변환 프로그램을 개발하시기 바랍니다!
