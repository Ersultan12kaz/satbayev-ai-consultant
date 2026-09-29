# Evaluation report

Questions: 300 (RU/KZ/EN), off-topic: 30, unanswerable: 15

## Experiment 1 - embedding models (dense retrieval)

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| Embedding A (LSA) | 0.573 | 0.833 | 0.903 | 0.686 |
| Embedding B (LSA + InfoNCE) | 0.523 | 0.843 | 0.900 | 0.663 |

## Experiment 2 - retrieval method

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | ms/query |
|---|---|---|---|---|---|---|---|
| Dense | 0.523 | 0.787 | 0.843 | 0.900 | 0.563 | 0.663 | 10.3 |
| BM25 (keyword) | 0.650 | 0.843 | 0.907 | 0.963 | 0.643 | 0.759 | 9.8 |
| Hybrid 70/30 | 0.637 | 0.790 | 0.847 | 0.927 | 0.611 | 0.728 | 19.8 |
| Hybrid + reranker | 0.677 | 0.813 | 0.873 | 0.927 | 0.645 | 0.759 | 27.2 |

## Experiment 3 - number of retrieved chunks

| Top-K | Recall@K | Precision@K |
|---|---|---|
| Top-3 | 0.813 | 0.644 |
| Top-5 | 0.873 | 0.645 |
| Top-10 | 0.927 | - |

## Experiment 4 - languages (full pipeline: hybrid + reranker, Top-5)

| Metric | Русский | Қазақша | English | All |
|---|---|---|---|---|
| Answer accuracy (answered with correct source) | 0.500 | 0.520 | 0.540 | 0.520 |
| Recall@1 | 0.580 | 0.720 | 0.670 | 0.657 |
| Recall@5 | 0.850 | 0.880 | 0.870 | 0.867 |
| MRR | 0.673 | 0.780 | 0.754 | 0.736 |
| Answer rate | 0.800 | 0.780 | 0.830 | 0.803 |
| Faithfulness | 1.000 | 0.994 | 1.000 | 0.998 |
| Citation accuracy | 0.596 | 0.654 | 0.627 | 0.625 |
| Language detection accuracy | 1.000 | 1.000 | 1.000 | 1.000 |
| Hallucination rate (unanswerable+off-topic answered) | 0.067 | 0.067 | 0.067 | 0.067 |
| Latency avg_ms | 66 | 73 | 66 | 68 |
| Latency p50_ms | 65 | 70 | 61 | 65 |
| Latency p95_ms | 87 | 111 | 91 | 100 |

## Refusals (hallucination guard)

| Set | Correctly refused |
|---|---|
| off_topic | 1.000 |
| unanswerable | 0.800 |

## By category

| Category | Recall@5 | MRR | Answer accuracy |
|---|---|---|---|
| admission | 0.767 | 0.532 | 0.367 |
| education | 0.933 | 0.678 | 0.467 |
| exams | 0.967 | 0.950 | 0.567 |
| scholarship | 0.500 | 0.347 | 0.200 |
| dormitory | 1.000 | 0.961 | 0.733 |
| academic_leave | 1.000 | 1.000 | 0.833 |
| transfer | 0.967 | 0.811 | 0.567 |
| internship | 0.800 | 0.451 | 0.267 |
| graduation | 0.800 | 0.800 | 0.500 |
| student_services | 0.933 | 0.828 | 0.700 |
