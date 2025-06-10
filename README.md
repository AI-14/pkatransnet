<h1 align="center">Enhancing radiology report generation: A prior knowledge-aware transformer network for effective alignment and fusion of multi-modal radiological data</h1>

<p align="center">
  <strong>
    <a href="https://scholar.google.com/citations?user=FeMCtswAAAAJ&hl=en">Amaan Izhar</a>,
    <a href="https://scholar.google.com.my/citations?user=IgUMlGcAAAAJ&hl=en">Norisma Idris</a>,
    <a href="https://scholar.google.com/citations?user=TyH59tkAAAAJ&hl=en">Nurul Japar</a>
  </strong>
  <br/><br/>
  <a href="">
    <img src="https://img.shields.io/badge/Read%20Paper-Elsevier-brightgreen?style=for-the-badge" alt="Read Paper Badge">
  </a>
</p>

---

## 📄 Abstract
<p align="justify">Medical imaging and their reports are essential in healthcare, providing crucial insights into internal structures and abnormalities for diagnosis and treatment. However, medical radiology report generation is time-consuming and further complicated by the shortage of expert radiologists. This paper presents a deep learning-based prior knowledge-aware transformer network designed to address the challenges of aligning and fusing medical images with textual data. Our method integrates medical signals of contextual biomedical entities and auxiliary medical knowledge embeddings extracted from reports with the visual features of radiology images to enhance alignment. Further, to tackle the fusion issue, we introduce Prior-KnowledgeAware-Report-Generator, a novel module with pre-normalization layers designed to improve training stability and efficiency, a prior knowledge-aware cross-attention mechanism to focus on multi-modal unified fused prior knowledge representation of radiology images and medical signals, and a feedforward layer utilizing the SwiGLU gated activation function, enhancing receptive field coverage. This ensures the model effectively incorporates and exploits prior medical knowledge to generate high-quality reports. We evaluate our method using standard natural language generation metrics on three widely used publicly available datasets - IUXRAY, COVCTR, and PGROSS. Our approach achieves average Bleu scores of 0.383, 0.647, and 0.191 for the respective datasets, outperforming existing state-of-the-art methods further evidenced by rigorous ablation and qualitative analysis conducted that taps into the contributions of various components of our model granting relevant clinical insights. The results demonstrate that the fusion of medical signals with radiology images significantly improves report accuracy and alignment with clinical findings, providing valuable assistance to radiologists.</p>

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Model Architecture" />
</p>

---

## 📚 Citation

If you find this work useful, please consider citing our paper and giving this repository a ⭐:

```bibtex
@article{izhar2025pkatransnet,
  title={Enhancing radiology report generation: A prior knowledge-aware transformer network for effective alignment and fusion of multi-modal radiological data},
  author={Izhar, Amaan and Idris, Norisma and Japar, Nurul},
  journal={Image and Vision Computing},
  year={2025}
}
```

---

## 🛠️ Reproducibility

### ✅ System Requirements

| Component        | Specification                       |
|------------------|-------------------------------------|
| OS               | Ubuntu 22.04                        |
| GPU              | ≥ 6 GB VRAM                         |
| RAM              | ≥ 16 GB                             |
| Disk Space       | ≥ 20 GB                             |
| Env Modules      | Miniconda                           |
| Dependencies     | CUDA ≥ 12.1                         |

---

### 📦 Environment Setup
1. Run:
```bash
# Clone the repository
git clone https://github.com/AI-14/pkatransnet.git
cd pkatransnet

# Create and activate conda environment
conda create -n env python=3.10 --yes
conda activate env

# Install dependencies
conda install pip
pip install -r requirements.txt
```

---

### 🔬 Dataset Setup & Running Experiments

#### IUXRAY

1. Create the following directory structure:
```bash
datasets
|-- iuxray
    |-- images
    |-- reports
```
2. Download from [IUXRAY](https://openi.nlm.nih.gov/).
3. Place image files under `datasets/iuxray/images` and reports under `datasets/iuxray/reports`.
4. Run:
```bash
source scripts/iuxray/preprocess.sh
source scripts/iuxray/fg.sh
```

---

#### COVCTR

1. Create the following directory structure:
```bash
datasets
|-- covctr
    |-- images
    |-- reports.csv
```
2. Download from [COVCTR](https://github.com/mlii0117/COV-CTR).
3. Place image files under `datasets/covctr/images`.
4. Rename `reports_ZH_EN.csv` → `reports.csv` and move the report file to `datasets/covctr/reports.csv`.
5. Run:
```bash
source scripts/covctr/preprocess.sh
source scripts/covctr/fg.sh
```

---

#### PGROSS

1. Create the following directory structure:
```bash
datasets
|-- pgross
    |-- images
    |-- captions.json
    |-- tags.json
    |-- def.json
```
2. Download from [PGROSS](https://github.com/wang-zhanyu/medical-reports-datasets).
3. Remove the following files: `train_images.tsv`, `test_images.tsv`, `peir_gross.tsv`.
4. Rename `peir_gross_captions.json` → `captions.json` and `peir_gross_tags.json` → `tags.json`.
5. Create a `def.json` file with the following content:
      <details>
      <summary><strong>Click to view full <code>def.json</code> content</strong></summary>
  
      ```json
      {
        "cardiovascular": "The term cardiovascular refers to the heart (cardio) and the blood vessels (vascular).",
        "nervous": "The nervous system is a complex network of nerves and nerve cells (neurons) that carry signals or messages to and from the brain and spinal cord to different parts of the body.",
        "respiratory": "Respiratory system is the organs and structures in your body that allow you to breathe.",
        "gastrointestinal": "The organs that make up your gastrointestinal tract, in the order that they are connected, include your mouth, esophagus, stomach, small intestine, large intestine, and anus.",
        "urinary": "The urinary system includes your kidneys, ureters, bladder and urethra. This system filters your blood, removing waste and excess water.",
        "hepatobiliary": "The hepatobiliary system is made up of the liver and biliary tract, which are essential for digestion.",
        "endocrine": "The endocrine system is a network of glands and organs that produce hormones and release them into the bloodstream.",
        "musculoskeletal": "The musculoskeletal system (locomotor system) is a human body system that provides our body with movement, stability, shape, and support.",
        "hematologic": "The hematology system consists of the blood and the bone marrow that create the cellular elements of the blood, as well as accessory organs, including the spleen and the liver.",
        "female reproductive": "The female reproductive system is a complex system of internal and external organs that work together to enable reproduction, pregnancy, and childbirth.",
        "male reproductive": "The male reproductive system is made up of internal and external organs that are essential for reproduction, sexual function, and urination.",
        "head": "The upper portion of the body, consisting of the skull with its coverings and contents, including the lower jaw.",
        "extremities": "A limb of the body, such as the arm or leg.",
        "skin": "The skin is the body's largest organ and primary protective barrier, covering the entire external surface.",
        "body": "The body is the physical material of an organism, made up of living cells and extracellular materials.",
        "oral": "The oral cavity, or more commonly known as the mouth or buccal cavity, serves as the first portion of the digestive system.",
        "abdomen": "The part of the body that contains all of the structures between the thorax (chest) and the pelvis, and is separated from the thorax via the diaphragm.",
        "lymphatic": "The lymphatic system is a network of organs, tissues, vessels, and capillaries that transport a fluid called lymph from body tissues back into the bloodstream.",
        "pancreas": "The pancreas is a long, flat gland in the abdomen that has both digestive and endocrine functions.",
        "thorax": "The area of the body between the neck and the abdomen.",
        "breast": "The breast is a glandular organ on the chest, also known as the mammary gland, that's made up of connective tissue, fat, and breast tissue.",
        "brain": "The organ inside the head that controls all body functions of a human being.",
        "eye": "Human eye, specialized sense organ in humans that is capable of receiving visual images, which are relayed to the brain."
      }
      ```
  
  </details>

6. Run:
```bash
source scripts/pgross/preprocess.sh
source scripts/pgross/fg.sh
```

---

## 🧹 Clean Up

1. Run:
```bash
cd ..
conda deactivate
conda remove --name env --all
rm -r pkatransnet
```

---

> ⚠️ **Note:** Due to the small size of some datasets and the sensitivity to random seeds, slight performance variance may occur. For more stable comparisons, consider running multiple trials with different seeds.
