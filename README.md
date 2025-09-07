# Guess & Check Plus (GC+) Codes

This repository contains Python implementations and experiments for **Guess & Check Plus (GC+) codes**, a family of systematic short blocklength codes designed to correct random **edit errors** (deletions, insertions, and substitutions). GC+ codes are motivated by the requirements of **DNA data storage**, where short DNA sequences (oligos) must be robust to noise introduced during synthesis, storage, and sequencing.

The source accompanies the paper:

> Serge Kas Hanna, *GC+ Code: A Systematic Short Blocklength Code for Correcting Random Edit Errors in DNA Storage*, IEEE (2025).  
> [arXiv preprint](https://arxiv.org/abs/2401.12345)

---

## Features
- **Binary GC+ implementation**  
  Encoding, decoding, and performance analysis under i.i.d. and localized edit errors.
- **DNA GC+ implementation**  
  Integration into a DNA storage pipeline with binary-to-quaternary mapping, outer Reed–Solomon codes, and edit error channels.
- **Theoretical analysis**  
  Evaluation of decoding error probabilities using analytical expressions.
- **Simulation scripts**  
  Reproduce the figures in the paper for frame error rate (FER) vs. edit probability and localized errors.

---

## Repository Structure
```
src/
 ├── GCP binary/              # Binary-domain GC+ code
 │   ├── GCP_Encode_binary.py
 │   ├── GCP_Decode_binary.py
 │   ├── GCP_theoretical_binary.py
 │   ├── binary_channel.py
 │   ├── burst_patterns.py
 │   ├── main_FERvsPe_binary.py
 │   └── main_FERvsW_localized_binary.py
 │
 └── GCP dna/                 # DNA-domain GC+ code
     ├── GCP_Encode_DNA.py
     ├── GCP_Decode_DNA.py
     ├── GCP_theoretical_DNA.py
     ├── DNA_channel.py
     ├── Outer_Encode.py      # Outer Reed–Solomon layer
     ├── main_FERvsPe_DNA.py
     ├── main_FERvsW_localized_DNA.py
     └── main_Inner+Outer.py
```

---

## Installation
```bash
git clone https://github.com/your-username/GCP.git
cd GCP
pip install -r requirements.txt
```

---

## Usage
### Binary simulations
Run frame error rate experiments under i.i.d. edits:
```bash
python src/GCP\ binary/main_FERvsPe_binary.py
```

Run localized edit experiments:
```bash
python src/GCP\ binary/main_FERvsW_localized_binary.py
```

### DNA simulations
Run quaternary-domain experiments:
```bash
python src/GCP\ dna/main_FERvsPe_DNA.py
```

Run end-to-end inner + outer code simulations:
```bash
python src/GCP\ dna/main_Inner+Outer.py
```

---

## Citation
If you use this code in your research, please cite:
```bibtex
@article{kas2025gcplus,
  author    = {Serge Kas Hanna},
  title     = {GC+ Code: A Systematic Short Blocklength Code for Correcting Random Edit Errors in DNA Storage},
  journal   = {IEEE Transactions on Information Theory},
  year      = {2025},
}
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
