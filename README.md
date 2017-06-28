# Semantic Relevance Based Text Summarization Model
Code for "Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization"
## Requirements
* Tensorflow r1.0.1
* Python 3.5
* CUDA 8.0 (For GPU)
* [ROUGE](http://research.microsoft.com/~cyl/download/ROUGE-1.5.5.tgz)
## Data
The dataset in the paper is Large Scale Chinese Short Text Summarization ([LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html)).
To preprocess the data, please split the sentences into characters, and transform the characters into numbers (ids).
## Run
```bash
python3 train.py
```
## Cite
To use this code, please cite the following paper:<br><br>
Shuming Ma, Xu Sun, Jingjing Xu, Houfeng Wang, Wenjie Li and Qi Su. 
Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization. In proceedings of ACL.
[[pdf]](https://arxiv.org/pdf/1706.02459.pdf)
