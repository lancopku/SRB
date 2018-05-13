# Semantic Relevance Based Text Summarization Model
Code for "Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization"
The codes are also used for "A Semantic Relevance Based Neural Network for Text Summarization and Text Simplification"
## Requirements
* Tensorflow r1.0.1
* Python 3.5
* CUDA 8.0 (For GPU)
* [ROUGE](http://research.microsoft.com/~cyl/download/ROUGE-1.5.5.tgz)
## Data
The dataset in the paper is Large Scale Chinese Short Text Summarization [(LCSTS)](http://icrc.hitsz.edu.cn/Article/show/139.html).
To preprocess the data, please split the sentences into characters, and transform the characters into numbers (ids).
## Run
```bash
python3 MleTrain.py
```
## Cite
If you use this code for your research, please cite the paper this code is 
based on: <a href="https://arxiv.org/pdf/1706.02459.pdf">Improving Semantic Relevance for Sequence-to-Sequence Learning of
Chinese Social Media Text Summarization</a>:

@inproceedings{MaEA2017,
	author    = {Shuming Ma and
	Xu Sun and
	Jingjing Xu and
	Houfeng Wang and
	Wenjie Li and
	Qi Su},
	title     = {Improving Semantic Relevance for Sequence-to-Sequence Learning of
	Chinese Social Media Text Summarization},
	booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational
	Linguistics, {ACL} 2017, Vancouver, Canada, July 30 - August 4, Volume
	2: Short Papers},
	pages     = {635--640},
	year      = {2017}
}
