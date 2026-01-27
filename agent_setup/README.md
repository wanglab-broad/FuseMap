# Tutorial on how to install and use FuseMap-agent interface

Note: FuseMap tutorial is here: https://fusemap.readthedocs.io/en/latest/


### Part A. How to install:

1. First download the code 
`git clone https://github.com/wanglab-broad/FuseMap.git`

2. Download 
- 2.1.Download pretrained foundation model weights at https://drive.google.com/drive/u/2/folders/1auybpmekWuW_G-7YPloJr-B96qiT1nFS. Put the molCCF folder under `FuseMap/`
- 2.2 Download atlas molCCF data at https://drive.google.com/file/d/15LIkQTridS_ATwDy6dejIdzbMm39sEv3/view?usp=sharing. Put under `FuseMap/agent_setup/atlas_data/`

3. If you need FuseMap to analyze certain datasets, put anndata files under `path/to/your/input/anndata`.
- The example datasets of disease, aging, embryo, species are in https://drive.google.com/drive/folders/1ZRIbHTd9TAjmtr3V6WLkvrY4iLF5SH_U?usp=drive_link

4. Run in terminal:
```
cd FuseMap
conda create -n fusemap python=3.10.16
conda activate fusemap
pip install fusemap
streamlit run app.py
```

5. Open in the browser:
`http://localhost:xxxx`

You should see the interface running smoothly.



### Part B. How to use in the interface:

1. Input OPENAI API key

2. (Optional) Input Base URL
- Leave blank if default


4. Input Tavily API
- This is free, get one in https://www.tavily.com/

4. Example input:
`How do the cell state changes in mouse hippocampus with Alzheimer's disease? I have measured spatial transcriptomics datasets of mouse brain with two Alzheimer's disease model at path '/path/to/data'. Save results at 'path/to/output'. Help me analyze the cell types.`

5. All output results are saved under `path/to/output`.
- To view html figures on remote ssh, you can run: python3 -m http.server 8000

Final analyzed anndata file is in `path/to/output/data/annotated_user_data.h5ad`.



