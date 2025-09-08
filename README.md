# Scheduler-AI-assistant
Concept repo for the usage of an AI as an assistant chat bot for the the project manger, construction mangers and stakeholders to follow up on the planned and actual project status.


# Local LLM Fine-Tuning Guide (assumed 8 GB GPU)

## Requirements

- **OS**: Windows 10/11 (Admin rights required)  
- **GPU**: The Higher the GPU the better
- **RAM**: 16 GB+ (more is better)  
- **Disk**: 25–40 GB free (model + caches + outputs)  
- **Time**: A few hours for installs + training time (varies by dataset size) -- a simple excel was used as a proof case  

> ⚠️ If `bitsandbytes` (4-bit loader) fails on native Windows, switch to **WSL2 (Ubuntu)** *ASK AI Tool*.  
This avoids Windows build issues while still running locally.

---

## 1) Install the Basics (once)

- [Visual Studio Code](https://code.visualstudio.com/)  
  (Editor — open it when creating/editing files)  

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)  
  *(During install, tick **“Add Python to PATH.””*)  

- [Git](https://git-scm.com/download/win)  

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)  

- **NVIDIA driver** up to date (via GeForce Experience or NVIDIA site).  
  *(No need to install CUDA toolkit; PyTorch will fetch the runtime.)*  

---

## 2) Create Project & Python Environment

Open **Anaconda Prompt** (or VS Code → Terminal): each line at a time,

```powershell
mkdir %USERPROFILE%\my_project
cd %USERPROFILE%\my_project

conda create -n llm_local python=3.10 -y
conda activate llm_local

## Install PyTorch & Libraries

```bash
# PyTorch (CUDA 12.1 wheel; works if your system driver is 12.x-compatible)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core LLM tooling
pip install transformers datasets accelerate peft trl huggingface_hub

# Quantization + Excel + utilities
pip install bitsandbytes openpyxl pandas tqdm evaluate gradio

# (Optional) Axolotl trainer (we'll use it)
pip install axolotl


---

## 3) Put your Excel in the project folder


Place your file as:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   C:\Users\\my_project\construction.xlsx   `

It must have headers exactly:**ActivityID, ActivityName, Predecessor, Successor**