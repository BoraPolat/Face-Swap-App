=========================================================================
DEEP LEARNING-BASED FACESWAP - INSTALLATION GUIDE
=========================================================================

This project performs real-time, high-quality face swapping between photos
using Deep Learning methods.
It is optimized to run on NVIDIA RTX series graphics cards (CUDA).

REQUIREMENTS:
--------------
1. Operating System: Windows 10 or 11
2. Hardware: NVIDIA RTX Graphics Card (Recommended: RTX 3060 and above)
3. Software: Python 3.11.x version

-------------------------------------------------------------------------
STEP 1: BASIC SETUP
-------------------------------------------------------------------------
1. If you do not have Python installed, download the "python-3.11.9-amd64.exe" file.
   When installing, you MUST check the box at the bottom that says "Add python.exe to PATH".

2. C++ Compilers (Required for InsightFace):
   "Visual Studio C++ Build Tools" must be installed on your computer.
   If not installed, download the Visual Studio Installer and select/install
   "Desktop development with C++".

-------------------------------------------------------------------------
STEP 2: PREPARING THE PROJECT ENVIRONMENT
-------------------------------------------------------------------------
1. Enter the project folder.
2. Right-click inside the folder and open the Command Prompt (CMD) or Terminal.
3. Enter the following commands in order (Press Enter after each line):

   >> python -m venv venv
   >> venv\Scripts\activate

   (You should see (venv) appear at the beginning of the command line).

-------------------------------------------------------------------------
STEP 3: INSTALLING LIBRARIES
-------------------------------------------------------------------------
While (venv) is active in the terminal, copy and paste the following commands in order:

1. Basic Libraries:
   >> pip install numpy==1.26.4 opencv-python==4.10.0.84 PyQt5 scipy

2. AI Libraries:
   >> pip install insightface==0.7.3 onnxruntime-gpu

3. NVIDIA Acceleration Packages (CUDA 12):
   >> pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

-------------------------------------------------------------------------
STEP 4: CONFIGURING NVIDIA SETTINGS (IMPORTANT!)
-------------------------------------------------------------------------
DLL files must be moved so that Python can detect the NVIDIA libraries.
Run the script below to do this automatically for you:

Type this command in the terminal:
   >> python setup_nvidia.py

If this fails (e.g., warnings like "requirements not installed" appear), you must manually go to:
"C:\Users\YOUR_USER\Desktop\Faceswap\venv\Lib\site-packages\nvidia"
Open every folder located there, go into the "bin" folder inside each one, COPY the DLL files, and PASTE them into your "venv/scripts" folder.

If there were no issues: (You should see "SUCCESS: ... DLL files moved" on the screen).

-------------------------------------------------------------------------
STEP 5: MODEL FILE
-------------------------------------------------------------------------
Ensure that the "inswapper_128.onnx" file is located next to the "interface.py" file.
If it is missing, you cannot run the project.

-------------------------------------------------------------------------
STEP 6: RUNNING THE APPLICATION
-------------------------------------------------------------------------
Everything is now ready! To start the application:

   >> python interface.py

NOTES:
- When you click the "Swap Faces" button for the first time, there may be a 5-10 second delay.
  During this time, the AI model is being loaded onto the graphics card.
- Subsequent operations will happen in milliseconds.
- You can enable "Analysis Mode" in the interface to view the 3D mesh over the face.

Project_Folder/
│
├── interface.py          (Our Interface Code - Fixed version)
├── main.py               (Our Deep Learning Engine)
├── setup_nvidia.py       (The new auto-installer script we wrote)
├── README.txt            (The guide above)
├── inswapper_128.onnx    (500 MB model file)
└── (venv folder should be deleted while copying; the recipient will install it themselves using the README)