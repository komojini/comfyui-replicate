import os

CUSTOM_NODE_REPOS = [
    "https://github.com/ltdrdata/ComfyUI-Manager.git",
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
    "https://github.com/WASasquatch/was-node-suite-comfyui.git",
    "https://github.com/Gourieff/comfyui-reactor-node.git",
    "https://github.com/rgthree/rgthree-comfy.git",
    "https://github.com/mav-rik/facerestore_cf.git",
]


for custom_node_repo in CUSTOM_NODE_REPOS:
    os.system(f"cd ComfyUI/custom_nodes && git clone {custom_node_repo}")