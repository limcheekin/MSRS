import modal

FORCE_BUILD = False  # Set to True to force a new image build
LLAMA_CPP_RELEASE = "b6739"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"



image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12", force_build=FORCE_BUILD)
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp"
                f" && cd llama.cpp && git checkout tags/{LLAMA_CPP_RELEASE}")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-server"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

model_cache = modal.Volume.from_name("qwen3-cache", create_if_missing=True)
cache_dir = "/root/.cache/qwen3"
gguf_model_path=f"{cache_dir}/Qwen3-4B-Thinking-2507-GGUF/Qwen3-4B-Thinking-2507-Q4_K_M.gguf"
app = modal.App("qwen3-4b-thinking")

@app.function(
    image=image, 
    volumes={cache_dir: model_cache},
    gpu="T4", 
    timeout=30 * MINUTES,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(80, startup_timeout=20 * MINUTES)
def llama_server():
    import subprocess

    command = [
        "/llama.cpp/llama-server",     # Server binary
        "-m", gguf_model_path,
        "--n-gpu-layers", "99",        # Offload 40 out of 49 layers to GPU for T4
        "-c", "65536",                # Max context size 262,144
        "-sm", "row",
        # REF: https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html#llama-server
        "--temp", "0.6",
        "--top-k", "20",
        "--top-p", "0.95",
        "--min-p", "0",
        "--presence_penalty", "1.0",    # presence_penalty = 0.0 to 2.0
        #"--cache-type-k", "q8_0",      # Use 8-bit KV cache for T4
        #"--cache-type-v", "q8_0",        
        #"--threads", "-1",             # Number of CPU threads (adjust per Modal instance vCPUs)   
        "-np", "1",                    # Number of parallel sequences
        "--port", "80",                # Server port
        "--host", "0.0.0.0",           # Server host
        # "--reasoning-format", "deepseek", # Enabled for Qwen3 thinking model if needed 
        "--verbose",                   # Enable verbose logging
        #"-fa",                         # Enable flash attention for V cache quantization
        "--no-context-shift",
        "--jinja",
    ]
    
    print("🦙 running commmand:", command, sep="\n\t")
    subprocess.Popen(command)
